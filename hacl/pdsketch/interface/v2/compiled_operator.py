import jacinle
import torch
from typing import Optional, Union, Tuple, Sequence, Mapping

from hacl.pdsketch.interface.v2.value import BOOL, ObjectType, Value, Variable
from hacl.pdsketch.interface.v2.optimistic import RELAX_MAGIC_NUMBER, relaxed_value_id, OptimisticValueContext, is_optimistic_value, RelaxedBackwardContext, RelaxedExecutionContext
from hacl.pdsketch.interface.v2.state import State
from hacl.pdsketch.interface.v2.expr import ExpressionExecutionContext, Expression
from hacl.pdsketch.interface.v2.expr import ConstantExpression, VariableExpression, ValueOutputExpression
from hacl.pdsketch.interface.v2.expr import FeatureApplication, BoolOp, BoolOpType, AssignOp, is_simple_bool
from hacl.pdsketch.interface.v2.domain import Domain, Precondition, Effect, OperatorApplier

__all__ = ['CompiledConjunction', 'CompiledAssignments', 'compile_conjunction', 'compile_effects', 'CompiledOperatorApplier', 'compile_operator']


class CompiledExpression(object):
    def fforward(self, state: State):
        raise NotImplementedError()

    def fforward_relaxed(self, ctx: RelaxedExecutionContext, state: State):
        raise NotImplementedError()


class CompiledConjunction(ValueOutputExpression, CompiledExpression):
    def __init__(self, conjunctions: Tuple[Tuple[str, int, Tuple[int, ...]], ...]):  # name, negation_flag, args
        self.conjunctions = conjunctions

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        assert isinstance(ctx.state, State)
        return self.fforward(ctx.state)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Value:
        assert isinstance(ctx.state, State)
        return self.fforward_relaxed(ctx.relaxed_context, ctx.state)

    def fforward(self, state: State) -> Value:
        values = tuple((state.tensors[name][args].item() ^ neg for name, neg, args in self.conjunctions))
        return Value(BOOL, [], torch.tensor(all(values), dtype=torch.int64), quantized=True)

    def fforward_relaxed(self, ctx: RelaxedExecutionContext, state: State) -> Value:
        raw_values = tuple((state.tensors[name][args].item() for name, _, args in self.conjunctions))
        values = tuple((value ^ neg for value, (_, neg, _) in zip(raw_values, self.conjunctions)))
        if any(tuple((value == 0 for value in values))):
            return Value(BOOL, [], torch.tensor(0, dtype=torch.int64), quantized=True)
        if any(tuple(is_optimistic_value(value) for value in values)):
            value = Value(BOOL, [], torch.tensor(RELAX_MAGIC_NUMBER, dtype=torch.int64), quantized=True)
            value.set_backward_function(self._backward_relaxed, value, state)
            return value
        return Value(BOOL, [], torch.tensor(1, dtype=torch.int64), quantized=True)

    def _backward_relaxed(self, ctx: RelaxedBackwardContext, state):
        for name, _, args in self.conjunctions:
            v = state.tensors[name][args].item()
            if is_optimistic_value(v):
                ctx.rv_set.add(relaxed_value_id(v))

    def __str__(self):
        return type(self).__name__ + str(self.conjunctions)


class CompiledAssignments(Expression, CompiledExpression):
    def __init__(self, assignments: Tuple[Tuple[str, Tuple[int, ...], int], ...]):  # name, args, value
        self.assignments = assignments

    def _forward(self, ctx: ExpressionExecutionContext):
        assert isinstance(ctx.state, State)
        self.fforward(ctx.state)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext):
        assert isinstance(ctx.state, State)
        self.fforward_relaxed(ctx.relaxed_context, ctx.state)

    def fforward(self, state: State):
        for name, args, value in self.assignments:
            state.tensors[name][args] = value

    def fforward_relaxed(self, ctx: RelaxedExecutionContext, state: State):
        for name, args, value in self.assignments:
            v = state.tensors[name][args].item()
            if not is_optimistic_value(v) and v != value:
                state.tensors[name][args] = RELAX_MAGIC_NUMBER + ctx.op_identifier

    def __str__(self):
        return type(self).__name__ + str(self.assignments)


def compile_conjunction(domain: Domain, state: State, preconditions: Union[BoolOp, Sequence[Union[Precondition, ValueOutputExpression]]], var2const: Optional[Mapping[str, str]] = None):
    if isinstance(preconditions, BoolOp):
        assert preconditions.bool_op_type is BoolOpType.AND
        return compile_conjunction(domain, state, preconditions.arguments, var2const)

    # no need to set var2const to {} if None.
    conjunctions = list()
    for pre in preconditions:
        if isinstance(pre, Precondition):
            pre = pre.bool_expr
        assert is_simple_bool(pre)

        neg = False
        if isinstance(pre, BoolOp) and pre.bool_op_type is BoolOpType.NOT:
            neg = True
            pre = pre.arguments[0]

        assert isinstance(pre, FeatureApplication)
        name = pre.feature_def.name
        args = tuple((state.get_typed_index(
            var2const[arg.name] if isinstance(arg, VariableExpression) else arg.name
        ) for arg in pre.arguments))
        conjunctions.append((name, neg, args))
    return CompiledConjunction(tuple(conjunctions))


def compile_effects(domain: Domain, state: State, effects: Sequence[Union[Effect, AssignOp]], var2const: Mapping[str, str]):
    assignments = list()
    for eff in effects:
        if isinstance(eff, Effect):
            eff= eff.assign_expr
        if not isinstance(eff, AssignOp):
            raise NotImplementedError('Unsupported effect op (only supports AssignOp): {}.'.format(eff))
        name = eff.feature.feature_def.name
        args = tuple((state.get_typed_index(
            var2const[arg.name] if isinstance(arg, VariableExpression) else arg.name
        ) for arg in eff.feature.arguments))
        assert isinstance(eff.value, ConstantExpression)
        value = eff.value.value.item()
        assignments.append((name, args, value))
    return CompiledAssignments(tuple(assignments))


class CompiledOperatorApplier(OperatorApplier):
    def __init__(self, op_applier: OperatorApplier, precondition: CompiledConjunction, effect: CompiledAssignments):
        super().__init__(op_applier.operator, *op_applier.arguments)
        self.precondition = precondition
        self.effect = effect

    def apply_precondition(self, state: State, optimistic_context: Optional[OptimisticValueContext] = None, relaxed_context: Optional[RelaxedExecutionContext] = None) -> bool:
        if relaxed_context is not None:
            pred_value = self.precondition.fforward_relaxed(relaxed_context, state)
        else:
            pred_value = self.precondition.fforward(state)
        rv = pred_value.item()
        if is_optimistic_value(rv):
            if relaxed_context is not None:
                relaxed_context.add_backward_value(pred_value)
        else:
            if rv < 0.5:
                return False
        return True

    def apply_effect(self, state: State, optimistic_context: Optional[OptimisticValueContext] = None, relaxed_context: Optional[RelaxedExecutionContext] = None, clone: Optional[bool] = True) -> State:
        if clone:
            state = state.clone()
        if relaxed_context is not None:
            self.effect.fforward_relaxed(relaxed_context, state)
        else:
            self.effect.fforward(state)
        return state


@jacinle.deprecated
def compile_operator(op: OperatorApplier, state: State):
    """compile_operator has been deprecated. Use pds.strips.StripsTranslator to compile the problem into strips instead."""
    for arg in op.operator.arguments:
        assert isinstance(arg, Variable) and isinstance(arg.type, ObjectType)
    var2const = {
        arg.name: argv for arg, argv in zip(op.operator.arguments, op.arguments)
    }
    return CompiledOperatorApplier(
        op,
        precondition=compile_conjunction(op.operator.domain, state, op.operator.preconditions, var2const),
        effect=compile_effects(op.operator.domain, state, op.operator.effects, var2const)
    )
