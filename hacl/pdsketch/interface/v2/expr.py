import itertools
import contextlib
from abc import ABC
from typing import Optional, Union, Tuple, Sequence, List, Mapping, Dict

import torch
import jacinle
import jactorch
from jacinle.utils.enum import JacEnum
from jacinle.utils.printing import indent_text
from jacinle.utils.defaults import wrap_custom_as_default, gen_get_default
from .value import ObjectType, ValueType, NamedValueType, BOOL, INT64, RUNTIME_BINDING, is_intrinsically_quantized
from .value import QINDEX, Variable, StringConstant, QuantizedTensorValue, Value, MixedValue
from .optimistic import RELAX_MAGIC_NUMBER, is_optimistic_value, OptimisticConstraint, EqualOptimisticConstraint, OptimisticValueContext, cvt_opt_value, RelaxedExecutionContext
from .state import StateLike, SingleStateLike, MixedState

__all__ = [
    'FunctionDef', 'FeatureDef', 'PredicateDef',
    'ExpressionDefinitionContext', 'ExpressionExecutionContext', 'get_definition_context', 'get_execution_context',
    'Expression', 'VariableExpression', 'ObjectConstantExpression', 'ConstantExpression',
    'ValueOutputExpression', 'is_value_output_expression',
    'forward_args', 'has_partial_execution_value', 'expand_argument_values',
    'FunctionApplication', 'FeatureApplication', 'PredicateApplication', 'ExternalFunctionApplication', 'is_feature_application', 'is_external_function_application',
    'DeicticSelectOp',
    'BoolOp', 'BoolOpType', 'AndOp', 'OrOp', 'NotOp', 'is_constant_bool_expr', 'is_simple_bool', 'split_simple_bool', 'get_simple_bool_def', 'is_and_expr', 'is_or_expr', 'is_not_expr',
    'QuantificationOp', 'QuantifierType', 'ForallOp', 'ExistsOp', 'is_forall_expr', 'is_exists_expr',
    'FeatureEqualOp',
    'VariableAssignmentExpression', 'is_variable_assignment_expression',
    'AssignOp', 'ConditionalAssignOp', 'DeicticAssignOp',
    'flatten_expression', 'iter_exprs'
]


class FunctionDef(object):
    def __init__(self, name: str, output_type: Union[ValueType, List[ValueType]], arguments: Sequence[Union[ValueType, Variable]], **kwargs):
        self.name = name
        self.output_type = output_type
        self.arguments = arguments
        self.kwargs = kwargs

        self._check_arguments()

    def _check_arguments(self):
        pass

    def __str__(self) -> str:
        arguments = ', '.join([str(arg) for arg in self.arguments])
        return type(self).__name__ + '::' f'{self.name}({arguments}) -> {self.output_type}'

    __repr__ = jacinle.repr_from_str


class FeatureDef(FunctionDef):
    def __init__(self, name: str, feature_type: ValueType, arguments: Sequence[Variable], expr: Optional['ValueOutputExpression'] = None, derived: Optional[bool] = False, cacheable: Optional[bool] = None):
        super().__init__(name, feature_type, arguments)
        self.expr = expr
        self.group = None
        self.derived = derived

        if cacheable is None:
            self.cacheable = self.guess_is_cacheable()
        else:
            self.cacheable = cacheable

        self.static = False
        self._check_arguments_cacheable()

        from .ao_discretization import AOFeatureCodebook
        self.ao_discretization: Optional[AOFeatureCodebook] = None  # for AODiscretization

    def set_group(self, group: str):
        self.group = group

    def mark_static(self, flag=True):
        self.static = flag

    def guess_is_cacheable(self) -> bool:
        """Return whether the feature function can be cached. Specifically, if it contains only "ObjectTypes" as arguments, it
        can be statically evaluated."""
        for arg_def in self.arguments:
            if isinstance(arg_def.type, ValueType):
                return False
        return True

    def _check_arguments(self):
        for arg_def in self.arguments:
            assert isinstance(arg_def, Variable)

    def _check_arguments_cacheable(self):
        if self.cacheable:
            for arg_def in self.arguments:
                assert isinstance(arg_def.type, ObjectType)
        else:
            for arg_def in self.arguments:
                assert isinstance(arg_def.type, (ObjectType, ValueType))

    def __str__(self) -> str:
        cacheable_str = 'cacheable' if self.cacheable else 'non-cacheable'
        derived_str = ', derived' if self.derived else ''
        static_str = ', static' if self.static else ''
        arguments = ', '.join([str(arg) for arg in self.arguments])
        fmt = type(self).__name__ + '::' f'{self.name}[{cacheable_str}{derived_str}{static_str}]({arguments}) -> {self.output_type}'
        if self.expr is not None:
            fmt += ' {\n'
            fmt += '  ' + str(self.expr)
            fmt += '\n}'
        else:
            fmt += ' {}'
        return fmt


class PredicateDef(FeatureDef):
    def __init__(self, name: str, arguments: Sequence[Variable]):
        super().__init__(name, BOOL, arguments)


class ExpressionDefinitionContext(object):
    """The context for defining a PDSketch expression. During definition, it will only keep track
    of the type of the object."""

    def __init__(self, *variables: Variable, domain=None, scope=None, precondition_constraints=None, effect_constraints=None, is_effect_definition=None):
        self.variables = list(variables)
        self.variable_name2obj = {v.name: v for v in self.variables}
        self.domain = domain
        self.scope = scope
        self.precondition_constraints = precondition_constraints
        self.effect_constraints = effect_constraints

        if is_effect_definition is None:
            is_effect_definition = self.effect_constraints is not None and len(self.effect_constraints) > 0
        self.is_effect_definition_stack = [is_effect_definition]

        self.name_counter = itertools.count()

    def __getitem__(self, variable_name) -> 'VariableExpression':
        if variable_name == '??':
            return VariableExpression(Variable('??', RUNTIME_BINDING))
        if variable_name not in self.variable_name2obj:
            raise ValueError('Unknown variable: {}; available variables: {}.'.format(variable_name, self.variables))
        return VariableExpression(self.variable_name2obj[variable_name])

    def generate_random_named_variable(self, type) -> 'Variable':
        """Generate a variable expression with a random name.

        This utility is useful in "flatten_expression". See the doc for that
        function for details.
        """
        name = '_t' + str(next(self.name_counter))
        return Variable(name, type)

    @contextlib.contextmanager
    def new_arguments(self, *args: Variable):
        """Adding a list of new variables."""
        for arg in args:
            assert arg.name not in self.variable_name2obj, 'Variable name {} already exists.'.format(arg.name)
            self.variables.append(arg)
            self.variable_name2obj[arg.name] = arg
        yield self
        for arg in reversed(args):
            self.variables.pop()
            del self.variable_name2obj[arg.name]

    @wrap_custom_as_default
    def as_default(self):
        yield self

    def check_precondition(self, function_def: FunctionDef, scope_string: str):
        def check():
            if self.precondition_constraints is None:
                return True
            return self.domain.feature_in_group(function_def, self.precondition_constraints)
        if not check():
            raise KeyError(f'Cannot access {function_def} during the definition of {scope_string}: allowed feature/predicate groups are: {", ".join(self.precondition_constraints)}.')

    def check_effect(self, function_def: FunctionDef, scope_string: str):
        def check():
            if self.effect_constraints is None:
                return True
            return self.domain.feature_in_group(function_def, self.effect_constraints)
        if not check():
            raise KeyError(f'Cannot change the value of {function_def} during the definition of {scope_string}: allowed feature/predicate groups are: {", ".join(self.effect_constraints)}.')

    @property
    def is_effect_definition(self):
        return self.is_effect_definition_stack[-1]

    @contextlib.contextmanager
    def mark_is_effect_definition(self, value=True):
        self.is_effect_definition_stack.append(value)
        yield
        self.is_effect_definition_stack.pop()


class ExpressionExecutionContext(object):
    def __init__(
        self,
        domain, state: StateLike,
        bounded_variables: Mapping[str, Mapping[str, Union[int, slice, torch.Tensor]]],
        optimistic_context: Optional[OptimisticValueContext] = None,
        relaxed_context: Optional[RelaxedExecutionContext] = None
    ):
        """The expression execution context.

        Args:
            domain (Domain): The current domain.
            state (State): The input state.
            bounded_variables (Mapping[str, Mapping[str, int]]): A mapping for bounded variables. Stored in the following form:

                ```
                {
                    'item': {'?obj1': 0}
                    'location': {'?loc': 1}
                }
                ```

                The key to the outer mapping is the typename. The key to the inner mapping is the variable type.
                The values are "typed_index".
        """
        self.domain = domain
        self.state = state
        self.bounded_variables = bounded_variables
        self.optimistic_context = optimistic_context
        self.relaxed_context = relaxed_context

    def __str__(self) -> str:
        flags = '[relaxed]' if self.is_relaxed_execution else ''
        fmt = f'ExpressionExecutionContext{flags}(domain={self.domain.name},\n'
        # fmt += f'  state={indent_text(str(self.state)).lstrip()},\n'
        fmt += f'  bounded_variables={indent_text(str(self.bounded_variables)).lstrip()},\n'
        fmt += f'  optimistic_context={indent_text(str(self.optimistic_context)).lstrip()}\n'
        fmt += ')'
        return fmt

    __repr__ = jacinle.repr_from_str

    @property
    def is_optimistic_execution(self) -> bool:
        return self.optimistic_context is not None

    @property
    def is_relaxed_execution(self) -> bool:
        return self.relaxed_context is not None

    @property
    def is_mixed_state(self) -> bool:
        return isinstance(self.state, MixedState)

    @property
    def value_quantizer(self):
        return self.domain.value_quantizer

    def get_external_function(self, name):
        return self.domain.get_external_function(name)

    @contextlib.contextmanager
    def new_bounded_variables(self, variable2index: Mapping[Variable, Union[int, slice]]):
        for arg, index in variable2index.items():
            assert arg.type.typename not in self.bounded_variables or arg.name not in self.bounded_variables[arg.type.typename], 'Variable name {} already exists.'.format(arg.name)
            if arg.type.typename not in self.bounded_variables:
                self.bounded_variables[arg.type.typename] = dict()
            self.bounded_variables[arg.type.typename][arg.name] = index
        yield self
        for arg, index in variable2index.items():
            del self.bounded_variables[arg.type.typename][arg.name]

    def get_bounded_variable(self, variable: Variable) -> Union[int, slice, Value]:
        if variable.name == '??':
            return QINDEX
        return self.bounded_variables[variable.type.typename][variable.name]

    @wrap_custom_as_default
    def as_default(self):
        yield self


def get_definition_context() -> ExpressionDefinitionContext: ...
get_definition_context = gen_get_default(ExpressionDefinitionContext)


def get_execution_context() -> ExpressionExecutionContext: ...
get_execution_context = gen_get_default(ExpressionExecutionContext)


class Expression(ABC):
    # @jacinle.log_function
    def forward(self, ctx: ExpressionExecutionContext) -> Union[Value, MixedValue]:
        """
        Forward the computation of the expression. There are four different cases.

            - Standard: does all the computation based on concrete values.
            - Quantized: the execution works for quantized values.
            - Optimistic execution: the execution contains optimistic values. Thus, the execution result is either a
                concrete Value object, or just a PartialExecutionValue (if optimistic values are involved).
            - Mixed State: the execution output is either a concrete Value object, or a set of possible `Value`s.
                In this case, the output will be of type `MixedValue`.

        Note that mixed state execution and partial execution asserts quantized state.
        """
        if ctx.is_mixed_state:
            assert ctx.state.batch_dims == 0, 'Mixed state execution does not support batch.'
            # return self._forward_mixed(ctx)
            raise NotImplementedError('Mixed state execution is not implemented.')
        if ctx.is_relaxed_execution:
            assert ctx.state.batch_dims == 0, 'Relaxed execution does not support batch.'
            return self._forward_relaxed(ctx)
        return self._forward(ctx)

    def _forward(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, Value]]:
        raise NotImplementedError()

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, Value]]:
        raise NotImplementedError()

    # def _forward_mixed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, MixedValue]]:
    #     raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    __repr__ = jacinle.repr_from_str


class VariableExpression(Expression):
    def __init__(self, variable: Variable):
        self.variable = variable

    @property
    def name(self):
        return self.variable.name

    @property
    def type(self):
        return self.variable.type

    @property
    def output_type(self):
        return self.variable.type

    def _forward(self, ctx: ExpressionExecutionContext) -> Union[int, slice, torch.Tensor, Value]:
        return ctx.get_bounded_variable(self.variable)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, Value]]:
        return self._forward(ctx)

    def _forward_mixed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, MixedValue]]:
        return self._forward(ctx)

    def __str__(self) -> str:
        return f'V::{self.name}'


class ObjectConstantExpression(Expression):
    def __init__(self, constant: StringConstant):
        self.constant = constant

    @property
    def name(self):
        return self.constant.name

    @property
    def type(self):
        return self.constant.type if self.constant.type is not None else RUNTIME_BINDING

    @property
    def output_type(self):
        return self.constant.type if self.constant.type is not None else RUNTIME_BINDING

    def _forward(self, ctx: ExpressionExecutionContext) -> Union[int, torch.Tensor]:
        if ctx.state.batch_dims > 0:  # Batched state.
            return torch.tensor(ctx.state.get_typed_index(self.name), dtype=torch.int64)
        assert isinstance(ctx.state, SingleStateLike)
        assert self.name in ctx.state.object_names, f'Object {self.name} does not exist.'
        return ctx.state.get_typed_index(self.name)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, Value]]:
        return self._forward(ctx)

    def _forward_mixed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, MixedValue]]:
        return self._forward(ctx)

    def __str__(self):
        return f'OBJ::{self.name}'


class ValueOutputExpression(Expression):
    @property
    def output_type(self):
        raise NotImplementedError()

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        raise NotImplementedError()

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Value:
        raise NotImplementedError()

    # def _forward_mixed(self, ctx: ExpressionExecutionContext) -> MixedValue:
    #     raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


def is_value_output_expression(expr: Expression) -> bool:
    return isinstance(expr, ValueOutputExpression)


class ConstantExpression(ValueOutputExpression):
    def __init__(self, type: ValueType, value: Union[bool, int, float, torch.Tensor], quantized=False):
        assert torch.is_tensor(value)
        self.value = Value(type, [], value, quantized=quantized)

    @property
    def output_type(self):
        return self.value.dtype

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        return self.value

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, Value]]:
        value = self.value.clone(False)
        value.set_backward_function(self._backward_relaxed)
        return value

    def _backward_relaxed(self, output_value):
        pass

    @classmethod
    def true(cls):
        return cls(BOOL, torch.tensor(1, dtype=torch.int64), quantized=True)

    @classmethod
    def false(cls):
        return cls(BOOL, torch.tensor(0, dtype=torch.int64), quantized=True)

    @classmethod
    def int64(cls, value):
        return cls(INT64, torch.tensor(value, dtype=torch.int64), quantized=True)

    def __str__(self):
        return str(self.value)


ConstantExpression.TRUE = ConstantExpression.true()
ConstantExpression.FALSE = ConstantExpression.false()


class FunctionApplicationError(Exception):
    def __init__(self, index, expect, got):
        msg = f'Argument #{index} type does not match: expect {expect}, got {got}.'
        super().__init__(msg)

        self.index = index
        self.expect = expect
        self.got = got


def forward_args(ctx, *args, force_tuple=False):
    rv = list()
    for arg in args:
        if isinstance(arg, Value):
            rv.append(arg)
        elif isinstance(arg, Expression):
            rv.append(arg.forward(ctx))
        elif isinstance(arg, int):  # object index.
            rv.append(arg)
        else:
            raise TypeError('Unknown argument type: {}.'.format(type(arg)))
    if len(rv) == 1 and not force_tuple:
        return rv[0]
    return tuple(rv)


def has_partial_execution_value(argument_values: Sequence[Value]) -> bool:
    for argv in argument_values:
        if isinstance(argv, Value):
            if argv.has_optimistic_value():
                return True
        elif isinstance(argv, (int, slice)):
            pass
        else:
            raise TypeError('Unknown argument value type: {}.'.format(type(argv)))
    return False


def expand_argument_values(argument_values: Sequence[Value]) -> List[Value]:
    has_slot_var = False
    for arg in argument_values:
        if isinstance(arg, Value):
            for var in arg.batch_variables:
                if var == '??':
                    has_slot_var = True
                    break
    if has_slot_var:
        return list(argument_values)

    if len(argument_values) < 2:
        return list(argument_values)

    argument_values = list(argument_values)
    batch_variables = list()
    batch_sizes = list()
    for arg in argument_values:
        if isinstance(arg, Value):
            for var in arg.batch_variables:
                if var not in batch_variables:
                    batch_variables.append(var)
                    batch_sizes.append(arg.get_variable_size(var))
        else:
            assert isinstance(arg, (int, slice)), arg

    masks = list()
    for i, arg in enumerate(argument_values):
        if isinstance(arg, Value):
            argument_values[i] = arg.expand(batch_variables, batch_sizes)
            if argument_values[i].tensor_mask is not None:
                masks.append(argument_values[i].tensor_mask)

    if len(masks) > 0:
        final_mask = torch.stack(masks, dim=-1).amin(dim=-1)
        for arg in argument_values:
            if isinstance(arg, Value):
                arg.tensor_mask = final_mask
                arg._mask_certified_flag = True  # now we have corrected the mask.
    return argument_values


class FunctionApplication(ValueOutputExpression, ABC):
    def __init__(self, function_def: FunctionDef, *arguments: Expression):
        self.function_def = function_def
        self.arguments = arguments
        self._check_arguments()

    def _check_arguments(self):
        try:
            if len(self.function_def.arguments) != len(self.arguments):
                raise TypeError('Argument number mismatch: expect {}, got {}.'.format(len(self.function_def.arguments), len(self.arguments)))
            for i, (arg_def, arg) in enumerate(zip(self.function_def.arguments, self.arguments)):
                if isinstance(arg_def, Variable):
                    if isinstance(arg, VariableExpression):
                        if arg_def.type != arg.type and not isinstance(arg.type, type(RUNTIME_BINDING)):
                            raise FunctionApplicationError(i, arg_def.type, arg.type)
                    elif isinstance(arg, ObjectConstantExpression):
                        if arg_def.type != arg.type:
                            raise FunctionApplicationError(i, arg_def.type, arg.type)
                    elif isinstance(arg, FunctionApplication):
                        if arg_def.type != arg.output_type:
                            raise FunctionApplicationError(i, arg_def.type, arg.output_type)
                    else:
                        raise FunctionApplicationError(i, 'VariableExpression or FunctionApplication', type(arg))
                elif isinstance(arg_def, (ValueType, NamedValueType)):
                    if isinstance(arg, ValueOutputExpression):
                        pass
                    elif isinstance(arg, VariableExpression) and isinstance(arg.output_type, (ValueType, NamedValueType)):
                        pass
                    else:
                        raise FunctionApplicationError(i, 'ValueOutputExpression', type(arg))
                    if arg_def != arg.output_type:
                        raise FunctionApplicationError(i, arg_def, arg.output_type)
                else:
                    raise TypeError('Unknown argdef type: {}.'.format(type(arg_def)))
        except (TypeError, FunctionApplicationError) as e:
            error_header = 'Error during applying {}.\n'.format(str(self.function_def))
            try:
                arguments_str = ', '.join(str(arg) for arg in self.arguments)
                error_header += ' Arguments: {}\n'.format(arguments_str)
            except:
                pass
            raise TypeError(error_header + str(e)) from e

    @property
    def output_type(self):
        return self.function_def.output_type

    def __str__(self):
        arguments = ', '.join([str(arg) for arg in self.arguments])
        return f'{self.function_def.name}({arguments})'

    def _forward_external_function(self, ctx: ExpressionExecutionContext, external_function, argument_values, force_quantized=False, force_non_optimistic=False):
        if ctx.is_relaxed_execution:  # relaxed execution branch.
            assert all([argv.quantized for argv in argument_values])
            argument_values = expand_argument_values(argument_values)
            optimistic_masks = [is_optimistic_value(argv.tensor) for argv in argument_values]
            if len(optimistic_masks) > 0:
                optimistic_mask = torch.stack(optimistic_masks, dim=-1).any(dim=-1)
                if optimistic_mask.sum().item() == 0:
                    pass  # just do the standard execution.
                else:
                    retain_mask = torch.logical_not(optimistic_mask)
                    rv = torch.zeros(
                        argument_values[0].tensor.shape,
                        dtype=torch.int64,
                        device=argument_values[0].tensor.device
                    ) + RELAX_MAGIC_NUMBER

                    if retain_mask.sum().item() > 0:
                        argument_values_r = [Value(argv.dtype, ['?x'], argv.tensor[retain_mask], 0, quantized=argv.quantized) for argv in argument_values]
                        rv_r = self._forward_external_function(ctx, external_function, argument_values_r, force_quantized=True, force_non_optimistic=True)
                        rv[retain_mask] = rv_r.tensor

                    rv = Value(
                        self.function_def.output_type, argument_values[0].batch_variables if len(argument_values) > 0 else [],
                        rv,
                        batch_dims=ctx.state.batch_dims, quantized=True
                    )
                    rv.set_backward_function(NotImplemented)

        if not force_non_optimistic and ctx.is_optimistic_execution:  # optimistic execution branch.
            argument_values = expand_argument_values(argument_values)
            optimistic_masks = [is_optimistic_value(argv.tensor) for argv in argument_values if argv.quantized]
            if len(optimistic_masks) > 0:
                optimistic_mask = torch.stack(optimistic_masks, dim=-1).any(dim=-1)
                if optimistic_mask.sum().item() == 0:
                    pass  # just do the standard execution.
                else:
                    retain_mask = torch.logical_not(optimistic_mask)
                    rv = torch.zeros(
                        argument_values[0].tensor.shape,
                        dtype=torch.int64,
                        device=argument_values[0].tensor.device
                    )

                    if retain_mask.sum().item() > 0:
                        argument_values_r = [Value(argv.dtype, ['?x'], argv.tensor[retain_mask], 0, quantized=argv.quantized) for argv in argument_values]
                        rv_r = self._forward_external_function(ctx, external_function, argument_values_r, force_quantized=True, force_non_optimistic=True)
                        rv[retain_mask] = rv_r.tensor

                    for ind in optimistic_mask.nonzero().tolist():
                        ind = tuple(ind)
                        new_identifier = ctx.optimistic_context.new_var(self.output_type)
                        rv[ind] = new_identifier
                        ctx.optimistic_context.add_constraint(OptimisticConstraint.from_function_def(
                            self.function_def,
                            [argv.tensor[ind].item() if argv.quantized else argv.tensor[ind] for argv in argument_values],
                            new_identifier
                        ))

                    return Value(
                        self.function_def.output_type, argument_values[0].batch_variables if len(argument_values) > 0 else [],
                        rv,
                        batch_dims=ctx.state.batch_dims, quantized=True
                    )

        # Standard execution branch.
        quantized = False
        all_quantized = all([v.quantized for v in argument_values])
        if all_quantized and external_function.function_quantized is not None:
            if external_function.auto_broadcast:
                argument_values = expand_argument_values(argument_values)
            rv = external_function.function_quantized(*argument_values)
            quantized = True
        else:
            argument_values = list(argument_values)
            for i, argv in enumerate(argument_values):
                if argv.quantized and not is_intrinsically_quantized(argv.dtype):
                    argument_values[i] = ctx.value_quantizer.unquantize_value(argv)
            if external_function.auto_broadcast:
                argument_values = expand_argument_values(argument_values)
            rv = external_function(*argument_values)

        if torch.is_tensor(rv):
            rv = Value(
                self.function_def.output_type, argument_values[0].batch_variables if len(argument_values) > 0 else [],
                QuantizedTensorValue(rv, None, argument_values[0].tensor_mask if len(argument_values) > 0 else None),
                batch_dims=ctx.state.batch_dims, quantized=quantized
            )
        elif isinstance(rv, QuantizedTensorValue):
            rv = Value(
                self.function_def.output_type, argument_values[0].batch_variables if len(argument_values) > 0 else [],
                rv,
                batch_dims=ctx.state.batch_dims, quantized=quantized
            )
        else:
            assert isinstance(rv, Value), 'Expect external function return Tensor, Value, or QuantizedTensorValue objects, got {}.'.format(type(rv))

        if not rv.quantized and force_quantized:
            return ctx.value_quantizer.quantize_value(rv)
        return rv


class FeatureApplication(FunctionApplication):
    RUNTIME_BINDING_CHECK = False

    def _check_arguments(self):
        assert isinstance(self.function_def, FeatureDef)
        super()._check_arguments()

    def _check_arguments_runtime(self, ctx):
        if ctx.state.batch_dims == 0:  # Does not work for BatchState.
            for arg_def, arg in zip(self.function_def.arguments, self.arguments):
                if isinstance(arg, ObjectConstantExpression):
                    if arg.type == RUNTIME_BINDING:
                        got_type = ctx.state.get_typename(arg.name)
                        exp_type = arg_def.type.typename
                        if got_type != exp_type:
                            error_header = 'Error during applying {}.\n'.format(str(self.function_def))
                            raise TypeError(error_header + f'Runtime type check for argument {arg_def.name}: expect {exp_type}, got {got_type}.')

    @property
    def feature_def(self) -> FeatureDef:
        return self.function_def

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        if self.RUNTIME_BINDING_CHECK:
            self._check_arguments_runtime(ctx)
        argument_values = forward_args(ctx, *self.arguments, force_tuple=True)

        if self.feature_def.cacheable and self.feature_def.name in ctx.state.features:  # i.e., all variables are ObjectType
            batch_variables = [arg.name for arg, value in zip(self.arguments, argument_values) if value == QINDEX]
            value = ctx.state.features[self.feature_def.name][argument_values]
            value = value.rename_batch_variables(batch_variables)
            if ctx.is_relaxed_execution:
                value.set_backward_function(self._backward_relaxed_cached, ctx.state, argument_values)
            return value
        elif self.feature_def.expr is not None:
            return self._forward_expr_internal(ctx, argument_values)
        else:
            # dynamic predicate is exactly the same thing as a pre-defined external function.
            external_function = ctx.get_external_function('feature::' + self.feature_def.name)
            return self._forward_external_function(ctx, external_function, argument_values)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, Value]]:
        return self._forward(ctx)

    def _backward_relaxed_cached(self, value: Value, state, argument_values: Tuple[Union[int, slice], ...]):
        state.features[self.feature_def.name].tensor_grad[argument_values] += value.tensor_grad

    def _forward_expr_internal(self, ctx, argument_values):
        bounded_variables = dict()
        for var, value in zip(self.feature_def.arguments, argument_values):
            if var.type.typename not in bounded_variables:
                bounded_variables[var.type.typename] = dict()
            bounded_variables[var.type.typename][var.name] = value

        nctx = ExpressionExecutionContext(ctx.domain, ctx.state, bounded_variables=bounded_variables, optimistic_context=ctx.optimistic_context)
        with nctx.as_default():
            rv = self.feature_def.expr.forward(nctx)

        guess_new_names = [arg.name for arg, value in zip(self.arguments, argument_values) if isinstance(arg, VariableExpression) and value == QINDEX]
        if len(guess_new_names) > 0:
            return rv.rename_batch_variables(guess_new_names)
        return rv


class PredicateApplication(FeatureApplication):  # just an alias.
    def _check_arguments(self):
        assert isinstance(self.function_def, PredicateDef)
        super()._check_arguments()


class ExternalFunctionApplication(FunctionApplication):
    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        external_function = ctx.get_external_function(self.function_def.name)
        argument_values = forward_args(ctx, *self.arguments, force_tuple=True)
        return self._forward_external_function(ctx, external_function, argument_values)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Value:
        raise NotImplementedError()


def is_feature_application(expr: Expression) -> bool:
    return isinstance(expr, FeatureApplication)


def is_external_function_application(expr: Expression) -> bool:
    return isinstance(expr, ExternalFunctionApplication) or (
        isinstance(expr, FeatureApplication) and
        expr.feature_def.expr is None and
        not expr.feature_def.cacheable
    )


class ConditionalSelectOp(ValueOutputExpression):
    def __init__(self, feature: ValueOutputExpression, condition: ValueOutputExpression):
        self.feature = feature
        self.condition = condition

    def _check_arguments(self):
        assert isinstance(self.condition, ValueOutputExpression) and self.condition.output_type == BOOL

    @property
    def output_type(self):
        return self.feature.output_type

    def _forward(self, ctx: ExpressionExecutionContext):
        value, condition = forward_args(ctx, self.feature, self.condition)
        value = value.clone()
        if value.tensor_mask is None:
            value.tensor_mask = torch.ones(value.tensor.shape[:value.total_batch_dims], device=value.tensor.device, dtype=torch.int64)
        value.tensor_mask = torch.min(value.tensor_mask, condition.tensor)
        return value

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, Value]]:
        raise NotImplementedError()

    def __str__(self):
        return f'cond-select({self.feature} if {self.condition})'


class DeicticSelectOp(ValueOutputExpression):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        self.variable = variable
        self.expr = expr
        self._check_arguments()

    @property
    def output_type(self):
        return self.expr.output_type

    def _check_arguments(self):
        assert isinstance(self.variable.type, ObjectType)

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        with ctx.new_bounded_variables({self.variable: QINDEX}):
            return self.expr.forward(ctx)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext):
        return self._forward(ctx)

    def __str__(self):
        return f'deictic-select({self.variable}: {self.expr})'


class BoolOpType(JacEnum):
    AND = 'and'
    OR = 'or'
    NOT = 'not'


def _forward_relaxed_cd(op_type: BoolOpType, value: torch.Tensor, mask: torch.Tensor, dim: int):
    if op_type is BoolOpType.AND:
        f = value
        f = (torch.zeros_like(f) + 1) * mask + f * (1 - mask)
        f = f.amin(dim=dim)
        mask = mask.any(dim=dim).long()
        rv = (
            (torch.zeros_like(f) + RELAX_MAGIC_NUMBER) * mask * f
            + f * (1 - mask)
        )
    elif op_type is BoolOpType.OR:
        f = value
        f = torch.zeros_like(f) * mask + f * (1 - mask)
        f = f.amax(dim=dim)
        mask= mask.any(dim=dim).long()
        rv = (
            (torch.zeros_like(f) + RELAX_MAGIC_NUMBER) * mask * (1 - f)
            + f * (1 - mask)
        )
    else:
        raise ValueError('Unknown relaxed bool op type: {}.'.format(op_type))
    return rv


def _backward_relaxed_cd(op_type: BoolOpType, grad: torch.Tensor, f: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    if op_type is BoolOpType.AND:
        return grad.unsqueeze(dim) * mask
    elif op_type is BoolOpType.OR:
        cmask = torch.cumsum(mask, dim)
        cmask1, cmask2 = torch.split_with_sizes(cmask, [cmask.size(dim) - 1, 1], dim)
        cmask_shift = torch.cat([torch.zeros_like(cmask2), cmask1], dim=dim)
        cmask_first = ((cmask > 0) - (cmask_shift > 0)).long()
        return grad.unsqueeze(dim) * cmask_first
    else:
        raise ValueError('Unknown relaxed bool op type: {}.'.format(op_type))


class BoolOp(ValueOutputExpression):
    def __init__(self, boolean_op_type: BoolOpType, arguments: Sequence[ValueOutputExpression]):
        self.bool_op_type = boolean_op_type
        self.arguments = arguments
        self._check_arguments()

    def _check_arguments(self):
        if self.bool_op_type is BoolOpType.NOT:
            assert len(self.arguments) == 1, f'Number of arguments for NotOp should be 1, got: {len(self.arguments)}.'
        for i, arg in enumerate(self.arguments):
            assert isinstance(arg, ValueOutputExpression), f'BooOp only accepts ValueOutputExpressions, got argument #{i} of type {type(arg)}.'

    @property
    def output_type(self):
        return self.arguments[0].output_type

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        argument_values = forward_args(ctx, *self.arguments, force_tuple=True)
        argument_values = list(expand_argument_values(argument_values))

        if ctx.is_optimistic_execution:
            optimistic_masks = [is_optimistic_value(argv.tensor) for argv in argument_values if argv.quantized]
            if len(optimistic_masks) > 0:
                optimistic_mask = torch.stack(optimistic_masks, dim=-1).any(dim=-1)
                if optimistic_mask.sum().item() > 0:
                    for i, argv in enumerate(argument_values):
                        if not argv.quantized:
                            argument_values[i] = argv.make_quantized()
                        # assert argv.quantized, 'Found optimistic values in BoolOp, but at least one of the arguments is not quantized.'
                    retain_mask = torch.logical_not(optimistic_mask)
                    rv = torch.zeros(
                        argument_values[0].tensor.shape,
                        dtype=torch.int64,
                        device=argument_values[0].tensor.device
                    )
                    if retain_mask.sum().item() > 0:
                        argument_values_r = [Value(argv.dtype, ['?x'], argv.tensor[retain_mask], 0, quantized=True) for argv in argument_values]
                        rv_r = self._forward_inner(ctx, argument_values_r)
                        rv[retain_mask] = rv_r.tensor

                    for ind in optimistic_mask.nonzero().tolist():
                        ind = tuple(ind)

                        this_argv = [argv.tensor[ind].item() for argv in argument_values]
                        determined = None
                        if self.output_type == BOOL:
                            if self.bool_op_type is BoolOpType.NOT:
                                pass  # nothing we can do.
                            elif self.bool_op_type is BoolOpType.AND:
                                if 0 in this_argv:
                                    determined = False
                            elif self.bool_op_type is BoolOpType.OR:
                                if 1 in this_argv:
                                    determined = True
                            this_argv = list(filter(is_optimistic_value, this_argv))
                        else:  # generalized boolean operations.
                            pass

                        if determined is None:
                            new_identifier = ctx.optimistic_context.new_var(self.output_type)
                            rv[ind] = new_identifier
                            ctx.optimistic_context.add_constraint(OptimisticConstraint(
                                self.bool_op_type,
                                [cvt_opt_value(v, self.output_type) for v in this_argv],
                                cvt_opt_value(new_identifier, self.output_type),
                            ))
                        else:
                            rv[ind] = determined
                    return Value(self.output_type, argument_values[0].batch_variables, rv, argument_values[0].batch_dims, quantized=True)
                else:
                    return self._forward_inner(ctx, argument_values)
            else:  # if len(optimistic_masks) == 0
                return self._forward_inner(ctx, argument_values)
        else:
            return self._forward_inner(ctx, argument_values)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Value:
        argument_values = forward_args(ctx, *self.arguments, force_tuple=True)
        argument_values = expand_argument_values(argument_values)

        all_quantized = all([argv.quantized for argv in argument_values])
        assert all_quantized

        dtype = argument_values[0].dtype
        batch_variables = argument_values[0].batch_variables
        optimistic_masks = [is_optimistic_value(argv.tensor) for argv in argument_values]
        optimistic_mask = torch.stack(optimistic_masks, dim=-1).long()
        if self.bool_op_type is BoolOpType.NOT:
            f = argument_values[0].tensor
            rv = 1 - f
            rv = (torch.zeros_like(rv) + RELAX_MAGIC_NUMBER) * optimistic_mask[..., 0] + (1 - rv) * (1 - optimistic_mask[..., 0])
        elif self.bool_op_type in (BoolOpType.AND, BoolOpType.OR):
            if len(argument_values) == 1:
                return argument_values[0]
            f = torch.stack([arg.tensor for arg in argument_values], dim=-1)
            rv = _forward_relaxed_cd(self.bool_op_type, f, optimistic_mask, dim=-1)
        else:
            raise ValueError('Unknown Boolean op type: {}.'.format(self.bool_op_type))

        value = Value(
            dtype, batch_variables,
            QuantizedTensorValue(rv, None, argument_values[0].tensor_mask),
            batch_dims=argument_values[0].batch_dims, quantized=True
        )
        value.set_backward_function(self._backward_relaxed, f, optimistic_mask, argument_values)
        return value

    def _backward_relaxed(self, value: Value, f, mask, argument_values):
        if self.bool_op_type is BoolOpType.NOT:
            argument_values[0].tensor_grad = value.tensor_grad
            argument_values[0].backward_inner()
        else:
            grad_tensor = _backward_relaxed_cd(self.bool_op_type, value.tensor_grad, f, mask, -1)
            splits = grad_tensor.split(1, dim=-1)
            for argv, s in zip(argument_values, splits):
                argv.tensor_grad = s[..., 0]
                argv.backward_inner()

    def _forward_inner(self, ctx: ExpressionExecutionContext, argument_values) -> Value:
        for value in argument_values:
            assert value.tensor_indices is None, 'Does not support quantification over values with tensor_indices.'

        if len(argument_values) == 0:
            return Value(BOOL, [], torch.tensor(1, dtype=torch.int64), quantized=True)

        all_quantized = all([argv.quantized for argv in argument_values])
        # NB: when not all quantized, the tensors should be casted to float32.
        # Interestingly, PyTorch automatically handles that.

        dtype = argument_values[0].dtype
        batch_variables = argument_values[0].batch_variables
        if self.bool_op_type is BoolOpType.NOT:
            rv = argument_values[0].tensor
            rv = torch.logical_not(rv) if rv.dtype == torch.bool else 1 - rv
        elif self.bool_op_type is BoolOpType.AND:
            if len(argument_values) == 1:
                return argument_values[0]
            rv = torch.stack([arg.tensor for arg in argument_values], dim=-1).amin(dim=-1)
        elif self.bool_op_type is BoolOpType.OR:
            if len(argument_values) == 1:
                return argument_values[0]
            rv = torch.stack([arg.tensor for arg in argument_values], dim=-1).amax(dim=-1)
        else:
            raise ValueError('Unknown Boolean op type: {}.'.format(self.bool_op_type))
        return Value(
            dtype, batch_variables,
            QuantizedTensorValue(rv, None, argument_values[0].tensor_mask),
            batch_dims=argument_values[0].batch_dims, quantized=all_quantized
        )

    def __str__(self):
        arguments = ', '.join([str(arg) for arg in self.arguments])
        return f'{self.bool_op_type.value}({arguments})'


class AndOp(BoolOp):
    def __init__(self, *arguments: ValueOutputExpression):
        super().__init__(BoolOpType.AND, arguments)


class OrOp(BoolOp):
    def __init__(self, *arguments: ValueOutputExpression):
        super().__init__(BoolOpType.OR, arguments)


class NotOp(BoolOp):
    def __init__(self, arg: ValueOutputExpression):
        super().__init__(BoolOpType.NOT, [arg])


def is_and_expr(expr: Expression):
    return isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.AND


def is_or_expr(expr: Expression):
    return isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.OR


def is_not_expr(expr: Expression):
    return isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.NOT


def is_constant_bool_expr(expr: Expression):
    if isinstance(expr, ConstantExpression) and expr.output_type == BOOL:
        return True
    return False


def is_simple_bool(expr: Expression) -> bool:
    if isinstance(expr, FeatureApplication) and expr.feature_def.cacheable and expr.feature_def.group in ('basic', 'augmented'):
        return True
    if isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.NOT:
        return is_simple_bool(expr.arguments[0])
    return False


def split_simple_bool(expr: Expression, initial_negated: bool = False) -> Tuple[Optional[FeatureApplication], bool]:
    """
    If the expression is a feature application with a cacheable feature, and the feature is either basic or augmented.
    it returns the feature definition and a boolean indicating whether the expression is negated.

    Args:
        expr (Expression): the expression to be checked.
        initial_negated (bool, optional): whether outer context of the feature expression is a negated function.

    Returns:
        (FeatureApplication, bool): A tuple of the feature application and a boolean indicating whether the expression is negated.
        The first element is None if the feature is not a simple Boolean feature application.
    """
    if isinstance(expr, FeatureApplication) and expr.feature_def.cacheable and expr.feature_def.group in ('basic', 'augmented', 'augmented-input'):
        return expr, initial_negated
    if isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.NOT:
        return split_simple_bool(expr.arguments[0], not initial_negated)
    return None, initial_negated


def get_simple_bool_def(expr: Expression):
    if isinstance(expr, FeatureApplication):
        return expr.feature_def
    assert isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.NOT
    return expr.arguments[0].feature_def


class QuantifierType(JacEnum):
    FORALL = 'forall'
    EXISTS = 'exists'


class QuantificationOp(ValueOutputExpression):
    def __init__(self, quantifier_type: QuantifierType, variable: Variable, expr: ValueOutputExpression):
        self.quantifier_type = quantifier_type
        self.variable = variable
        self.expr = expr

        self._check_arguments()

    def _check_arguments(self):
        assert isinstance(self.expr, ValueOutputExpression), f'QuantificationOp only accepts ValueOutputExpressions, got type {type(self.expr)}.'
        assert isinstance(self.variable.type, ObjectType)

    @property
    def output_type(self):
        return self.expr.output_type

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        with ctx.new_bounded_variables({self.variable: QINDEX}):
            value = self.expr.forward(ctx)
        assert self.variable.name in value.batch_variables, f'Quantified argument is not in batch_variables: expect {self.variable.name}, got {value.batch_variables}.'
        rv_r = self._forward_inner(ctx, value)

        if ctx.is_optimistic_execution and value.quantized:
            dim = value.batch_variables.index(self.variable.name) + value.batch_dims
            value_transformed = value.tensor
            if dim != value.tensor.ndim - 1:
                value_transformed = value.tensor.transpose(dim, -1)  # put the target dimension last.
            optimistic_mask = is_optimistic_value(value_transformed).any(dim=-1)

            if optimistic_mask.sum().item() == 0:
                return rv_r

            rv = rv_r.tensor.clone()
            for ind in optimistic_mask.nonzero().tolist():
                ind = tuple(ind)

                this_argv = value_transformed[ind].tolist()
                determined = None
                if self.output_type == BOOL:
                    if self.quantifier_type is QuantifierType.FORALL:
                        if 0 in this_argv:
                            determined = False
                    else:
                        if 1 in this_argv:
                            determined = True
                    this_argv = list(filter(is_optimistic_value, this_argv))
                else:  # generalized quantization.
                    pass

                if determined is None:
                    new_identifier = ctx.optimistic_context.new_var(self.output_type)
                    rv[ind] = new_identifier
                    ctx.optimistic_context.add_constraint(OptimisticConstraint(
                        self.quantifier_type,
                        [cvt_opt_value(v, value.dtype) for v in this_argv],
                        cvt_opt_value(new_identifier, value.dtype)
                    ))
                else:
                    rv[ind] = determined
            return Value(self.output_type, rv_r.batch_variables, rv, batch_dims=rv_r.batch_dims, quantized=True)
        else:
            return rv_r

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Value:
        with ctx.new_bounded_variables({self.variable: QINDEX}):
            value = self.expr.forward(ctx)
        assert self.variable.name in value.batch_variables, f'Quantified argument is not in batch_variables: expect {self.variable.name}, got {value.batch_variables}.'
        assert value.dtype == BOOL
        assert value.tensor_mask is None
        dim = value.batch_variables.index(self.variable.name) + value.batch_dims
        batch_variables = list(value.batch_variables)
        batch_variables.remove(self.variable.name)

        optimistic_mask = is_optimistic_value(value.tensor).long()
        if self.quantifier_type is QuantifierType.FORALL:
            rv = _forward_relaxed_cd(BoolOpType.AND, value.tensor, optimistic_mask, dim)
        elif self.quantifier_type is QuantifierType.EXISTS:
            rv = _forward_relaxed_cd(BoolOpType.OR, value.tensor, optimistic_mask, dim)
        else:
            raise ValueError('Unknown quantifier type: {}.'.format(self.quantifier_type))
        value = Value(self.output_type, batch_variables, rv, batch_dims=value.batch_dims, quantized=True)
        value.set_backward_function(self._backward_relaxed, value, optimistic_mask, dim)
        return value

    def _backward_relaxed(self, output_value: Value, input_value: Value, mask, dim):
        if self.quantifier_type is QuantifierType.FORALL:
            bool_op_type = BoolOpType.AND
        elif self.quantifier_type is QuantifierType.EXISTS:
            bool_op_type = BoolOpType.OR
        else:
            raise ValueError('Unknown quantifier type: {}.'.format(self.quantifier_type))
        input_value.tensor_grad = _backward_relaxed_cd(bool_op_type, output_value.tensor_grad, input_value.tensor, mask, dim=dim)
        input_value.backward_inner()

    def _forward_inner(self, ctx: ExpressionExecutionContext, value):
        dim = value.batch_variables.index(self.variable.name) + value.batch_dims
        batch_variables = list(value.batch_variables)
        batch_variables.remove(self.variable.name)

        assert value.tensor_indices is None, 'Does not support quantification over values with tensor_indices.'
        if value.tensor_mask is None:
            masked_tensor = value.tensor
            tensor_mask = None
        else:
            if self.quantifier_type is QuantifierType.FORALL:
                masked_tensor = (value.tensor * value.tensor_mask + (1 - value.tensor_mask)).to(value.tensor.dtype)
            elif self.quantifier_type is QuantifierType.EXISTS:
                masked_tensor = (value.tensor * value.tensor_mask)
            else:
                raise ValueError('Unknown quantifier type: {}.'.format(self.quantifier_type))
            tensor_mask = value.tensor_mask.narrow(dim, 0, 1).squeeze(dim)

        if self.quantifier_type is QuantifierType.FORALL:
            return Value(
                value.dtype, batch_variables, QuantizedTensorValue(masked_tensor.amin(dim=dim), None, tensor_mask),
                batch_dims=value.batch_dims, quantized=value.quantized
            )
        elif self.quantifier_type is QuantifierType.EXISTS:
            return Value(
                value.dtype, batch_variables, QuantizedTensorValue(masked_tensor.amax(dim=dim), None, tensor_mask),
                batch_dims=value.batch_dims, quantized=value.quantized
            )
        else:
            raise ValueError('Unknown quantifier type: {}.'.format(self.quantifier_type))

    def __str__(self):
        return f'{self.quantifier_type.value}({self.variable}: {self.expr})'


class ForallOp(QuantificationOp):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        super().__init__(QuantifierType.FORALL, variable, expr)


class ExistsOp(QuantificationOp):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        super().__init__(QuantifierType.EXISTS, variable, expr)


def is_forall_expr(expr: Expression):
    return isinstance(expr, QuantificationOp) and expr.quantifier_type is QuantifierType.FORALL


def is_exists_expr(expr: Expression):
    return isinstance(expr, QuantificationOp) and expr.quantifier_type is QuantifierType.EXISTS


class _FeatureValueExpression(Expression, ABC):
    def __init__(self, feature: Union[VariableExpression, FeatureApplication, PredicateApplication], value: ValueOutputExpression):
        self.feature = feature
        self.value = value
        self._check_arguments()

    def _check_arguments(self):
        try:
            if isinstance(self.feature.output_type, (ValueType, NamedValueType)):
                if self.feature.output_type.assignment_type() != self.value.output_type:
                    raise FunctionApplicationError(0, f'{self.feature.output_type}(assignment type is {self.feature.output_type.assignment_type()})', self.value.output_type)
            else:
                raise TypeError('Unknown argdef type: {}.'.format(type(self.feature.output_type)))
        except TypeError as e:
            raise e
        except FunctionApplicationError as e:
            error_header = 'Error during FeatureValueExpression checking: feature = {} value = {}.\n'.format(str(self.feature), str(self.value))
            raise TypeError(
                error_header +
                f'Value type does not match: expect: {e.expect}, got {e.got}.'
            ) from e


class FeatureEqualOp(ValueOutputExpression, _FeatureValueExpression):
    def _check_arguments(self):
        super()._check_arguments()
        assert isinstance(self.feature, (VariableExpression, FeatureApplication, PredicateApplication)), 'FeatureEqualOp only support dest type VariableExpression or FeatureApplication or PredicateApplication, got {}.'.format(type(self.feature))

    @property
    def output_type(self):
        return BOOL

    def _forward(self, ctx: ExpressionExecutionContext):
        feature, value = forward_args(ctx, self.feature, self.value)
        feature, value = expand_argument_values([feature, value])

        if ctx.is_optimistic_execution and (feature.quantized or value.quantized):
            return self._forward_optimistic(ctx, feature, value)

        if feature.quantized and value.quantized:
            rv = torch.eq(feature.tensor, value.tensor)
        elif feature.tensor_indices is not None and value.tensor_indices is not None:
            rv = torch.eq(feature.tensor_indices, value.tensor_indices)
        elif isinstance(feature.dtype, NamedValueType):
            rv = ctx.domain.get_external_function(f'type::{feature.dtype.typename}::equal')(feature, value)
        else:
            raise NotImplementedError('Unsupported FeatureEqual computation for dtype {}.'.format(feature, value))

        return Value(BOOL, feature.batch_variables, QuantizedTensorValue(rv, None, feature.tensor_mask), batch_dims=feature.batch_dims, quantized=True)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Value:
        feature, value = forward_args(ctx, self.feature, self.value)
        feature, value = expand_argument_values([feature, value])

        assert feature.quantized and value.quantized
        assert feature.tensor_mask is None and value.tensor_mask is None
        f = torch.eq(feature.tensor, value.tensor)
        mask = torch.logical_or(is_optimistic_value(feature.tensor), is_optimistic_value(feature.tensor)).long()
        rv = (torch.zeros_like(f) + RELAX_MAGIC_NUMBER) * mask + f * (1 - mask)
        value = Value(BOOL, feature.batch_variables, rv, batch_dims=feature.batch_dims, quantized=True)
        value.set_backward_function(NotImplemented)
        return value

    def _forward_optimistic(self, ctx: ExpressionExecutionContext, feature: Value, value: Value) -> Value:
        feature, value = ctx.value_quantizer.quantize_value(feature), ctx.value_quantizer.quantize_value(value)
        rv_r = torch.eq(feature.tensor, value.tensor)
        optimistic_mask = torch.logical_or(is_optimistic_value(feature.tensor), is_optimistic_value(value.tensor))

        if optimistic_mask.sum().item() == 0:
            return Value(BOOL, feature.batch_variables, rv_r, batch_dims=feature.batch_dims, quantized=True)

        rv = rv_r.clone().to(torch.int64)
        for ind in optimistic_mask.nonzero().tolist():
            ind = tuple(ind)
            this_argv = feature.tensor[ind].item(), value.tensor[ind].item()
            new_identifier = ctx.optimistic_context.new_var(BOOL)
            rv[ind] = new_identifier
            ctx.optimistic_context.add_constraint(EqualOptimisticConstraint(
                *[cvt_opt_value(v, feature.dtype) for v in this_argv],
                cvt_opt_value(new_identifier, BOOL)
            ))

        return Value(BOOL, feature.batch_variables, rv, batch_dims=feature.batch_dims, quantized=True)

    def __str__(self):
        return f'__EQ__({self.feature}, {self.value})'


class VariableAssignmentExpression(Expression, ABC):
    pass


def is_variable_assignment_expression(expr: Expression) -> bool:
    return isinstance(expr, VariableAssignmentExpression)


class AssignOp(_FeatureValueExpression, VariableAssignmentExpression):
    def __init__(self, feature: FeatureApplication, value: Expression):
        _FeatureValueExpression.__init__(self, feature, value)

    def _check_arguments(self):
        super()._check_arguments()
        assert isinstance(self.feature, (FeatureApplication, PredicateApplication)), 'AssignOp only support dest type FeatureApplication or PredicateApplication, got {}.'.format(type(self.feature))

    def _forward(self, ctx: ExpressionExecutionContext):
        argument_values = forward_args(ctx, *self.feature.arguments, force_tuple=True)
        value = forward_args(ctx, self.value)

        if ctx.state.features[self.feature.function_def.name].quantized:
            if not value.quantized:
                value = ctx.value_quantizer.quantize_value(value)
        else:
            if value.quantized:
                value = ctx.value_quantizer.unquantize_value(value)
        ctx.state.features[self.feature.feature_def.name][argument_values] = value

    def _forward_relaxed(self, ctx: ExpressionExecutionContext):
        argument_values = forward_args(ctx, *self.feature.arguments, force_tuple=True)
        value = forward_args(ctx, self.value)

        assert ctx.state.features[self.feature.function_def.name].quantized
        assert value.quantized
        current = ctx.state.features[self.feature.feature_def.name][argument_values]
        eq_mask = torch.eq(current.tensor, value.tensor)
        eq_mask = torch.logical_or(eq_mask, is_optimistic_value(current.tensor))
        value_tensor = current.tensor * eq_mask + torch.logical_not(eq_mask) * torch.tensor(RELAX_MAGIC_NUMBER + ctx.relaxed_context.op_identifier, dtype=torch.int64)
        ctx.state.features[self.feature.feature_def.name].tensor[argument_values] = value_tensor
        ctx.relaxed_context.add_backward_value(value)

    def __str__(self):
        return f'assign({self.feature}: {self.value})'


class ConditionalAssignOp(_FeatureValueExpression, VariableAssignmentExpression):
    OPTIONS = {'quantize': False}

    @staticmethod
    def set_options(**kwargs):
        ConditionalAssignOp.OPTIONS.update(kwargs)

    def __init__(self, feature: FeatureApplication, value: ValueOutputExpression, condition: ValueOutputExpression):
        self.condition = condition
        _FeatureValueExpression.__init__(self, feature, value)

    def _check_arguments(self):
        super()._check_arguments()
        assert isinstance(self.condition, ValueOutputExpression) and self.condition.output_type == BOOL

    def _forward(self, ctx: ExpressionExecutionContext):
        argument_values = forward_args(ctx, *self.feature.arguments, force_tuple=True)
        value = forward_args(ctx, self.value)
        condition = forward_args(ctx, self.condition)

        if ctx.state.features[self.feature.function_def.name].quantized:
            if not value.quantized:
                value = ctx.value_quantizer.quantize_value(value)
        else:
            if value.quantized:
                value = ctx.value_quantizer.unquantize_value(value)

        condition_tensor = jactorch.quantize(condition.tensor) if ConditionalAssignOp.OPTIONS['quantize'] else condition.tensor
        if value.tensor.dim() > condition_tensor.dim():
            condition_tensor = condition_tensor.unsqueeze(-1)

        origin_tensor = ctx.state.features[self.feature.feature_def.name].tensor[argument_values]
        assert value.tensor.dim() == condition_tensor.dim()
        ctx.state.features[self.feature.feature_def.name][argument_values] = (
            value.tensor * condition_tensor +
            origin_tensor * (1 - condition_tensor.float())
        )

    def _forward_relaxed(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, Value]]:
        raise NotImplementedError()

    def __str__(self):
        return f'cond-assign({self.feature}: {self.value} if {self.condition})'


class DeicticAssignOp(VariableAssignmentExpression):
    def __init__(self, variable: Variable, expr: Union[VariableAssignmentExpression]):
        self.variable = variable
        self.expr = expr
        self._check_arguments()

    def _check_arguments(self):
        assert isinstance(self.variable.type, ObjectType)

    def _forward(self, ctx: ExpressionExecutionContext):
        with ctx.new_bounded_variables({self.variable: QINDEX}):
            self.expr.forward(ctx)

    def _forward_relaxed(self, ctx: ExpressionExecutionContext):
        self._forward(ctx)

    def __str__(self):
        return f'deictic-assign({self.variable}: {self.expr})'


def flatten_expression(
    expr: Expression,
    mappings: Optional[Dict[Union[FeatureApplication, VariableExpression], Union[Variable, ValueOutputExpression]]] = None,
    ctx: Optional[ExpressionDefinitionContext] = None,
    flatten_cacheable_bool: bool = True,
) -> Union[AssignOp, ConditionalAssignOp, DeicticAssignOp, VariableExpression, ValueOutputExpression]:
    if ctx is None:
        ctx = ExpressionDefinitionContext()
    if mappings is None:
        mappings = {}

    with ctx.as_default():
        return _flatten_expression_inner(expr, mappings, flatten_cacheable_bool=flatten_cacheable_bool)


# @jacinle.log_function
def _flatten_expression_inner(
    expr: Expression,
    mappings: Dict[Union[FeatureApplication, VariableExpression], Union[Variable, ValueOutputExpression]],
    flatten_cacheable_bool: bool,
) -> Union[VariableExpression, ValueOutputExpression, VariableAssignmentExpression]:
    ctx = get_definition_context()

    if isinstance(expr, BoolOp):
        return BoolOp(expr.bool_op_type, [_flatten_expression_inner(e, mappings, flatten_cacheable_bool) for e in expr.arguments])
    elif isinstance(expr, QuantificationOp):
        with ctx.new_arguments(expr.variable):
            dummy_variable = ctx.generate_random_named_variable(expr.variable.type)
            mappings_inner = mappings.copy()
            mappings_inner[VariableExpression(expr.variable)] = dummy_variable
            return QuantificationOp(expr.quantifier_type, dummy_variable, _flatten_expression_inner(expr.expr, mappings_inner, flatten_cacheable_bool))
    elif isinstance(expr, FeatureEqualOp):
        return FeatureEqualOp(_flatten_expression_inner(expr.feature, mappings, flatten_cacheable_bool), _flatten_expression_inner(expr.value, mappings, flatten_cacheable_bool))
    elif isinstance(expr, FeatureApplication):
        for k, v in mappings.items():
            if not isinstance(k, FeatureApplication):
                continue
            if expr.feature_def.name == k.feature_def.name and all(
                isinstance(a1, VariableExpression) and isinstance(a2, VariableExpression) and a1.name == a2.name for a1, a2 in zip(expr.arguments, k.arguments)
            ):
                return VariableExpression(v)
        if expr.feature_def.expr is None or expr.feature_def.group in ('augmented', ) or (not flatten_cacheable_bool and expr.feature_def.cacheable and expr.output_type == BOOL):
            return type(expr)(expr.function_def, *[_flatten_expression_inner(e, mappings, flatten_cacheable_bool) for e in expr.arguments])
        else:
            for arg in expr.function_def.arguments:
                assert isinstance(arg, Variable)
            mappings_inner = mappings.copy()
            argvs = [_flatten_expression_inner(e, mappings, flatten_cacheable_bool) for e in expr.arguments]
            nctx = ExpressionDefinitionContext(*expr.function_def.arguments)
            with nctx.as_default():
                for arg, argv in zip(expr.feature_def.arguments, argvs):
                    if isinstance(arg, Variable):
                        mappings_inner[VariableExpression(arg)] = argv
                return _flatten_expression_inner(expr.feature_def.expr, mappings_inner, flatten_cacheable_bool)
    elif isinstance(expr, FunctionApplication):
        return type(expr)(expr.function_def, *[_flatten_expression_inner(e, mappings, flatten_cacheable_bool) for e in expr.arguments])
    elif isinstance(expr, ConditionalSelectOp):
        return type(expr)(_flatten_expression_inner(expr.feature, mappings, flatten_cacheable_bool), _flatten_expression_inner(expr.condition, mappings, flatten_cacheable_bool))
    elif isinstance(expr, VariableExpression):
        rv = expr
        for k, v in mappings.items():
            if isinstance(k, VariableExpression):
                if k.name == expr.name:
                    if isinstance(v, Variable):
                        rv = VariableExpression(v)
                    else:
                        rv = v
        return rv
    elif isinstance(expr, (ConstantExpression, ObjectConstantExpression)):
        return expr
    elif isinstance(expr, AssignOp):
        return AssignOp(_flatten_expression_inner(expr.feature, mappings, flatten_cacheable_bool), _flatten_expression_inner(expr.value, mappings, flatten_cacheable_bool))
    elif isinstance(expr, ConditionalAssignOp):
        return ConditionalAssignOp(_flatten_expression_inner(expr.feature, mappings, flatten_cacheable_bool), _flatten_expression_inner(expr.value, mappings, flatten_cacheable_bool), _flatten_expression_inner(expr.condition, mappings, flatten_cacheable_bool))
    elif isinstance(expr, (DeicticSelectOp, DeicticAssignOp)):
        with ctx.new_arguments(expr.variable):
            dummy_variable = ctx.generate_random_named_variable(expr.variable.type)
            mappings_inner = mappings.copy()
            mappings_inner[VariableExpression(expr.variable)] = dummy_variable
            return type(expr)(dummy_variable, _flatten_expression_inner(expr.expr, mappings_inner, flatten_cacheable_bool))
    else:
        raise TypeError('Unknown expression type: {}.'.format(type(expr)))


def iter_exprs(expr: Expression):
    """Iterate over all sub-expressions of the input."""
    yield expr
    if isinstance(expr, BoolOp):
        for arg in expr.arguments:
            yield from iter_exprs(arg)
    elif isinstance(expr, QuantificationOp):
        yield from iter_exprs(expr.expr)
    elif isinstance(expr, FeatureEqualOp):
        yield from iter_exprs(expr.feature)
        yield from iter_exprs(expr.value)
    elif isinstance(expr, FunctionApplication):
        for arg in expr.arguments:
            yield from iter_exprs(arg)
    elif isinstance(expr, AssignOp):
        yield from iter_exprs(expr.value)
    elif isinstance(expr, ConditionalSelectOp):
        yield from iter_exprs(expr.feature)
        yield from iter_exprs(expr.condition)
    elif isinstance(expr, ConditionalAssignOp):
        yield from iter_exprs(expr.value)
        yield from iter_exprs(expr.condition)
    elif isinstance(expr, (DeicticSelectOp, DeicticAssignOp)):
        yield from iter_exprs(expr.expr)
    elif isinstance(expr, (FeatureApplication, VariableExpression, ConstantExpression, ObjectConstantExpression)):
        pass
    else:
        raise TypeError('Unknown expression type: {}.'.format(type(expr)))
