import os.path as osp
import itertools
from typing import Optional, Union, Sequence, Tuple, Set
from lark import Lark, Tree, Transformer, v_args

import jacinle
import hacl.pdsketch.interface.v2.expr as E
from .value import ValueType, BasicValueType, VectorValueType, BOOL, Variable, StringConstant
from .state import State
from .expr import ExpressionDefinitionContext, get_definition_context
from .domain import Domain, Precondition, Effect, Problem

__all__ = ['PDSketchParser', 'load_domain_file', 'parse_domain_string', 'parse_expression', 'load_problem_file', 'PDDLTransformer']

logger = jacinle.get_logger(__file__)

# lark.v_args
inline_args = v_args(inline=True)
DEBUG_LOG_COMPOSE = False


def _log_function(func):
    if DEBUG_LOG_COMPOSE:
        return jacinle.log_function(func)
    return func


class PDSketchParser(object):
    grammar_file = osp.join(osp.dirname(__file__), 'pdsketch-v2.grammar')

    def __init__(self):
        with open(type(self).grammar_file) as f :
            self.lark = Lark(f)

    def load(self, file):
        with open(file) as f:
            return self.lark.parse(f.read())

    def loads(self, string):
        return self.lark.parse(string)

    def make_domain(self, tree: Tree) -> Domain:
        assert tree.children[0].data == 'definition'
        transformer = PDDLTransformer(Domain())
        transformer.transform(tree)
        domain = transformer.domain
        domain.post_init()
        return domain

    def make_problem(self, tree: Tree, domain: Domain, **kwargs) -> Problem:
        assert tree.children[0].data == 'definition'
        transformer = PDDLTransformer(domain, **kwargs)
        transformer.transform(tree)
        problem = transformer.problem
        return problem

    def make_expression(self, domain: Domain, tree: Tree, variables: Optional[Sequence[Variable]] = None) -> E.Expression:
        if variables is None:
            variables = list()
        transformer = PDDLTransformer(domain, allow_object_constants=True)
        node = transformer.transform(tree).children[0]
        assert isinstance(node, (_FunctionApplicationImm, _QuantifierApplicationImm))
        with ExpressionDefinitionContext(*variables, domain=domain).as_default():
            return node.compose()


_parser = PDSketchParser()


def load_domain_file(filename) -> Domain:
    tree = _parser.load(filename)
    domain = _parser.make_domain(tree)
    return domain


def parse_domain_string(domain_string) -> Domain:
    tree = _parser.loads(domain_string)
    domain = _parser.make_domain(tree)
    return domain


def load_problem_file(filename, domain: Domain, **kwargs) -> Tuple[State, E.ValueOutputExpression]:
    tree = _parser.load(filename)
    with ExpressionDefinitionContext(domain=domain).as_default():
        problem = _parser.make_problem(tree, domain, **kwargs)
    return problem.to_state(domain), problem.goal


def parse_expression(domain, string, variables) -> E.Expression:
    tree = _parser.loads(string)
    expr = _parser.make_expression(domain, tree, variables)
    return expr


class PDDLTransformer(Transformer):
    def __init__(self, init_domain: Optional[Domain] = None, allow_object_constants: bool = True, ignore_unknown_predicates: bool = False):
        super().__init__()

        self.domain = init_domain
        self.problem = Problem()
        self.allow_object_constants = allow_object_constants
        self.ignore_unknown_predicates = ignore_unknown_predicates
        self.ignored_predicates: Set[str] = set()

    @inline_args
    def definition_decl(self, definition_type, definition_name):
        if definition_type.value == 'domain':
            self.domain.name = definition_name.value

    def type_definition(self, args):
        # Very ugly hack to handle multi-line definition in PDDL.
        # In PDDL, type definition can be separated by newline.
        # This kinds of breaks the parsing strategy that ignores all whitespaces.
        # More specifically, consider the following two definitions:
        # ```
        # (:types
        #   a
        #   b - a
        # )
        # ```
        # and
        # ```
        # (:types
        #   a b - a
        # )
        if isinstance(args[-1], Tree) and args[-1].data == "parent_type_name":
            parent_line, parent_name = args[-1].children[0]
            args = args[:-1]
        else:
            parent_line, parent_name = -1, 'object'

        for arg in args:
            arg_line, arg_name = arg
            if arg_line == parent_line:
                self.domain.register_type(arg_name, parent_name)
            else:
                self.domain.register_type(arg_name, parent_name)

    @inline_args
    def constants_definition(self, *args):
        raise NotImplementedError()

    @inline_args
    def predicate_definition(self, name, *args):
        name, kwargs = name

        return_type = kwargs.pop('return_type', None)
        kwargs.setdefault('group', 'basic')
        if return_type is None:
            self.domain.register_predicate(name, args, **kwargs)
        else:
            self.domain.register_feature(name, args, return_type, **kwargs)

    @inline_args
    def predicate_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def feature_definition(self, name, parameters, output_type, expr):
        name, kwargs = name
        parameters = tuple(parameters.children)
        if len(expr.children) > 0:
            expr = expr.children[0]
            assert isinstance(expr, _FunctionApplicationImm), 'Expression of a feature must be a function: got {}.'.format(type(expr))
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"feature::{name}", precondition_constraints=['augmented-input'], effect_constraints=[]).as_default():
                expr = expr.compose(output_type)
        else:
            expr = None
        self.domain.register_feature(name, parameters, output_type, expr=expr, **kwargs)

    @inline_args
    def feature_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def type_name(self, name):
        # propagate the "lineno" of the type definition up.
        return name.line, name.value

    @inline_args
    def object_type_name(self, type):
        return type

    @inline_args
    def value_type_name(self, type):
        lineno, type = type
        if isinstance(type, VectorValueType):
            return lineno, type
        assert isinstance(type, str)
        if type in self.domain.types:
            return lineno, self.domain.types[type]
        return lineno, BasicValueType(type)

    @inline_args
    def vector_type_name(self, dtype, dim, choices, kwargs=None):
        choices = choices.children[0] if len(choices.children) > 0 else 0
        if kwargs is None:
            kwargs = dict()
        lineno, dtype = dtype
        return lineno, VectorValueType(dtype, dim, choices, **kwargs)

    @inline_args
    def object_type_name_unwrapped(self, type):
        return type[1]

    @inline_args
    def value_type_name_unwrapped(self, type):
        return type[1]

    @inline_args
    def predicate_group_definition(self, *args):
        raise NotImplementedError()

    @inline_args
    def action_definition(self, name, parameters, precondition, effect):
        name, kwargs = name
        parameters = tuple(parameters.children)
        precondition = precondition.children[0]
        effect = effect.children[0]

        with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}", effect_constraints=[]).as_default():
            precondition = _canonize_precondition(precondition)
        with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}", effect_constraints=['basic', 'augmented']).as_default():
            effect = _canonize_effect(effect)
        self.domain.register_operator(name, parameters, precondition, effect, **kwargs)

    @inline_args
    def action_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def axiom_definition(self, decorator, vars, context, implies):
        kwargs = dict() if len(decorator.children) == 0 else decorator.children[0]
        vars = tuple(vars.children)
        precondition = context.children[0]
        effect = implies.children[0]

        name = kwargs.pop('name', None)
        scope = None if name is None else f"axiom::{name}"

        with ExpressionDefinitionContext(*vars, domain=self.domain, scope=scope, effect_constraints=[]).as_default():
            precondition = _canonize_precondition(precondition)
        with ExpressionDefinitionContext(*vars, domain=self.domain, scope=scope, effect_constraints=['basic', 'augmented']).as_default():
            effect = _canonize_effect(effect)
        self.domain.register_axiom(name, vars, precondition, effect, **kwargs)

    @inline_args
    def derived_definition(self, signature, expr):
        name, args, kwargs = signature
        expr = expr

        return_type = kwargs.pop('return_type', BOOL)
        with ExpressionDefinitionContext(*args, domain=self.domain, scope=f"derived::{name}", effect_constraints=[]).as_default():
            expr = expr.compose(return_type)
        self.domain.register_derived(name, args, return_type, expr=expr, **kwargs)

    @inline_args
    def derived_signature(self, name, *args):
        name, kwargs = name
        return name, args, kwargs

    @inline_args
    def derived_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def generator_definition(self, name, parameters, certifies, context, generates):
        name, kwargs = name
        parameters = tuple(parameters.children)
        certifies = certifies.children[0]
        context = context.children[0]
        generates = generates.children[0]

        with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"generator::{name}", effect_constraints=[]).as_default():
            certifies = certifies.compose(BOOL)

        ctx = ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"generator::{name}", effect_constraints=[])
        with ctx.as_default():
            assert context.name == 'and'
            context = [_compose(ctx, c) for c in context.arguments]
            assert generates.name == 'and'
            generates = [_compose(ctx, c) for c in generates.arguments]

        self.domain.register_generator(name, parameters, certifies, context, generates, **kwargs)

    @inline_args
    def generator_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def object_definition(self, constant):
        self.problem.add_object(constant.name, constant.typename)

    @inline_args
    def init_definition_item(self, function_call):
        if function_call.name not in self.domain.features:
            if self.ignore_unknown_predicates:
                if function_call.name not in self.ignored_predicates:
                    logger.warning(f"Unknown predicate: {function_call.name}.")
                    self.ignored_predicates.add(function_call.name)
            else:
                raise ValueError(f"Unknown predicate: {function_call.name}.")
            return
        self.problem.add_predicate(function_call.compose())

    @inline_args
    def goal_definition(self, function_call):
        self.problem.set_goal(function_call.compose())

    @inline_args
    def variable(self, name) -> Variable:
        return Variable(name.value)

    @inline_args
    def typedvariable(self, name, typename):
        # name is of type `Variable`.
        return Variable(name.name, self.domain.types[typename])

    @inline_args
    def constant(self, name) -> StringConstant:
        assert self.allow_object_constants
        return StringConstant(name.value)

    @inline_args
    def typedconstant(self, name, typename):
        return StringConstant(name.name, self.domain.types[typename])

    @inline_args
    def bool(self, v):
        return v.value == 'true'

    @inline_args
    def int(self, v):
        return int(v.value)

    @inline_args
    def float(self, v):
        return float(v.value)

    @inline_args
    def string(self, v):
        return v.value[1:-1]

    @inline_args
    def list(self, *args):
        return list(args)

    @inline_args
    def decorator_k(self, k):
        return k.value

    @inline_args
    def decorator_v(self, v):
        return v

    @inline_args
    def decorator_kwarg(self, k, v):
        return (k, v)

    def decorator_kwargs(self, args):
        return {k: v for k, v in args}

    @inline_args
    def slot(self, _, name, kwargs=None):
        return _Slot(name.children[0].value, kwargs)

    @inline_args
    def function_name(self, name):
        return name

    @inline_args
    def method_name(self, feature_name, _, method_name):
        return _MethodName(feature_name, method_name)

    @inline_args
    def function_call(self, name, *args):
        assert isinstance(name, (str, _MethodName, _Slot))
        return _FunctionApplicationImm(name, args)

    @inline_args
    def simple_function_call(self, name, *args):
        return _FunctionApplicationImm(name.value, args)

    @inline_args
    def pm_function_call(self, pm_sign, function_call):
        if pm_sign.value == '+':
            return function_call
        else:
            return _FunctionApplicationImm('not', [function_call])

    @inline_args
    def quantified_function_call(self, quantifier, variable, expr):
        return _QuantifierApplicationImm(quantifier, variable, expr)


class _FunctionApplicationImm(object):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    def __str__(self):
        arguments_str = ', '.join([str(arg) for arg in self.arguments])
        return f'IMM::{self.name}({arguments_str})'

    __repr__ = jacinle.repr_from_str

    @_log_function
    def compose(self, expect_value_type: Optional[ValueType] = None, check_function_call_group=True):
        ctx = get_definition_context()
        if isinstance(self.name, _Slot):
            assert ctx.scope is not None, 'Cannot define slots inside anonymous actino/axioms.'

            name = ctx.scope + '::' + self.name.name
            arguments = self._compose_arguments(ctx, self.arguments)
            argument_types = [arg.variable if isinstance(arg, E.VariableExpression) else arg.output_type for arg in arguments]
            return_type = self.name.kwargs.pop('return_type', None)
            if return_type is None:
                assert expect_value_type is not None, f'Cannot infer return type for function {name}; please specify by [return_type=Type]'
                return_type = expect_value_type
            else:
                if expect_value_type is not None:
                    assert return_type == expect_value_type, f'Return type mismatch for function {name}: expect {expect_value_type}, got {return_type}.'
            function_def = ctx.domain.register_external_function(name, argument_types, return_type, kwargs=self.name.kwargs)
            return E.ExternalFunctionApplication(function_def, *arguments)
        elif isinstance(self.name, _MethodName):
            assert self.name.feature_name in ctx.domain.features, 'Unkwown feature: {}.'.format(self.name.feature_name)
            predicate_def = ctx.domain.features[self.name.feature_name]

            if self.name.method_name == 'equal':
                nr_index_arguments = len(self.arguments) - 1
            elif self.name.method_name == 'assign':
                nr_index_arguments = len(self.arguments) - 1
            elif self.name.method_name == 'cond-select':
                nr_index_arguments = len(self.arguments) - 1
            elif self.name.method_name == 'cond-assign':
                nr_index_arguments = len(self.arguments) - 2
            else:
                raise NameError('Unknown method name: {}.'.format(self.name.method_name))

            arguments = self._compose_arguments(ctx, self.arguments[:nr_index_arguments], predicate_def.arguments)
            with ctx.mark_is_effect_definition(False):
                value = self._compose_arguments(ctx, [self.arguments[-1]], predicate_def.output_type.assignment_type())[0]

            if isinstance(predicate_def, E.PredicateDef):
                feature = E.PredicateApplication(predicate_def, *arguments)
            else:
                feature = E.FeatureApplication(predicate_def, *arguments)

            if self.name.method_name == 'equal':
                ctx.check_precondition(predicate_def, ctx.scope)
                return E.FeatureEqualOp(feature, value)
            elif self.name.method_name == 'assign':
                ctx.check_effect(predicate_def, ctx.scope)
                return E.AssignOp(feature, value)
            elif self.name.method_name == 'cond-select':
                with ctx.mark_is_effect_definition(False):
                    condition = self._compose_arguments(ctx, [self.arguments[-1]], BOOL)[0]
                return E.ConditionalSelectOp(feature, condition)
            elif self.name.method_name == 'cond-assign':
                with ctx.mark_is_effect_definition(False):
                    condition = self._compose_arguments(ctx, [self.arguments[-2]], BOOL)[0]
                ctx.check_effect(predicate_def, ctx.scope)
                return E.ConditionalAssignOp(feature, value, condition)
            else:
                raise NameError('Unknown method name: {}.'.format(self.name.method_name))
        elif self.name == 'and':
            arguments = [arg.compose(expect_value_type) for arg in self.arguments]
            return E.AndOp(*arguments)
        elif self.name == 'or':
            arguments = [arg.compose(expect_value_type) for arg in self.arguments]
            return E.OrOp(*arguments)
        elif self.name == 'not':
            arguments = [arg.compose(expect_value_type) for arg in self.arguments]
            return E.NotOp(*arguments)
        elif self.name == 'equal':
            assert len(self.arguments) == 2, 'FeatureEqualOp takes two arguments, got: {}.'.format(len(self.arguments))
            feature = self.arguments[0]
            feature = _compose(ctx, feature, None)
            value = self.arguments[1]
            value = _compose(ctx, value, feature.output_type.assignment_type())
            return E.FeatureEqualOp(feature, value)
        elif self.name == 'assign':
            assert len(self.arguments) == 2, 'AssignOp takes two arguments, got: {}.'.format(len(self.arguments))
            assert isinstance(self.arguments[0], _FunctionApplicationImm)
            feature = self.arguments[0].compose(None, check_function_call_group=False)
            assert isinstance(feature, E.FeatureApplication)
            ctx.check_effect(feature.function_def, ctx.scope)
            value = self.arguments[1]
            with ctx.mark_is_effect_definition(False):
                value = _compose(ctx, value, feature.output_type.assignment_type())
            return E.AssignOp(feature, value)
        elif self.name in ctx.domain.features:
            predicate_def = ctx.domain.features[self.name]
            arguments = self._compose_arguments(ctx, self.arguments, predicate_def.arguments)
            if check_function_call_group:
                ctx.check_precondition(predicate_def, ctx.scope)
            if isinstance(predicate_def, E.PredicateDef):
                return E.PredicateApplication(predicate_def, *arguments)
            else:
                return E.FeatureApplication(predicate_def, *arguments)
        else:
            raise ValueError('Unknown function: {}.'.format(self.name))

    def _compose_arguments(self, ctx, arguments, expect_value_type=None):
        if isinstance(expect_value_type, (tuple, list)):
            assert len(expect_value_type) == len(arguments), 'Mismatched number of arguments: expect {}, got {}. Expression: {}.'.format(len(expect_value_type), len(arguments), self)
            return [ _compose(ctx, arg, evt) for arg, evt in zip(arguments, expect_value_type) ]
        return [ _compose(ctx, arg, expect_value_type) for arg in arguments ]


def _compose(ctx, arg, evt=None):
    if isinstance(arg, Variable):
        return ctx[arg.name]
    elif isinstance(arg, StringConstant):
        return E.ObjectConstantExpression(arg)
    else:
        return arg.compose(evt)


class _Slot(object):
    def __init__(self, name, kwargs=None):
        self.scope = None
        self.name = name
        self.kwargs = kwargs

        if self.kwargs is None:
            self.kwargs = dict()

    def set_scope(self, scope):
        self.scope = scope

    def __str__(self):
        kwargs = ', '.join([f'{k}={v}' for k, v in self.kwargs.items()])
        return f'??{self.name}[{kwargs}]'


class _MethodName(object):
    def __init__(self, feature_name, method_name):
        self.feature_name = feature_name
        self.method_name = method_name

    def __str__(self):
        return f'{self.feature_name}::{self.method_name}'


class _QuantifierApplicationImm(object):
    def __init__(self, quantifier, darg: Variable, expr: _FunctionApplicationImm):
        self.quantifier = quantifier
        self.darg = darg
        self.expr = expr

    def __str__(self):
        return f'QIMM::{self.quantifier}({self.darg}: {self.expr})'

    @_log_function
    def compose(self, expect_value_type: Optional[ValueType] = None):
        ctx = get_definition_context()

        with ctx.new_arguments(self.darg):
            if ctx.is_effect_definition:
                expr = _canonize_effect(self.expr)
            else:
                expr = self.expr.compose(expect_value_type)

        if self.quantifier in ('foreach', 'forall') and ctx.is_effect_definition:
            outputs = list()
            for e in expr:
                assert isinstance(e, Effect)
                outputs.append(E.DeicticAssignOp(self.darg, e.assign_expr))
            return outputs
        if self.quantifier == 'foreach':
            assert E.is_value_output_expression(expr)
            return E.DeicticSelectOp(self.darg, expr)

        assert E.is_value_output_expression(expr)
        return E.QuantificationOp(E.QuantifierType.from_string(self.quantifier), self.darg, expr)


@_log_function
def _canonize_precondition(precondition: Union[_FunctionApplicationImm, _QuantifierApplicationImm]):
    if isinstance(precondition, _FunctionApplicationImm) and precondition.name == 'and':
        return list(itertools.chain(*[_canonize_precondition(pre) for pre in precondition.arguments]))
    return [Precondition(precondition.compose(BOOL))]


@_log_function
def _canonize_effect(effect: Union[_FunctionApplicationImm, _QuantifierApplicationImm]):
    if isinstance(effect, _QuantifierApplicationImm):
        effect = effect.compose()
        if isinstance(effect, list):
            effect = [Effect(e) for e in effect]
    else:
        assert isinstance(effect, _FunctionApplicationImm)

        if effect.name == 'and':
            return list(itertools.chain(*[_canonize_effect(eff) for eff in effect.arguments]))

        if isinstance(effect.name, _MethodName):
            effect = effect.compose()
        elif effect.name == 'assign':
            effect = effect.compose()
        elif effect.name == 'not':
            assert len(effect.arguments) == 1, 'NotOp only takes 1 argument, got {}.'.format(len(effect.arguments))
            feat = effect.arguments[0].compose()
            assert feat.output_type == BOOL
            effect = E.AssignOp(feat, E.ConstantExpression.FALSE)
        elif effect.name == 'when':
            assert len(effect.arguments) == 2, 'WhenOp takes two arguments, got: {}.'.format(len(effect.arguments))
            condition = effect.arguments[0].compose(BOOL)
            if effect.arguments[1].name == 'and':
                inner_effects = effect.arguments[1].arguments
            else:
                inner_effects = [effect.arguments[1]]
            inner_effects = list(itertools.chain(*[_canonize_effect(arg) for arg in inner_effects]))
            effect = list()
            for e in inner_effects:
                assert isinstance(e.assign_expr, E.AssignOp)
                effect.append(Effect(E.ConditionalAssignOp(e.assign_expr.feature, e.assign_expr.value, condition)))
            return effect
        else:
            feat = effect.compose()
            assert isinstance(feat, E.FeatureApplication) and feat.output_type == BOOL
            effect = E.AssignOp(feat, E.ConstantExpression.TRUE)

    if isinstance(effect, list):
        return effect

    assert isinstance(effect, E.VariableAssignmentExpression)
    return [Effect(effect)]

