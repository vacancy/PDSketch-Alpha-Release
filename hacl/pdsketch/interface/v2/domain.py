import itertools
import networkx as nx
import torch
from typing import Any, Iterable, Optional, Union, Tuple, Sequence, List, Mapping, Dict

import jacinle
from jacinle.utils.printing import indent_text, stprint, stformat
from .value import ObjectType, BasicValueType, NamedValueType, NamedValueTypeSlot, VectorValueType, BOOL, INT64, Value, Variable, is_intrinsically_quantized, QINDEX
from .optimistic import OptimisticConstraint, EqualOptimisticConstraint, OptimisticValueContext, is_optimistic_value, DeterminedValue, RelaxedExecutionContext
from .state import StateLike, State, ValueDict
from .expr import ExpressionDefinitionContext, ExpressionExecutionContext, PredicateDef, FeatureDef, FunctionDef, Expression
from .expr import ConstantExpression, VariableExpression, ObjectConstantExpression, ValueOutputExpression, VariableAssignmentExpression
from .expr import FunctionApplication, FeatureApplication, BoolOp, QuantificationOp, FeatureEqualOp, AssignOp, ConditionalAssignOp, DeicticAssignOp
from .expr import flatten_expression

__all__ = [
    'Precondition', 'Effect', 'Operator', 'OperatorApplier',
    'GeneratorDef', 'Domain', 'Problem',
    'get_used_features_and_functions', 'get_used_functions',
    'get_feature_and_function_dependencies', 'AugmentedFeatureStage', 'build_augmented_feature_stages',
    'extract_generator_data',
    'ValueQuantizer'
]

logger = jacinle.get_logger(__file__)


class Precondition(object):
    def __init__(self, bool_expr: ValueOutputExpression):
        self.bool_expr = bool_expr
        self.ao_discretization = None

    def forward(self, ctx: ExpressionExecutionContext):
        return self.bool_expr.forward(ctx)

    def __str__(self):
        return str(self.bool_expr)

    __repr__ = jacinle.repr_from_str


class Effect(object):
    def __init__(self, assign_expr: VariableAssignmentExpression):
        self.assign_expr = assign_expr
        self.ao_discretization = None

    def forward(self, ctx: ExpressionExecutionContext):
        return self.assign_expr.forward(ctx)

    @property
    def unwrapped_assign_expr(self) -> Union[AssignOp, ConditionalAssignOp]:
        """Unwrap the DeicticAssignOps and return the innermost AssignOp."""
        expr = self.assign_expr
        if isinstance(expr, DeicticAssignOp):
            expr = expr.expr
        assert isinstance(expr, (AssignOp, ConditionalAssignOp))
        return expr

    def __str__(self):
        return str(self.assign_expr)

    __repr__ = jacinle.repr_from_str


class Operator(object):
    def __init__(self, domain: 'Domain', name: str, arguments: Sequence[Variable], preconditions: Sequence[Precondition], effects: Sequence[Effect], is_axiom: Optional[bool] = False):
        self.domain = domain
        self.name = name
        self.arguments = arguments
        self.preconditions = preconditions
        self.effects = effects
        self.is_axiom = is_axiom

    @property
    def nr_arguments(self) -> int:
        return len(self.arguments)

    def __call__(self, *args: Union[int, str, NamedValueTypeSlot, torch.Tensor, Value]):
        output_args = list()
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg == '??':
                output_args.append(NamedValueTypeSlot(self.arguments[i].type))
            else:
                output_args.append(arg)
        return OperatorApplier(self, *output_args)

    def apply_precondition(self, state: State, *args, optimistic_context: Optional[OptimisticValueContext] = None, relaxed_context: Optional[RelaxedExecutionContext] = None) -> bool:
        ctx = ExpressionExecutionContext(self.domain, state, bounded_variables=state.compose_bounded_variables(self.arguments, args), optimistic_context=optimistic_context, relaxed_context=relaxed_context)

        all_rvs = list()
        with ctx.as_default():
            for pred in self.preconditions:
                try:
                    pred_value = pred.forward(ctx)
                    all_rvs.append(pred_value)
                    rv = pred_value.item()
                except Exception:
                    logger.warning('Precondition evaluation failed: {}.'.format(pred.bool_expr))
                    raise
                if is_optimistic_value(rv):
                    if optimistic_context is not None:
                        optimistic_context.add_constraint(EqualOptimisticConstraint.from_bool(rv, True))
                    if relaxed_context is not None:
                        relaxed_context.add_backward_value(pred_value)
                else:
                    if rv < 0.5:
                        return False
        return True

    def apply_effect(self, state: State, *args, optimistic_context: Optional[OptimisticValueContext] = None, relaxed_context: Optional[RelaxedExecutionContext] = None, clone: Optional[bool] = True) -> State:
        if clone:
            state = state.clone()
        ctx = ExpressionExecutionContext(self.domain, state, bounded_variables=state.compose_bounded_variables(self.arguments, args), optimistic_context=optimistic_context, relaxed_context=relaxed_context)
        with ctx.as_default():
            for effect in self.effects:
                try:
                    effect.forward(ctx)
                except Exception:
                    logger.warning('Effect application failed: {}.'.format(effect.assign_expr))
                    raise
            return state

    def apply(self, state: State, *args, optimistic_context: Optional[OptimisticValueContext] = None, relaxed_context: Optional[RelaxedExecutionContext] = None, clone: Optional[bool] = True) -> Tuple[bool, State]:
        if self.apply_precondition(state, *args, optimistic_context=optimistic_context, relaxed_context=relaxed_context):
            return True, self.apply_effect(state, *args, optimistic_context=optimistic_context, relaxed_context=relaxed_context, clone=clone)
        return False, state

    def __str__(self):
        if not self.is_axiom:
            def_name = 'action'
        else:
            def_name= 'axiom'
        arg_string = ', '.join([str(arg) for arg in self.arguments])
        return f'{def_name}::{self.name}({arg_string})'

    __repr__ = jacinle.repr_from_str

    def pddl_str(self) -> str:
        if not self.is_axiom:
            def_name, def_name_a, def_name_p, def_name_e = f'action {self.name}', 'parameters', 'precondition', 'effect'
        else:
            def_name, def_name_a, def_name_p, def_name_e = 'axiom', 'vars', 'context', 'implies'
        arg_string = ' '.join([str(arg) for arg in self.arguments])
        pre_string = '\n'.join([indent_text(str(pre), 2, tabsize=2) for pre in self.preconditions])
        eff_string = '\n'.join([indent_text(str(eff), 2, tabsize=2) for eff in self.effects])
        return f'''(:{def_name}
  :{def_name_a} ({arg_string})
  :{def_name_p} (and
    {pre_string.lstrip()}
  )
  :{def_name_e} (and
    {eff_string.lstrip()}
  )
)'''


class OperatorApplier(object):
    def __init__(self, operator: Operator, *args):
        self.operator = operator
        self.arguments = args

    @property
    def name(self):
        return self.operator.name

    def apply_precondition(self, state: State, optimistic_context: Optional[OptimisticValueContext] = None, relaxed_context: Optional[RelaxedExecutionContext] = None) -> bool:
        return self.operator.apply_precondition(state, *self.arguments, optimistic_context=optimistic_context, relaxed_context=relaxed_context)

    def apply_effect(self, state: State, optimistic_context: Optional[OptimisticValueContext] = None, relaxed_context: Optional[RelaxedExecutionContext] = None, clone: Optional[bool] = True) -> State:
        return self.operator.apply_effect(state, *self.arguments, optimistic_context=optimistic_context, relaxed_context=relaxed_context, clone=clone)

    def __call__(self, state: State, optimistic_context: Optional[OptimisticValueContext] = None, relaxed_context: Optional[RelaxedExecutionContext] = None, clone: Optional[bool] = True) -> Tuple[bool, State]:
        if self.apply_precondition(state, optimistic_context=optimistic_context, relaxed_context=relaxed_context):
            return True, self.apply_effect(state, optimistic_context=optimistic_context, relaxed_context=relaxed_context, clone=clone)
        return False, state

    def __str__(self):
        if not self.operator.is_axiom:
            def_name = 'action'
        else:
            def_name = 'axiom'
        arg_string = ', '.join([
            f'{arg.dtype}::{arg.item()}' if isinstance(arg, Value) and arg.quantized and arg.total_batch_dims == 0 else str(arg)
            for arg in self.arguments
        ])
        return f'{def_name}::{self.operator.name}({arg_string})'

    __repr__ = jacinle.repr_from_str


class GeneratorDef(object):
    def __init__(self, name, arguments, certifies, context, generates, function_def, output_vars, flatten_certifies, priority=0):
        self.name = name
        self.arguments = arguments
        self.certifies = certifies
        self.context = context
        self.generates = generates
        self.function_def = function_def
        self.output_vars = output_vars
        self.flatten_certifies = flatten_certifies
        self.priority = priority

    @property
    def input_vars(self):
        return self.function_def.arguments

    def __str__(self):
        arg_string = ', '.join([str(c) for c in self.context])
        gen_string = ', '.join([str(c) for c in self.generates])
        return (f'generator::{self.name}({arg_string}) -> {gen_string}' + ' {\n'
                + '  ' + str(self.function_def) + '\n'
                + '  ' + str(self.output_vars) + '\n'
                + '  ' + str(self.flatten_certifies) + '\n'
                + '}')

    __repr__ = jacinle.repr_from_str


class PythonFunctionRef(object):
    def __init__(self, function, function_quantized=None, auto_broadcast=True):
        self.function = function
        self.function_quantized = function_quantized
        self.auto_broadcast = auto_broadcast

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __str__(self):
        return f'PythonFunctionRef({self.function}, fq={self.function_quantized}, auto_broadcast={self.auto_broadcast})'

    def __repr__(self):
        return self.__str__()


class _PredicateView(object):
    def __init__(self, features):
        self.features = features

    def get(self, key, default=None):
        if key in self.features and isinstance(self.features[key], PredicateDef):
            return self.features[key]
        return default

    def keys(self):
        return [k for k, v in self.features.items() if isinstance(v, PredicateDef)]

    def values(self):
        return [v for _, v in self.features.items() if isinstance(v, PredicateDef)]

    def items(self):
        return [(k, v) for k, v in self.features.items() if isinstance(v, PredicateDef)]

    def __iter__(self):
        yield from self.keys()

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, key):
        if key in self.features:
            if isinstance(self.features[key], PredicateDef):
                return self.features[key]
        raise KeyError('Unknown predicate name: {}.'.format(key))


class Domain(object):
    name: str
    types: Dict[str, Union[ObjectType, NamedValueType]]
    predicate_groups: Dict[str, List[str]]
    features: Dict[str, FeatureDef]
    operators: Dict[str, Operator]
    axioms: Dict[str, Operator]
    external_functions: Dict[str, FunctionDef]
    external_functions_implementation: Dict[str, PythonFunctionRef]
    generators: Dict[str, GeneratorDef]

    def __init__(self):
        super().__init__()

        self.name = ''
        self.types = dict()
        self.predicate_groups = {
            '@basic': list(),
            '@augmented-input': list(),
            '@augmented': list(),
            '@goal': list(),
            '@derived': list()
        }
        self.features = dict()
        self.predicates = _PredicateView(self.features)
        self.operators = dict()
        self.axioms = dict()
        self.external_functions = dict()
        self.external_functions_implementation = dict()
        self.generators = dict()
        self.value_quantizer: Optional[ValueQuantizer] = None
        self.register_type('object')

    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError

        if item.startswith('t_'):
            return self.types[item[2:]]
        elif item.startswith('p_') or item.startswith('f_'):
            return self.features[item[2:]]
        elif item.startswith('op_'):
            return self.operators[item[3:]]
        elif item.startswith('gen_'):
            return self.generators[item[4:]]
        raise NameError('Unknown attribute: {}.'.format(item))

    def set_name(self, name: str):
        self.name = name

    def set_value_quantizer(self, value_quantizer: 'ValueQuantizer'):
        self.value_quantizer = value_quantizer

    def register_type(self, typename, parent_name: Optional[Union[VectorValueType, BasicValueType, str]] = None):
        if typename in self.types:
            if typename == 'object':
                return
            raise ValueError('Type {} already exists.'.format(typename))
        if parent_name == 'int64':
            dtype = NamedValueType(typename, INT64)
            self.types[typename] = dtype
            self.register_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL, {})
            self.register_external_function_implementation(f'type::{typename}::equal', lambda x, y: Value(BOOL, x.batch_variables, torch.eq(x.tensor, y.tensor), x.batch_dims, quantized=True))
        elif isinstance(parent_name, VectorValueType):
            dtype = NamedValueType(typename, parent_name)
            self.types[typename] = dtype
            self.register_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL, {})
        else:
            assert isinstance(parent_name, str) or parent_name is None
            if parent_name is not None:
                if parent_name not in self.types:
                    raise ValueError('Unknown type name when inheriting a parent type: {}.'.format(parent_name))
                assert parent_name == 'object', 'All object type must inherit from object.'
            else:
                assert typename == 'object', 'All object type must inherit from object.'
            self.types[typename] = ObjectType(typename)

    def register_predicate(self, name, arguments, group='basic'):
        if name in self.features:
            raise ValueError('Predicate/Feature {} already exists.'.format(name))
        assert group in ('basic', )
        feature_def = PredicateDef(name, arguments)
        self._register_feature_inner(name, feature_def, group)

    def register_feature(self, name, parameters, output_type, expr=None, group='augmented', **kwargs):
        assert group in ('basic', 'augmented-input', 'augmented')
        feature_def = FeatureDef(name, output_type, parameters, expr=expr, **kwargs)
        self._register_feature_inner(name, feature_def, group)

    def register_derived(self, name, parameters, output_type, expr=None, group='derived', **kwargs):
        if name in self.features:
            raise ValueError('Predicate/Feature {} already exists.'.format(name))
        assert group in ('goal', 'derived')
        feature_def = FeatureDef(name, output_type, parameters, expr=expr, derived=True, **kwargs)
        self._register_feature_inner(name, feature_def, group)

    def _register_feature_inner(self, name, feature_def, group):
        if name in self.features:
            raise ValueError('Predicate/Feature {} already exists.'.format(name))

        if group in ('basic', 'augmented-input'):
            assert feature_def.expr is None, 'Basic/Input features can not have an expression.'

        self.features[name] = feature_def
        feature_def.set_group(group)
        self.predicate_groups['@' + group].append(name)

        if not self.features[name].cacheable and self.features[name].expr is None:
            identifier = f'feature::{name}'
            self.external_functions[identifier] = feature_def

    def register_operator(self, name, parameters: Sequence[Variable], preconditions, effects):
        self.operators[name] = op = Operator(self, name, parameters, preconditions, effects)
        return op

    def register_external_function(self, identifier, arguments, return_type, kwargs: Optional[Mapping[str, Any]] = None):
        function_def = FunctionDef(identifier, return_type, arguments, **kwargs)
        self.external_functions[identifier] = function_def
        return function_def

    def register_external_function_implementation(self, identifier, function, function_quantized=None, auto_broadcast=True, notexists_ok=True):
        if identifier not in self.external_functions:
            if notexists_ok:
                logger.warning('Unknown external function: {}.'.format(identifier))
                return
            raise NameError('Unknown external function: {}.'.format(identifier))
        self.external_functions_implementation[identifier] = PythonFunctionRef(
            function,
            function_quantized,
            auto_broadcast=auto_broadcast
        )

    def register_axiom(self, name, parameters, preconditions, effects):
        if name is None:
            name = f'axiom_{len(self.axioms)}'
        self.axioms[name] = op = Operator(self, name, parameters, preconditions, effects, is_axiom=True)
        return op

    def register_generator(self, name, parameters, certifies, context, generates, priority=0):
        identifier = f'generator::{name}'
        arguments = [Variable(f'?c{i}', c.output_type) for i, c in enumerate(context)]
        output_type = [target.output_type for target in generates]
        output_vars = [Variable(f'?g{i}', g.output_type) for i, g in enumerate(generates)]
        function_def = FunctionDef(identifier, output_type, arguments)

        all_variables = {c: cv for c, cv in zip(context, arguments)}
        all_variables.update({g: gv for g, gv in zip(generates, output_vars)})
        ctx = ExpressionDefinitionContext(*arguments, *output_vars, domain=self)
        flatten_certifies = flatten_expression(certifies, all_variables, ctx)

        self.external_functions[identifier] = function_def
        self.generators[name] = gen = GeneratorDef(name, parameters, certifies, context, generates, function_def, output_vars, flatten_certifies, priority=priority)
        return gen

    def has_external_function(self, name):
        return name in self.external_functions_implementation

    def get_external_function(self, name):
        if name in self.external_functions_implementation:
            return self.external_functions_implementation[name]
        if name in self.features:
            return self.external_functions_implementation['feature::' + name]
        raise KeyError('Unknown external function: {}.'.format(name))

    def parse(self, string: Union[str, Expression], variables: Optional[Sequence[Variable]] = None) -> Expression:
        if isinstance(string, Expression):
            return string
        from .parser import parse_expression
        return parse_expression(self, string, variables)

    def compile(self, expr: Union[str, Expression], variables: Optional[Sequence[Variable]] = None):
        expr = self.parse(expr, variables=variables)

        def func(state, bounded_variables=None, optimistic_context: Optional[OptimisticValueContext] = None, relaxed_context: Optional[RelaxedExecutionContext] = None):
            if bounded_variables is None:
                bounded_variables = dict()
            ctx = ExpressionExecutionContext(self, state, bounded_variables=bounded_variables, optimistic_context=optimistic_context, relaxed_context=relaxed_context)
            with ctx.as_default():
                return expr.forward(ctx)

        return func

    def eval(
        self,
        expr: Union[str, Expression],
        state: StateLike,
        bounded_variables: Optional[Dict[str, Mapping[str, Any]]] = None,
        optimistic_context: Optional[OptimisticValueContext] = None,
        relaxed_context: Optional[RelaxedExecutionContext] = None
    ) -> Value:
        from .compiled_operator import CompiledExpression
        if isinstance(expr, CompiledExpression):
            if relaxed_context is not None:
                return expr.fforward_relaxed(relaxed_context, state)
            else:
                return expr.fforward(state)

        if bounded_variables is None:
            bounded_variables = dict()

        if isinstance(expr, str):
            variables = [Variable(name, self.types[typename]) for typename, args in bounded_variables.items() for name in args]
            expr = self.parse(expr, variables=variables)

        for typename, args in bounded_variables.items():
            typedef = self.types[typename]
            if isinstance(typedef, NamedValueType):
                for name, value in args.items():
                    if not isinstance(value, Value):
                        value = Value(typedef, [], value)
                        args[name] = value
            else:
                assert isinstance(typedef, ObjectType)
                for name, value in args.items():
                    if state.batch_dims > 0:
                        value = torch.tensor(state.get_typed_index(self.name), dtype=torch.int64)
                    else:
                        value = state.get_typed_index(self.name)
                    args[name] = value

        ctx = ExpressionExecutionContext(self, state, bounded_variables=bounded_variables, optimistic_context=optimistic_context, relaxed_context=relaxed_context)
        with ctx.as_default():
            return expr.forward(ctx)

    def feature_in_group(self, function_def: Union[FeatureDef, str], allowed_groups: Iterable[str]):
        if isinstance(function_def, FunctionDef):
            function_name = function_def.name
        else:
            assert isinstance(function_def, str)
            function_name = function_def

        for group in allowed_groups:
            if function_name in self.predicate_groups['@' + group]:
                return True
        return False

    def forward_expr(self, state: StateLike, arguments: Union[Sequence[Variable], Dict[Variable, int]], expr: ValueOutputExpression, return_ctx: Optional[bool] = False) -> Union[Value, Tuple[ExpressionExecutionContext, Value]]:
        bounded_variables = dict()

        if isinstance(arguments, Dict):
            for var, index in arguments.items():
                if var.typename not in bounded_variables:
                    bounded_variables[var.typename] = dict()
                bounded_variables[var.typename][var.name] = index
        else:
            for arg in arguments:
                if arg.type.typename not in bounded_variables:
                    bounded_variables[arg.type.typename] = dict()
                bounded_variables[arg.type.typename][arg.name] = QINDEX
        ctx = ExpressionExecutionContext(self, state, bounded_variables=bounded_variables)
        with ctx.as_default():
            value = expr.forward(ctx)
            if return_ctx:
                return ctx, value
            return value

    def forward_feature(self, state: StateLike, feature_def: FeatureDef):
        if feature_def.expr is None:
            assert feature_def.name in state.features
            return

        result = self.forward_expr(state, feature_def.arguments, feature_def.expr)
        state.features.set_feature(feature_def.name, result)

    def forward_augmented_features(self, state: StateLike):
        for feature_def in self.features.values():
            if self.feature_in_group(feature_def, ['basic', 'augmented-input', 'derived', 'goal']):
                continue
            else:
                assert self.feature_in_group(feature_def, ['augmented'])
                if feature_def.cacheable:
                    self.forward_feature(state, feature_def)

    def forward_derived_features(self, state: StateLike):
        for feature_def in self.features.values():
            if self.feature_in_group(feature_def, ['basic', 'augmented-input', 'augmented']):
                continue
            else:
                assert self.feature_in_group(feature_def, ['derived', 'goal'])
                if feature_def.cacheable:
                    self.forward_feature(state, feature_def)

    def forward_axioms(self, state: StateLike):
        for op in self.axioms.values():
            _, state = op.apply(state)
        return state

    def forward_features_and_axioms(self, state: StateLike, forward_augmented: Optional[bool] = True, forward_axioms: Optional[bool] = True, forward_derived: Optional[bool] = True):
        if forward_augmented:
            self.forward_augmented_features(state)
        if forward_axioms:
            state = self.forward_axioms(state)
        if forward_derived:
            self.forward_derived_features(state)
        return state

    def print_summary(self):
        print(f'Domain <{self.name}>')
        stprint(key='Types: ', data=self.types, indent=1)
        stprint(key='Features: ', data=self.features, indent=1)
        stprint(key='Predicate Groups: ', data=self.predicate_groups, indent=1)
        stprint(key='External Functions: ', data=self.external_functions, indent=1)
        stprint(key='External Function Implementations: ', data=self.external_functions_implementation, indent=1)
        stprint(key='Generators: ', data=self.generators, indent=1)
        print('  Operators:')
        if len(self.operators) > 0:
            for op in self.operators.values():
                print(indent_text(op.pddl_str(), level=2))
        else:
            print('    <Empty>')
        print('  Axioms:')
        if len(self.axioms) > 0:
            for op in self.axioms.values():
                print(indent_text(op.pddl_str(), level=2))
        else:
            print('    <Empty>')

    def post_init(self):
        self.analyze_static_predicates()

    def analyze_static_predicates(self):
        dynamic = set()
        for op in itertools.chain(self.operators.values(), self.axioms.values()):
            for eff in op.effects:
                if isinstance(eff.assign_expr, (AssignOp, ConditionalAssignOp)):
                    dynamic.add(eff.assign_expr.feature.feature_def.name)
                else:
                    expr = eff.unwrapped_assign_expr
                    assert isinstance(expr, (AssignOp, ConditionalAssignOp))
                    dynamic.add(expr.feature.feature_def.name)
        for p in self.features.values():
            if self.feature_in_group(p, ['basic', 'augmented-input', 'augmented']):
                if p.name not in dynamic:
                    p.mark_static()
            else:
                if p.cacheable and p.expr is not None:
                    used_features, _ = get_used_features_and_functions(self, p.expr)
                    static = True
                    for f in used_features:
                        if not self.features[f].static:
                            static = False
                            break
                    if static:
                        p.mark_static()


class Problem(object):
    def __init__(self):
        self.objects: Dict[str, str] = dict()
        self.predicates: List[FeatureApplication] = list()
        self.goal: Optional[ValueOutputExpression] = None

    def add_object(self, name, typename):
        self.objects[name] = typename

    def add_predicate(self, predicate):
        self.predicates.append(predicate)

    def set_goal(self, goal):
        self.goal = goal

    def to_state(self, domain) -> State:
        object_names = list(self.objects.keys())
        object_types = [domain.types[self.objects[name]] for name in object_names]
        state = State(object_types, ValueDict(), object_names)

        ctx = state.define_context(domain)
        predicates = list()
        for p in self.predicates:
            predicates.append(ctx.get_pred(p.feature_def.name)(*[arg.name for arg in p.arguments]))
        ctx.define_predicates(predicates)

        return state


def get_used_features_and_functions(domain: Domain, init_expr: Expression):
    used_features = set()
    used_functions = set()

    def dfs(expr):
        nonlocal used_features
        nonlocal used_functions

        if isinstance(expr, (tuple, list)):
            for e in expr:
                dfs(e)
        else:
            assert isinstance(expr, Expression)
            if isinstance(expr, (VariableExpression, ObjectConstantExpression, ConstantExpression)):
                pass
            elif isinstance(expr, FeatureApplication):
                feature_def: FeatureDef = expr.feature_def
                if not feature_def.cacheable:
                    for e in expr.arguments:
                        dfs(e)
                    used_functions.add(feature_def.name)
                if domain.feature_in_group(feature_def, ['basic', 'augmented-input']):
                    pass
                elif domain.feature_in_group(feature_def, ['augmented']):
                    used_features.add(feature_def.name)
                else:
                    assert feature_def.expr is not None
                    dfs(feature_def.expr)
            elif isinstance(expr, FunctionApplication):
                used_functions.add(expr.function_def.name)
                for e in expr.arguments:
                    dfs(e)
            elif isinstance(expr, BoolOp):
                for e in expr.arguments:
                    dfs(e)
            elif isinstance(expr, QuantificationOp):
                dfs(expr.expr)
            elif isinstance(expr, FeatureEqualOp):
                output_type = expr.feature.output_type
                if isinstance(output_type, NamedValueType):
                    used_functions.add(f'type::{output_type.typename}::equal')
                dfs(expr.feature)
                dfs(expr.value)
            elif isinstance(expr, AssignOp):
                pass
            elif isinstance(expr, ConditionalAssignOp):
                dfs(expr.condition)
            elif isinstance(expr, DeicticAssignOp):
                dfs(expr.expr)
            else:
                raise TypeError('Unknown expression type: {}.'.format(type(expr)))

    dfs(init_expr)
    return used_features, used_functions


def get_used_functions(domain: Domain, init_expr: Expression):
    used_functions = set()

    def dfs(expr):
        nonlocal used_functions
        if isinstance(expr, (tuple, list)):
            for e in expr:
                dfs(e)
        else:
            assert isinstance(expr, Expression)
            if isinstance(expr, (VariableExpression, ObjectConstantExpression, ConstantExpression)):
                pass
            elif isinstance(expr, FeatureApplication):
                feature_def: FeatureDef = expr.feature_def
                if not feature_def.cacheable:
                    for e in expr.arguments:
                        dfs(e)
                    used_functions.add(feature_def.name)
                if domain.feature_in_group(feature_def, ['basic', 'augmented', 'augmented-input']):
                    pass
                else:
                    assert feature_def.expr is not None
                    dfs(feature_def.expr)
            elif isinstance(expr, FunctionApplication):
                used_functions.add(expr.function_def.name)
                for e in expr.arguments:
                    dfs(e)
            elif isinstance(expr, BoolOp):
                for e in expr.arguments:
                    dfs(e)
            elif isinstance(expr, QuantificationOp):
                dfs(expr.expr)
            elif isinstance(expr, FeatureEqualOp):
                output_type = expr.feature.output_type
                if isinstance(output_type, NamedValueType):
                    used_functions.add(f'type::{output_type.typename}::equal')
                dfs(expr.feature)
                dfs(expr.value)
            elif isinstance(expr, AssignOp):
                pass
            elif isinstance(expr, ConditionalAssignOp):
                dfs(expr.condition)
            elif isinstance(expr, DeicticAssignOp):
                dfs(expr.expr)
            else:
                raise TypeError('Unknown expression type: {}.'.format(type(expr)))

    dfs(init_expr)
    return used_functions


def get_feature_and_function_dependencies(domain):
    edges = list()
    for predicate_name in domain.predicate_groups['@goal']:
        predicate_def = domain.features[predicate_name]
        u = get_used_features_and_functions(domain, predicate_def.expr)
        edges.append(dict(
            goal=predicate_def.name,
            features=u[0], functions=u[1]
        ))
    for op in domain.operators.values():
        effects = op.effects
        for i, eff in enumerate(effects):
            target_feature = eff.unwrapped_assign_expr.feature.function_def
            u = get_used_features_and_functions(domain, eff.assign_expr.value)
            v = get_used_features_and_functions(domain, eff.assign_expr.feature)
            u = (u[0] | v[0], u[1] | v[1])  # take the union of the used features and used functions.
            edges.append(dict(
                action=op.name,
                effect_index=i,
                effect=target_feature.name,
                features=u[0], functions=u[1]
            ))
    return edges


class AugmentedFeatureStage(object):
    def __init__(self, feature_name, feature_functions, all_functions, supervisions):
        self.feature_name = feature_name
        self.feature_functions = feature_functions
        self.all_functions = all_functions
        self.supervisions = supervisions

    def __str__(self):
        fmt = 'Stage(\n'
        fmt += f'  feature_names: {self.feature_name}\n'
        fmt += f'  feature_functions: {self.feature_functions}\n'
        fmt += f'  all_functions: {self.all_functions}\n'
        fmt += f'  supervisions: {indent_text(stformat(self.supervisions), indent_format="  ").strip()}\n'
        fmt += ')'
        return fmt

    __repr__ = jacinle.repr_from_str


def build_augmented_feature_stages(domain):
    dependencies = get_feature_and_function_dependencies(domain)
    graph = nx.DiGraph()
    for dep in dependencies:
        goal = dep['goal'] if 'goal' in dep else dep['effect']
        for feat in dep['features']:
            graph.add_edge(feat, goal)
        for f1, f2 in itertools.product(dep['features'], dep['features']):
            graph.add_edge(f1, f2)
    scc = nx.strongly_connected_components(graph)

    stages = list()
    visited = set()
    current = {dep['goal'] for dep in dependencies if 'goal' in dep}
    visited.update(current)
    while len(graph.nodes) > len(visited):
        nodes, edges = set(), list()
        for dep in dependencies:
            goal = dep['goal'] if 'goal' in dep else dep['effect']
            if goal in current:
                nodes.update(dep['features'])
        for component in scc:
            if len(set.intersection(component, nodes)):
                nodes.update(component)
        for dep in dependencies:
            if len(set.intersection(nodes, dep['features'])):
                edges.append(dep)

        feature_functions = set()
        for feature_name in nodes:
            feature_functions.update(get_used_functions(domain, domain.features[feature_name].expr))
        all_functions = feature_functions.copy()
        for dep in edges:
            all_functions.update(dep['functions'])
        stages.append(AugmentedFeatureStage(nodes, feature_functions, all_functions, edges))
        current = nodes
        visited.update(nodes)
    return stages


def extract_generator_data(domain: Domain, state: State, generator_def: GeneratorDef):
    ctx, result = domain.forward_expr(state, generator_def.arguments, generator_def.certifies, return_ctx=True)
    result.tensor = torch.ge(result.tensor, 0.5)
    if result.tensor_mask is not None:
        result.tensor = torch.logical_and(result.tensor, torch.ge(result.tensor_mask, 0.5))

    def _index(value, mask):
        value = value.expand_as(mask)
        return value.tensor[mask.tensor]

    with ctx.as_default():
        contexts = [_index(c.forward(ctx), result) for c in generator_def.context]
        generates = [_index(c.forward(ctx), result) for c in generator_def.generates]
    return contexts, generates


class ValueQuantizer(object):
    def __init__(self, domain: Domain, register: Optional[bool] = True):
        self.domain = domain
        self.values: Dict[str, Union[List[Any], Dict[Any, int]]] = dict()
        if register:
            self.domain.set_value_quantizer(self)

    def quantize(self, typename: str, value: Union[torch.Tensor, Value]) -> int:
        if not isinstance(value, Value):
            value = Value(self.domain.types[typename], [], value)
        use_hash = self.domain.has_external_function(f'type::{typename}::hash')
        if typename not in self.values:
            self.values[typename] = dict() if use_hash else list()

        if use_hash:
            hash_value = self.domain.get_external_function(f'type::{typename}::hash')(value)
            if hash_value not in self.values[typename]:
                self.values[typename][hash_value] = len(self.values[typename])
            return self.values[typename][hash_value]
        else:
            for i, v in enumerate(self.values[typename]):
                if bool(self.domain.get_external_function(f'type::{typename}::equal')(v, value)):
                    return i
            self.values[typename].append(value)
            return len(self.values[typename]) - 1

    def quantize_tensor(self, dtype: NamedValueType, tensor: torch.Tensor) -> torch.Tensor:
        tensor_flatten = tensor.reshape((-1,) + dtype.size_tuple())
        quantized_values = list()
        for value in tensor_flatten:
            quantized_values.append(self.quantize(dtype.typename, value))
        quantized_tensor = torch.tensor(quantized_values, dtype=torch.int64, device=tensor_flatten.device)
        quantized_tensor = quantized_tensor.reshape(tensor.shape[:-dtype.ndim()])
        return quantized_tensor

    def quantize_dict_list(self, continuous_values: Mapping[str, Sequence[int]]) -> Mapping[str, Sequence[Value]]:
        output_dict = dict()
        for typename, values in continuous_values.items():
            output_dict[typename] = set()
            for v in values:
                output_dict[typename].add(self.quantize(typename, v))
            output_dict[typename] = [Value(self.domain.types[typename], [], x, quantized=True) for x in output_dict[typename]]
        return output_dict

    def quantize_value(self, value: Value) -> Value:
        if value.quantized:
            return value
        if is_intrinsically_quantized(value.dtype):
            return value.make_quantized()
        if value.tensor_indices is not None:
            return value.make_quantized()
        assert isinstance(value.dtype, NamedValueType)
        return Value(value.dtype, value.batch_variables, self.quantize_tensor(value.dtype, value.tensor), value.batch_dims, quantized=True)

    def quantize_state(self, state: StateLike, includes=None, excludes=None) -> StateLike:
        state = state.clone()
        for feature_name in state.features.all_feature_names:
            if (includes is not None and feature_name not in includes) or (excludes is not None and feature_name in excludes):
                rv = state.features[feature_name]
            else:
                feature_def = self.domain.features[feature_name]
                if self.domain.feature_in_group(feature_def, ['augmented-input']):
                    rv = state.features[feature_name]
                else:
                    rv = self.quantize_value(state.features[feature_name])
            state.features[feature_name] = rv
        return state

    def unquantize(self, typename: str, value: int) -> Value:
        return self.values[typename][value]

    def unquantize_tensor(self, dtype: NamedValueType, tensor: torch.Tensor) -> torch.Tensor:
        assert dtype.typename in self.values
        lookup_table = self.values[dtype.typename]
        output = [lookup_table[x].tensor for x in tensor.flatten().tolist()]
        output = torch.stack(output, dim=0).reshape(tensor.shape + dtype.size_tuple())
        return output

    def unquantize_value(self, value: Value) -> Value:
        dtype = value.dtype
        assert isinstance(dtype, NamedValueType)
        if is_intrinsically_quantized(dtype):
            return Value(dtype, value.batch_variables, value.tensor, value.batch_dims, quantized=False)
        else:
            return Value(dtype, value.batch_variables, self.unquantize_tensor(dtype, value.tensor), value.batch_dims, quantized=False)

    def unquantize_optimistic_context(self, ctx: OptimisticValueContext):
        def _cvt(arg):
            if isinstance(arg, DeterminedValue):
                if not arg.quantized:
                    return arg
                elif is_intrinsically_quantized(arg.dtype):
                    if arg.dtype == BOOL:
                        return DeterminedValue(BOOL, bool(arg.value), True)
                    return DeterminedValue(arg.dtype, int(arg.value), True)
                else:
                    assert isinstance(arg.dtype, NamedValueType) and isinstance(arg.value, int)
                    return DeterminedValue(arg.dtype, self.unquantize(arg.dtype.typename, arg.value), False)
            else:
                return arg

        ctx = ctx.clone()
        for i, c in enumerate(ctx.constraints):
            new_args = tuple(map(_cvt, c.args))
            new_rv = _cvt(c.rv)
            ctx.constraints[i] = OptimisticConstraint(c.func_def, new_args, new_rv)
        return ctx
