"""
Generate And-Or discretization of custom types and functions in a domain.
"""

import numpy as np
import collections
from typing import Optional, Union, Iterable, Tuple, Sequence, List, Dict

import torch
import jacinle
import jactorch
from sklearn.tree import DecisionTreeClassifier
from jaclearn.logic.decision_tree.rule import extract_rule, DecisionRule, AtomicDecisionRule

from .value import BOOL, Value, is_intrinsically_quantized, QINDEX
from .state import StateLike, State
from .domain import Domain, OperatorApplier
from .value import Variable
from .expr import VariableExpression, ValueOutputExpression, FeatureApplication, ExternalFunctionApplication, FunctionApplication, flatten_expression, is_external_function_application, is_constant_bool_expr
from .expr import QuantificationOp, QuantifierType, FeatureEqualOp
from .expr import VariableAssignmentExpression, DeicticAssignOp, ConditionalAssignOp, AssignOp, DeicticSelectOp, ConditionalSelectOp
from .expr import is_not_expr, is_exists_expr, is_forall_expr, is_and_expr, is_or_expr
from .strips.strips_expr import StripsPredicateName, StripsExpression, StripsValueOutputExpression
from .strips.strips_expr import StripsSASPredicate, StripsSASExpression
from .strips.strips_expr import StripsBoolConstant, StripsBoolPredicate, StripsBoolAOFormula, StripsBoolNot, StripsBoolFEFormula
from .strips.strips_expr import StripsVariableAssignmentExpression, StripsDeicticAssignment, StripsConditionalAssignment, StripsAssignment

logger = jacinle.get_logger(__file__)

__all__ = [
    'AODiscretizationData', 'AODiscretizationDataset', 'AOFeatureCodebook', 'AOFeatureDiscretization', 'AOFunctionDiscretization', 'ao_discretize'
]


class AODiscretizationData(collections.namedtuple('AODiscretizationData', ['states', 'actions', 'dones', 'goal'])):
    states: Sequence[State]
    actions: Sequence[OperatorApplier]
    dones: torch.Tensor
    goal: ValueOutputExpression


class AODiscretizationDataset(object):
    def __init__(self):
        self.datasets: List[AODiscretizationData] = list()

    def feed(self, states, actions, dones, goal):
        self.datasets.append(AODiscretizationData(states, actions, dones, goal))

    def iter_all_states(self):
        for data in self.datasets:
            yield from data.states

    def iter_all_transitions(self):
        for data in self.datasets:
            for i in range(len(data.actions)):
                yield data.states[i], data.actions[i], data.states[i + 1]


class AOFeatureCodebook(object):
    def __init__(self, codebook: np.ndarray):
        self.codebook = codebook

    @property
    def size(self) -> int:
        return self.codebook.shape[0]

    @property
    def dim(self) -> int:
        return self.codebook.shape[1]

    def quantize(self, value: Value) -> Value:
        if value.quantized:
            return value
        value_tensor = value.tensor
        shape = value_tensor.shape[:value.total_batch_dims]
        value_tensor = value_tensor.reshape((-1, ) + value_tensor.shape[value.total_batch_dims:])
        code = _vq(value_tensor, self.codebook).reshape(shape)
        return Value(value.dtype, value.batch_variables, code, quantized=True)


class AOFeatureDiscretization(object):
    def __init__(self, domain: Domain, feature_dimensions: Dict[str, int]):
        self.domain = domain
        self.features = dict()
        self.feature_dimensions = feature_dimensions
        self.feature_codebooks: Dict[str, AOFeatureCodebook] = dict()

        self.dataset: Optional[AODiscretizationDataset] = None

    def feed(self, state_like: StateLike):
        self.domain.forward_augmented_features(state_like)
        for feature_name, feature in state_like.features.items():
            feature_def = self.domain.features[feature_name]
            if feature_def.group in ('basic', 'augmented') and not is_intrinsically_quantized(feature_def.output_type):
                feature = feature.tensor.reshape((-1, ) + feature.tensor.shape[feature.total_batch_dims:])
                if feature_name not in self.features:
                    self.features[feature_name] = list()
                self.features[feature_name].append(feature)

    def discretize(self, dataset: AODiscretizationDataset):
        self.features = dict()
        self.feature_codebooks = dict()
        self.dataset = dataset

        for state in self.dataset.iter_all_states():
            self.feed(state)
        self.run_discretization()

    def run_discretization(self):
        for feature_name, feature_list in self.features.items():
            feature_dim = self.feature_dimensions[feature_name]
            all_features = torch.cat(feature_list, dim=0)
            codebook = _kmeans(all_features, feature_dim)
            codebook = AOFeatureCodebook(codebook)
            self.feature_codebooks[feature_name] = codebook
            self.domain.features[feature_name].ao_discretization = codebook

    def set_quantization(self, feature_name, codebook: np.ndarray):
        codebook = AOFeatureCodebook(codebook)
        self.feature_codebooks[feature_name] = codebook
        self.domain.features[feature_name].ao_discretization = codebook

    def quantize(self, feature_name, value: Value):
        if value.quantized:
            return value
        if is_intrinsically_quantized(value.dtype):
            return Value(value.dtype, value.batch_variables, value.tensor, batch_dims=value.batch_dims, quantized=value.quantized)

        codebook = self.feature_codebooks[feature_name]
        return codebook.quantize(value)

    def __str__(self):
        codebook_str = '\n'.join(f'  {feature_name}: {codebook.shape}' for feature_name, codebook in self.feature_codebooks.items())
        return f'AOFeatureDiscretization(\n{codebook_str}\n)'

    _repr__ = __str__


class AOFunctionDiscretization(object):
    def __init__(self, domain: Domain, feature_discretization: AOFeatureDiscretization, cache_bool_features: bool = True, verbose: bool = False):
        self.domain = domain
        self.feature_discretization = feature_discretization
        self.functions = dict()
        self.function_output_codebooks = dict()
        self.cache_bool_features = cache_bool_features
        self.verbose = verbose
        self.dataset: Optional[AODiscretizationDataset] = None

    def discretize(self, dataset: AODiscretizationDataset):
        self.dataset = dataset
        self.run_discretization()

    def run_discretization(self):
        self.run_discretization_features()
        self.run_discretization_actions()

    def run_discretization_features(self):
        for name, feature_def in self.domain.features.items():
            if feature_def.expr is not None and feature_def.cacheable and feature_def.output_type == BOOL and feature_def.group != 'basic':
                # Compile the function into a deterministic program.
                expr = flatten_expression(feature_def.expr, flatten_cacheable_bool=not self.cache_bool_features)
                feature_def.ao_discretization = self.compose_value_expression(expr, feature_def.arguments)

                if self.verbose:
                    print(feature_def)
                    print(feature_def.ao_discretization)
                    print('-' * 80)
                    input('Press enter to continue...')

    def run_discretization_actions(self):
        for operator in self.domain.operators.values():
            if self.verbose:
                print('Operator::' + str(operator))
            for pred in operator.preconditions:
                bool_expr = flatten_expression(pred.bool_expr, flatten_cacheable_bool=not self.cache_bool_features)
                pred.ao_discretization = self.compose_value_expression(bool_expr, operator.arguments)
                if self.verbose:
                    print('Precondition::' + str(pred))
                    print(pred.ao_discretization)
            for effect in operator.effects:
                assign_expr = flatten_expression(effect.assign_expr, flatten_cacheable_bool=not self.cache_bool_features)
                applicable_states = list()
                variable2index_lists = list()

                for state, action, _ in self.dataset.iter_all_transitions():
                    if action.name == operator.name:
                        applicable_states.append(state)
                        variable2index = {variable: state.get_typed_index(obj_name) for variable, obj_name in zip(action.operator.arguments, action.arguments)}
                        variable2index_lists.append(variable2index)

                if len(applicable_states) == 0:
                    logger.warning(f'No applicable states for {operator.name}.')
                else:
                    effect.ao_discretization = self.compose_assignment_expr(assign_expr, operator.arguments, applicable_states, variable2index_lists)
                    print('Effect::' + str(effect))
                    print(effect.ao_discretization)
            if self.verbose:
                input('Press enter to continue...')

    def compose_assignment_expr(
        self,
        expr: VariableAssignmentExpression,
        broadcast_variables: Sequence[Variable],
        states: Sequence[State],
        variable2index_list: Sequence[Dict[Variable, int]]
    ) -> StripsVariableAssignmentExpression:
        if isinstance(expr, DeicticAssignOp):
            new_states = list()
            new_list = list()
            for state, variable2index in zip(states, variable2index_list):
                for index in range(state.get_nr_objects_by_type(expr.variable.typename)):
                    variable2index = variable2index.copy()
                    variable2index[expr.variable] = index
                    new_states.append(state)
                    new_list.append(variable2index)
            return StripsDeicticAssignment(
                expr.variable,
                self.compose_assignment_expr(expr.expr, broadcast_variables, new_states, new_list)
            )
        elif isinstance(expr, ConditionalAssignOp):
            condition = self.compose_value_expression(expr.condition, broadcast_variables, False, states=states, variable2index_list=variable2index_list)
            assign_op = self.compose_assignment_expr(AssignOp(expr.feature, expr.value), broadcast_variables, states, variable2index_list)
            return StripsConditionalAssignment(assign_op, condition)
        elif isinstance(expr, AssignOp):
            feature_def = expr.feature.feature_def
            feature_name = feature_def.name
            arguments = [arg.variable for arg in expr.feature.arguments]

            if feature_def.output_type == BOOL:
                assert is_constant_bool_expr(expr.value)
                return StripsAssignment(StripsBoolPredicate(feature_name, False, arguments), expr.value.value.item())

            # print('Composing assignment expression', expr.value)
            target_expr = self.compose_value_expression(expr.value, broadcast_variables=broadcast_variables, states=states, variable2index_list=variable2index_list, target_feature_name=feature_name)
            # print(target_expr)
            return StripsAssignment(StripsSASPredicate(feature_name, None, False, arguments), target_expr)

    def compose_value_expression(
        self,
        expr: ValueOutputExpression,
        broadcast_variables: Optional[Sequence[Variable]] = None,
        is_negated: bool = False,
        states: Optional[Sequence[State]] = None,
        variable2index_list: Optional[Sequence[Dict[Variable, int]]] = None,
        target_feature_name: Optional[str] = None,
    ) -> StripsExpression:
        if isinstance(expr, FeatureApplication) and expr.feature_def.cacheable and expr.feature_def.output_type == BOOL:
            arguments = list()
            for arg in expr.arguments:
                assert isinstance(arg, VariableExpression)
                arguments.append(arg.variable)
            assert expr.feature_def.output_type == BOOL
            return StripsBoolPredicate(expr.feature_def.name, is_negated, arguments)
        elif is_not_expr(expr):
            return self.compose_value_expression(expr.arguments[0], broadcast_variables, not is_negated, states, variable2index_list, target_feature_name)
        elif isinstance(expr, QuantificationOp):
            if broadcast_variables is not None:
                broadcast_variables = broadcast_variables + (expr.variable, )
            if variable2index_list is not None:
                # TODO: Make a copy?
                for mapping in variable2index_list:
                    mapping[expr.variable] = QINDEX
            strips_expr = self.compose_value_expression(expr.expr, broadcast_variables, is_negated, states, variable2index_list, target_feature_name)
            return StripsBoolFEFormula(expr.variable, strips_expr, is_disjunction=(
                expr.quantifier_type == QuantifierType.EXISTS and not is_negated or
                expr.quantifier_type == QuantifierType.FORALL and is_negated
            ))
        elif is_and_expr(expr) or is_or_expr(expr):
            classifiers = [
                self.compose_value_expression(arg, broadcast_variables, is_negated, states, variable2index_list, target_feature_name)
                for arg in expr.arguments
            ]
            return StripsBoolAOFormula(classifiers, is_disjunction=(
                is_or_expr(expr) and not is_negated or
                is_and_expr(expr) and is_negated
            ))
        elif isinstance(expr, FeatureEqualOp):
            return self.compose_external_function(expr, broadcast_variables, is_negated=is_negated, states=states, variable2index_list=variable2index_list)
        elif isinstance(expr, FunctionApplication):
            return self.compose_external_function(expr, broadcast_variables=broadcast_variables, is_negated=is_negated, states=states, variable2index_list=variable2index_list, target_feature_name=target_feature_name)
        else:
            raise TypeError('Unsupported expression type: {}.'.format(expr))

    def compose_external_function(
        self,
        expr: Union[FeatureApplication, ExternalFunctionApplication, FeatureEqualOp],
        broadcast_variables: Optional[Sequence[Variable]] = None,
        is_negated: bool = False,
        states: Optional[Sequence[State]] = None,
        variable2index_list: Optional[Sequence[Dict[Variable, int]]] = None,
        target_feature_name: Optional[str] = None,
    ) -> StripsValueOutputExpression:
        rvs = list()
        if states is None:
            for i, state in enumerate(self.dataset.iter_all_states()):
                rv = self.domain.forward_expr(state, broadcast_variables, expr)
                rvs.append(rv)
        else:
            for state, variable2index in zip(states, variable2index_list):
                rv = self.domain.forward_expr(state, variable2index, expr)
                rvs.append(rv)

        for i, rv in enumerate(rvs):
            broadcast_variables = [bv for bv in broadcast_variables if bv.name in rv.batch_variables]
            if rv.dtype == BOOL:
                tensor = rv.tensor > 0.5
            else:
                assert target_feature_name is not None
                rv = self.feature_discretization.quantize(target_feature_name, rv)
                tensor = rv.tensor
            tensor = tensor.reshape(-1)
            rvs[i] = tensor

        used_features = list(self._get_external_function_X(expr, broadcast_variables=broadcast_variables, states=states, variable2index_list=variable2index_list))
        rvs = torch.cat(rvs, dim=0)

        if len(used_features) == 0:
            if expr.output_type == BOOL:
                return StripsBoolConstant(bool(rvs[0]))
            else:
                return StripsSASExpression({rvs[0]: StripsBoolConstant(True)})

        rule = _extract_rule(
            jactorch.as_numpy(torch.cat([feat for feat, _ in used_features], dim=-1)),
            jactorch.as_numpy(rvs),
            sum([names for _, names in used_features], start=list())
        )
        # rule = _extract_rule_sas(used_features, jactorch.as_numpy(rvs))

        if expr.output_type == BOOL:
            if (not is_negated) in rule:
                return rule[not is_negated]
            return StripsBoolConstant(False)
        else:
            assert not is_negated
            return StripsSASExpression(rule)

    def _get_external_function_X(
        self,
        expr: Union[FeatureApplication, ExternalFunctionApplication, FeatureEqualOp],
        broadcast_variables: Optional[Sequence[Variable]] = None,
        states: Optional[Sequence[State]] = None,
        variable2index_list: Optional[Sequence[Dict[Variable, int]]] = None,
        flatten: bool = True,
    ) -> Iterable[Tuple[torch.Tensor, List[StripsSASPredicate]]]:
        if is_external_function_application(expr):
            for arg in expr.arguments:
                assert isinstance(arg, (FeatureApplication, ExternalFunctionApplication, DeicticSelectOp, ConditionalSelectOp))
                yield from self._get_external_function_X(arg, broadcast_variables=broadcast_variables, states=states, variable2index_list=variable2index_list)
        elif isinstance(expr, FeatureEqualOp):
            yield from self._get_external_function_X(expr.feature, broadcast_variables=broadcast_variables, states=states, variable2index_list=variable2index_list)
            yield from self._get_external_function_X(expr.value, broadcast_variables=broadcast_variables, states=states, variable2index_list=variable2index_list)
        elif isinstance(expr, DeicticSelectOp):
            subexpr = expr.expr
            assert isinstance(subexpr, ConditionalSelectOp)
            if broadcast_variables is not None:
                broadcast_variables = list(broadcast_variables)
                broadcast_variables.append(expr.variable)
            if variable2index_list is not None:
                for variable2index in variable2index_list:
                    variable2index[expr.variable] = QINDEX
            condition_expr = self.compose_value_expression(subexpr.condition, broadcast_variables=broadcast_variables, states=states, variable2index_list=variable2index_list)

            condition_values = list()
            for state, variable2index in zip(states, variable2index_list):
                rv = self.domain.forward_expr(state, variable2index, subexpr.condition)
                condition_values.append(rv)
            for i, rv in enumerate(condition_values):
                assert rv.dtype == BOOL
                tensor = rv.tensor > 0.5
                tensor = tensor.reshape(-1)
                condition_values[i] = tensor
            value_predicate_pairs = list(self._get_external_function_X(subexpr.feature, broadcast_variables=broadcast_variables, states=states, variable2index_list=variable2index_list, flatten=False))

            for pair_index, (tensors, predicates) in enumerate(value_predicate_pairs):
                for i, tensor in enumerate(tensors):
                    assert tensor.dim() == 2
                    cv = condition_values[i]
                    assert cv.dim() == 1
                    tensor = (tensor * cv.unsqueeze(-1)).amax(dim=0)
                    tensors[i] = tensor
                tensors = torch.stack(tensors, dim=0)
                value_predicate_pairs[pair_index] = (tensors, predicates)
                for i, predicate in enumerate(predicates):
                    predicates[i] = StripsBoolFEFormula(expr.variable, StripsBoolAOFormula([condition_expr, predicate], is_disjunction=False), is_disjunction=True)

            yield from value_predicate_pairs

            if broadcast_variables is not None:
                broadcast_variables.pop()
            if variable2index_list is not None:
                for variable2index in variable2index_list:
                    del variable2index[expr.variable]
        elif isinstance(expr, FeatureApplication):
            feature_def = expr.feature_def
            feature_name = feature_def.name

            arguments = list()
            for arg in expr.arguments:
                assert isinstance(arg, VariableExpression)
                arguments.append(arg.variable)
            argument_names = [arg.name for arg in arguments]

            results = list()
            if states is None:
                states = self.dataset.iter_all_states()
                for state in states:
                    batch_variables = [variable.name for variable in broadcast_variables]
                    batch_sizes = [state.get_nr_objects_by_type(variable.typename) for variable in broadcast_variables]

                    feature = state.features[feature_name].rename_batch_variables(argument_names, clone=True)
                    feature = self.feature_discretization.quantize(feature_name, feature)
                    feature = feature.expand(batch_variables, batch_sizes)
                    tensor = feature.tensor

                    if flatten:
                        tensor = tensor.reshape(-1)
                    tensor = jactorch.one_hot_nd(tensor.long(), self.feature_discretization.feature_dimensions[feature_name])
                    results.append(tensor)
            else:
                for state, variable2index in zip(states, variable2index_list):
                    batch_variables = [variable.name for variable in broadcast_variables if variable2index[variable] == QINDEX]
                    batch_sizes = [state.get_nr_objects_by_type(variable.typename) for variable in broadcast_variables if variable2index[variable] == QINDEX]

                    argument_values = tuple([variable2index[var] for var in arguments])
                    feature = state.features[feature_name][argument_values].rename_batch_variables([arg.name for arg in arguments if variable2index[arg] == QINDEX], clone=True)
                    feature = self.feature_discretization.quantize(feature_name, feature)
                    feature = feature.expand(batch_variables, batch_sizes)
                    tensor = feature.tensor

                    if flatten:
                        tensor = tensor.reshape(-1)
                    tensor = jactorch.one_hot_nd(tensor.long(), self.feature_discretization.feature_dimensions[feature_name])
                    results.append(tensor)

            yield torch.cat(results, dim=0) if flatten else results, [
                StripsSASPredicate(feature_name, i, False, arguments)
                for i in range(self.feature_discretization.feature_dimensions[feature_name])
            ]
        else:
            raise NotImplementedError('Unsupported expression type: {}.'.format(expr))


def ao_discretize(domain: Domain, dataset: AODiscretizationDataset, feature_dims: Dict[str, int], cache_bool_features: bool = True, manual_feature_discretizations: Optional[Dict[str, np.ndarray]] = None):
    pd = AOFeatureDiscretization(domain, feature_dims)
    pd.discretize(dataset)
    if manual_feature_discretizations is not None:
        for k, v in manual_feature_discretizations.items():
            pd.set_quantization(k, v)
    fd = AOFunctionDiscretization(domain, pd, cache_bool_features=cache_bool_features)
    fd.discretize(dataset)
    return pd, fd


def _kmeans(x, k, seed=None):
    from scipy.cluster.vq import kmeans, whiten
    x = jactorch.as_numpy(x)
    x = whiten(x)
    codebook, _ = kmeans(x, min(k, x.shape[0]), iter=20, thresh=1e-05, check_finite=True)
    return codebook


def _vq(x, codebook):
    from scipy.cluster.vq import vq
    x = jactorch.as_numpy(x)
    code, _ = vq(x, codebook)
    return torch.from_numpy(code)


def _extract_rule_sas(used_features, Y):
    mappings = dict()
    for i in range(Y.shape[0]):
        arguments = tuple([int(f[i].item()) for f, _ in used_features])
        rv = int(Y[i].item())
        if rv not in mappings:
            mappings[rv] = set()
        mappings[rv].add(arguments)
    mappings = {rv: [[used_features[i][1][j] for i, j in enumerate(clause)] for clause in rules] for rv, rules in mappings.items()}

    mappings = {
        rv: StripsBoolAOFormula([StripsBoolAOFormula(clause, is_disjunction=False) for clause in clauses], is_disjunction=True)
        for rv, clauses in mappings.items()
    }
    return mappings


def _extract_rule(X, Y, all_feature_names):
    counter = collections.Counter(Y)
    weights = {k: 1 / v for k, v in counter.items()}
    clf = DecisionTreeClassifier(class_weight=weights)
    clf.fit(X, Y)

    if Y.dtype == 'bool':
        rules = extract_rule(clf, all_feature_names, boolean_input=True, multi_output=False)
    else:
        rules = extract_rule(clf, all_feature_names, boolean_input=True, multi_output=True)

    for key, rule in rules.items():
        for i, clause in enumerate(rule.clauses):
            used_features = set()
            for literal in clause:
                var: StripsSASPredicate = literal.variable
                if isinstance(var, StripsSASPredicate) and literal.right_branch:
                    used_features.add((var.sas_name, tuple(var.arguments)))
            output_clause = list()
            for literal in clause:
                var = literal.variable
                if isinstance(var, StripsSASPredicate) and not literal.right_branch and (var.sas_name, tuple(var.arguments)) in used_features:
                    pass
                else:
                    output_clause.append(literal)
            rule.clauses[i] = output_clause

    rules = {k: _rule2expr(v) for k, v in rules.items()}
    return rules


def _rule2expr(rule: DecisionRule):
    clauses = rule.clauses
    if len(clauses) == 0:
        return StripsBoolConstant(True)

    if len(clauses) == 1:
        clause = clauses[0]
        if len(clause) == 0:
            return StripsBoolConstant(True)
        return StripsBoolAOFormula([_rule2expr_atomic(lit) for lit in clause], is_disjunction=False) if len(clause) > 1 else _rule2expr_atomic(clause[0])

    return StripsBoolAOFormula(
        [
            StripsBoolAOFormula([_rule2expr_atomic(lit) for lit in clause], is_disjunction=False) if len(clause) > 1 else _rule2expr_atomic(clause[0])
            for clause in clauses
        ], is_disjunction=True
    )


def _rule2expr_atomic(rule: AtomicDecisionRule):
    var = rule.variable
    if isinstance(var, StripsSASPredicate):
        return StripsSASPredicate(var.sas_name, var.sas_index, not rule.right_branch, var.arguments)
    else:
        if rule.right_branch:
            return var
        else:
            return StripsBoolNot(var)
