import warnings
import collections
import contextlib
from abc import ABC
from typing import Optional, Union, Callable, Iterable, Sequence, Tuple, List, Set, FrozenSet, Dict

import jacinle
from .strips_expr import StripsPredicateName, StripsProposition, StripsState

__all__ = [
    'GSOptimisticStaticObjectType', 'GS_OPTIMISTIC_STATIC_OBJECT',
    'GStripsClassifierForwardDiffReturn', 'GStripsExpression',
    'GStripsClassifier', 'GStripsBoolConstant', 'GStripsSimpleClassifier', 'GStripsComplexClassifier',
    'gs_is_constant_true', 'gs_is_constant_false', 'gs_is_simple_empty_classifier', 'gs_is_simple_conjunctive_classifier',
    'GStripsAssignmentExpression', 'GStripsSimpleAssignment', 'GStripsConditionalAssignment', 'GStripsSASAssignment', 'GStripsDerivedPredicate'
]


class GSOptimisticStaticObjectType(object):
    """OptimisticObject only occurs when the arguments to an operator is a complex-typed value."""
    pass


GS_OPTIMISTIC_STATIC_OBJECT = GSOptimisticStaticObjectType()


class GStripsClassifierForwardDiffReturn(collections.namedtuple('_StripsClassifierForwardDiffReturn', ['rv', 'propositions'])):
    pass


class GStripsExpression(ABC):
    def compile(self) -> Callable[[StripsState], Optional[Union[bool, GStripsClassifierForwardDiffReturn]]]:
        raise NotImplementedError()

    def iter_propositions(self) -> Iterable[StripsProposition]:
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    __repr__ = jacinle.repr_from_str


class GStripsClassifier(GStripsExpression):
    FORWARD_DIFF = False

    @staticmethod
    def set_forward_diff(value: bool = True):
        GStripsClassifier.FORWARD_DIFF = value

    @staticmethod
    @contextlib.contextmanager
    def enable_forward_diff_ctx():
        old_value = GStripsClassifier.FORWARD_DIFF
        GStripsClassifier.FORWARD_DIFF = True
        yield
        GStripsClassifier.FORWARD_DIFF = old_value

    def iter_propositions(self) -> Iterable[StripsProposition]:
        raise NotImplementedError()

    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsBoolConstant':
        raise NotImplementedError()


class GStripsBoolConstant(GStripsClassifier):
    def __init__(self, value: bool):
        super().__init__()
        self.value = value

    def compile(self):
        def function(state: StripsState):
            if GStripsClassifier.FORWARD_DIFF:
                return GStripsClassifierForwardDiffReturn(self.value, set())
            return self.value
        return function

    def iter_propositions(self) -> Iterable[StripsProposition]:
        return tuple()

    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsBoolConstant':
        return self

    def __str__(self):
        return '{}'.format(self.value).upper()


def gs_is_constant_true(classifier: GStripsClassifier) -> bool:
    return isinstance(classifier, GStripsBoolConstant) and classifier.value


def gs_is_constant_false(classifier: GStripsClassifier) -> bool:
    return isinstance(classifier, GStripsBoolConstant) and not classifier.value


class GStripsSimpleClassifier(GStripsClassifier):
    def __init__(self, classifier: Union[Set[str], FrozenSet[str]], is_disjunction: bool = False):
        super().__init__()
        self.classifier = frozenset(classifier)
        if len(self.classifier) > 1:
            self.is_disjunction = is_disjunction
        else:
            self.is_disjunction = False  # prefer to represent a single prop. as a conjunction.

    def compile(self):
        if self.is_disjunction:
            def function(state: StripsState, classifier=self.classifier) -> Optional[Union[bool, GStripsClassifierForwardDiffReturn]]:
                if GStripsClassifier.FORWARD_DIFF:
                    intersection = classifier.intersection(state)
                    if intersection:
                        return GStripsClassifierForwardDiffReturn(True, {next(iter(intersection))})
                    else:
                        return GStripsClassifierForwardDiffReturn(False, None)
                return len(classifier.intersection(state)) > 0
            return function
        else:
            def function(state: StripsState, classifier=self.classifier) -> Optional[Union[bool, GStripsClassifierForwardDiffReturn]]:
                if GStripsClassifier.FORWARD_DIFF:
                    return GStripsClassifierForwardDiffReturn(classifier <= state, classifier)
                return classifier <= state
            return function

    def iter_propositions(self) -> Iterable[StripsProposition]:
        yield from iter(self.classifier)

    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsClassifier':
        if state is None:
            state = set()

        diff = self.classifier - propositions
        new_classifiers = frozenset(self.classifier & propositions)

        if len(diff) == 0:
            return self
        else:
            if self.is_disjunction:
                if len(diff & state) > 0:
                    return GStripsBoolConstant(True)
                else:
                    if len(new_classifiers) == 0:
                        return GStripsBoolConstant(False)
                    else:
                        return GStripsSimpleClassifier(new_classifiers, self.is_disjunction)
            else:
                if len(diff & state) != len(diff):
                    return GStripsBoolConstant(False)
                else:
                    if len(new_classifiers) == 0:
                        return GStripsBoolConstant(True)
                    else:
                        return GStripsSimpleClassifier(new_classifiers, self.is_disjunction)

    def __str__(self):
        name = 'CONJ' if not self.is_disjunction else 'DISJ'
        classifier_str = ', '.join([str(x) for x in self.classifier])
        return f'{name}({classifier_str})'


def gs_is_simple_empty_classifier(classifier: GStripsClassifier) -> bool:
    return isinstance(classifier, GStripsSimpleClassifier) and len(classifier.classifier) == 0


def gs_is_simple_conjunctive_classifier(classifier: GStripsClassifier) -> bool:
    return isinstance(classifier, GStripsSimpleClassifier) and not classifier.is_disjunction


class GStripsComplexClassifier(GStripsClassifier):
    def __init__(self, expressions: Sequence[GStripsClassifier], is_disjunction: Optional[bool] = False):
        super().__init__()
        self.expressions = expressions
        self.is_disjunction = is_disjunction

    def compile(self) -> Callable[[StripsState], Optional[bool]]:
        compiled_functions = [e.compile() for e in self.expressions]
        if self.is_disjunction:
            def function(state: StripsState, functions=compiled_functions) -> Optional[Union[bool, GStripsClassifierForwardDiffReturn]]:
                if GStripsClassifier.FORWARD_DIFF:
                    for f in functions:
                        result, propositions = f(state)
                        if result:
                            return GStripsClassifierForwardDiffReturn(True, propositions)
                    return GStripsClassifierForwardDiffReturn(False, None)

                for f in functions:
                    if f(state):
                        return True
                return False
        else:
            def function(state: StripsState, functions=compiled_functions) -> Optional[Union[bool, GStripsClassifierForwardDiffReturn]]:
                if GStripsClassifier.FORWARD_DIFF:
                    all_propositions = list()
                    for f in functions:
                        result, propositions = f(state)
                        if not result:
                            return GStripsClassifierForwardDiffReturn(False, None)
                        all_propositions.append(propositions)
                    propositions = set()
                    for p in all_propositions:
                        propositions.update(p)
                    return GStripsClassifierForwardDiffReturn(True, propositions)

                for f in functions:
                    if not f(state):
                        return False
                return True
        return function

    def iter_propositions(self) -> Iterable[StripsProposition]:
        for e in self.expressions:
            yield from e.iter_propositions()

    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsClassifier':
        expressions = [e.filter_propositions(propositions, state=state) for e in self.expressions]
        return gstrips_compose_classifiers(expressions, self.is_disjunction, propagate_implicit_propositions=False)

    def __str__(self):
        name = 'ComplexAND' if not self.is_disjunction else 'ComplexOR'
        expressions = [str(x) for x in self.expressions]
        if sum([len(x) for x in expressions]) > 120:
            return f'{name}(\n' + ',\n'.join([jacinle.indent_text(jacinle.stformat(x)).rstrip() for x in expressions]) + '\n)'
        return f'{name}(' + ', '.join([str(x) for x in self.expressions]) + ')'


class GStripsAssignmentExpression(GStripsExpression, ABC):
    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsAssignmentExpression':
        raise NotImplementedError()

    def relax(self) -> 'GStripsAssignmentExpression':
        raise NotImplementedError()


class GStripsSimpleAssignment(GStripsAssignmentExpression):
    def __init__(self, add_effects, del_effects):
        super().__init__()
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)

    def compile(self) -> Callable[[StripsState], StripsState]:
        def function(state: StripsState, del_effects=self.del_effects, add_effects=self.add_effects) -> StripsState:
            new_state = (state - del_effects) | add_effects
            return StripsState(new_state)
        return function

    # Not used.
    # def compile_relaxed(self):
    #     def function(state: StripsState, add_effects=self.add_effects) -> StripsState:
    #         new_state = state | add_effects
    #         return StripsState(new_state)
    #     return function

    def iter_propositions(self) -> Iterable[StripsProposition]:
        yield from iter(self.add_effects)
        yield from iter(self.del_effects)

    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsSimpleAssignment':
        return GStripsSimpleAssignment(propositions & self.add_effects, propositions & self.del_effects)

    def relax(self) -> 'GStripsSimpleAssignment':
        return GStripsSimpleAssignment(self.add_effects, set())

    def __str__(self):
        return f'EFF[add={self.add_effects}, del={self.del_effects}]'


class GStripsSASAssignment(GStripsAssignmentExpression):
    def __init__(
        self,
        sas_name: StripsPredicateName, sas_size: int, arguments: Sequence[str],
        expression: Optional[Dict[int, GStripsClassifier]] = None,
    ):
        super().__init__()
        self.sas_name = sas_name
        self.sas_size = sas_size
        self.arguments = arguments
        self.arguments_str = ' '.join(str(arg) for arg in arguments)
        self.all_bool_predicates = frozenset({f'{sas_name}@{i} {self.arguments_str}' for i in range(sas_size)})
        self.all_neg_bool_predicates = frozenset({f'{sas_name}@{i}_not {self.arguments_str}' for i in range(sas_size)})
        self.expression = expression

    def compile(self) -> Callable[[StripsState], StripsState]:
        warnings.warn('SAS assignments are not supported. Run to_conditional_assignments() before compilation.', RuntimeWarning)
        expression_compiled = dict()
        for k, v in self.expression.items():
            expression_compiled[k] = v.compile()

        def function(
            state: StripsState, *,
            expression=expression_compiled,
            sas_name=self.sas_name, arguments_str=self.arguments_str, all_bool_predicates=self.all_bool_predicates,
        ) -> StripsState:
            new_value = None
            for k, v in expression.items():
                if v(state):
                    new_value = k
                    break
            if new_value is not None:
                current_value = state.intersection(all_bool_predicates)
                state = state - all_bool_predicates
                current_value_not = set()
                for v in current_value:
                    a = v.split()
                    a[0] = f'{a[0]}_not'
                    current_value_not.add(' '.join(a))

                state = state | current_value_not | {f'{sas_name}@{new_value} {arguments_str}'}
                return state
            return state
        return function

    def iter_propositions(self) -> Iterable[StripsProposition]:
        yield from self.all_bool_predicates
        for v in self.expression.values():
            yield from v.iter_propositions()

    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsSASAssignment':
        return GStripsSASAssignment(
            self.sas_name, self.sas_size, self.arguments,
            expression={k: v.filter_propositions(propositions, state=state) for k, v in self.expression.items()}
        )

    def to_conditional_assignments(self):
        rvs = list()
        for k, v in self.expression.items():
            this_add = {f'{self.sas_name}@{k} {self.arguments_str}'} | self.all_neg_bool_predicates - {f'{self.sas_name}@{k}_not {self.arguments_str}'}
            this_del = self.all_bool_predicates | {f'{self.sas_name}@{k}_not {self.arguments_str}'} - {f'{self.sas_name}@{k} {self.arguments_str}'}
            rvs.append(GStripsConditionalAssignment(v, GStripsSimpleAssignment(this_add, this_del)))
        return rvs

    def relax(self) -> 'GStripsSASAssignment':
        rvs = list()
        for k, v in self.expression.items():
            add_effects = {f'{self.sas_name}@{k} {self.arguments_str}'} | {f'{self.sas_name}@{i}_not {self.arguments_str}' for i in range(self.sas_size) if i != k}
            rvs.append(GStripsConditionalAssignment(v, GStripsSimpleAssignment(add_effects, set())))
        return rvs

    def __str__(self):
        expression_str = jacinle.stformat(self.expression).rstrip()
        return f'SAS[target={self.sas_name} {self.arguments_str}, value={expression_str}]'


class GStripsConditionalAssignment(GStripsAssignmentExpression):
    def __init__(self, condition: GStripsClassifier, assignment: GStripsSimpleAssignment):
        super().__init__()
        self.condition = condition
        self.assignment = assignment
        self.condition_func = None
        self.assignment_func = None

    @property
    def add_effects(self):
        assert isinstance(self.assignment, GStripsSimpleAssignment)
        return self.assignment.add_effects

    @property
    def del_effects(self):
        assert isinstance(self.assignment, GStripsSimpleAssignment)
        return self.assignment.del_effects

    def compile(self) -> Callable[[StripsState], StripsState]:
        condition_func = self.condition.compile()
        assignment_func = self.assignment.compile()
        def function(state: StripsState, condition_func=condition_func, assignment=assignment_func) -> StripsState:
            if condition_func(state):
                return assignment(state)
            return state
        self.condition_func = condition_func
        self.assignment_func = assignment_func
        return function

    def applicable(self, state: StripsState) -> Union[bool, GStripsClassifierForwardDiffReturn]:
        return self.condition_func(state)

    def apply(self, state: StripsState) -> StripsState:
        return self.assignment_func(state)

    def iter_propositions(self) -> Iterable[StripsProposition]:
        yield from self.assignment.iter_propositions()

    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsConditionalAssignment':
        return GStripsConditionalAssignment(
            self.condition.filter_propositions(propositions, state=state),
            self.assignment.filter_propositions(propositions, state=state)
        )

    def relax(self) -> 'GStripsConditionalAssignment':
        return GStripsConditionalAssignment(self.condition, self.assignment.relax())

    def __str__(self):
        if isinstance(self.assignment, GStripsSimpleAssignment):
            return f'CONDEFF[{self.condition} => add={self.assignment.add_effects}, del={self.assignment.del_effects}]'
        else:
            return f'CONDEFF[{self.condition} => {self.assignment}]'


class GStripsDerivedPredicate(GStripsExpression):
    def __init__(
        self,
        name,
        arguments,
        expression_true=None, expression_false=None, is_relaxed=False,
        effects: Optional[Iterable[GStripsConditionalAssignment]] = None
    ):
        super().__init__()
        self.name = name
        self.arguments = arguments

        self.pos_name = self.name + ' ' + ' '.join(str(x) for x in self.arguments)
        self.neg_name = self.name + '_not ' + ' '.join(str(x) for x in self.arguments)

        if effects is None:
            self.pos_effect = GStripsConditionalAssignment(expression_true, GStripsSimpleAssignment({self.pos_name}, {self.neg_name} if not is_relaxed else set()))
            self.neg_effect = GStripsConditionalAssignment(expression_false, GStripsSimpleAssignment({self.neg_name}, {self.pos_name} if not is_relaxed else set()))
            self.effects = [self.pos_effect, self.neg_effect]
        else:
            self.effects = effects

    def compile(self) -> Callable[[StripsState], StripsState]:
        effects_func = [effect.compile() for effect in self.effects]
        def function(state: StripsState, effects_func=effects_func) -> StripsState:
            for func in effects_func:
                state = func(state)
            return state
        return function

    def iter_propositions(self) -> Iterable[StripsProposition]:
        for eff in self.effects:
            yield from eff.iter_propositions()

    def iter_effect_propositions(self) -> Iterable[StripsProposition]:
        for eff in self.effects:
            yield from eff.assignment.iter_propositions()

    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsDerivedPredicate':
        return GStripsDerivedPredicate(
            self.name,
            self.arguments,
            effects=[eff.filter_propositions(propositions, state=state) for eff in self.effects]
        )

    def relax(self) -> 'GStripsDerivedPredicate':
        return GStripsDerivedPredicate(self.name, self.arguments, effects=[eff.relax() for eff in self.effects])

    def __str__(self):
        effects_str = '\n'.join(jacinle.indent_text(str(eff)) for eff in self.effects)
        return f'DERIVED[\n{effects_str}\n]'


# @jacinle.log_function
def _compose_strips_classifiers_inner(classifiers: Sequence[GStripsClassifier], is_disjunction: Optional[bool] = False) -> GStripsClassifier:
    new_classifiers = list()
    visited = [False for _ in classifiers]

    for i in range(len(classifiers)):
        if not visited[i]:
            c = classifiers[i]
            if gs_is_constant_true(c):
                visited[i] = True
                if is_disjunction:
                    return GStripsBoolConstant(True)
            elif gs_is_constant_false(c):
                visited[i] = True
                if not is_disjunction:
                    return GStripsBoolConstant(False)
            elif c == GS_OPTIMISTIC_STATIC_OBJECT:
                visited[i] = True

    new_set = set()

    def add_simple_classifier(c: GStripsSimpleClassifier):
        if c.is_disjunction == is_disjunction or len(c.classifier) == 1:
            new_set.update(c.classifier)
            return True
        return False

    for i in range(len(classifiers)):
        if not visited[i]:
            c = classifiers[i]
            if isinstance(c, GStripsSimpleClassifier):
                visited[i] = add_simple_classifier(c)

    complex = list()
    for i in range(len(classifiers)):
        if not visited[i]:
            c = classifiers[i]
            assert isinstance(c, (GStripsComplexClassifier, GStripsSimpleClassifier))
            if c.is_disjunction == is_disjunction:
                assert isinstance(c, GStripsComplexClassifier)
                for e in c.expressions:
                    if isinstance(e, GStripsSimpleClassifier):
                        if add_simple_classifier(e):
                            continue
                        else:
                            complex.append(e)
                    else:
                        complex.append(e)
            else:
                complex.append(c)

    if len(new_set) > 0:
        new_classifiers.append(GStripsSimpleClassifier(new_set, is_disjunction))
    new_classifiers.extend(complex)

    if len(new_classifiers) == 0:
        return GStripsBoolConstant(True if not is_disjunction else False)
    elif len(new_classifiers) == 1:
        return new_classifiers[0]
    else:
        return GStripsComplexClassifier(new_classifiers, is_disjunction)


def gstrips_compose_classifiers(
    classifiers: Union[Sequence[Tuple[GStripsClassifier, Set[StripsProposition]]], Sequence[GStripsClassifier]],
    is_disjunction: Optional[bool] = False,
    propagate_implicit_propositions: bool = True
) -> Union[Tuple[GStripsClassifier, Set[StripsProposition]], GStripsClassifier]:
    if propagate_implicit_propositions:
        rv = _compose_strips_classifiers_inner([c[0] for c in classifiers], is_disjunction)
        if gs_is_simple_empty_classifier(rv):
            return rv, set.union(*[c[1] for c in classifiers]) if len(classifiers) > 0 else set()
        return rv, set.union(*[c[1] for c in classifiers]) if len(classifiers) > 0 else set()
    else:
        rv = _compose_strips_classifiers_inner(classifiers, is_disjunction)
        return rv
