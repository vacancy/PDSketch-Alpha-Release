import itertools
from typing import Optional, Union, Callable, Iterable, Sequence, Iterator, Tuple, List, Set

import jacinle
from jacinle.utils.printing import indent_text

import hacl.pdsketch.interface.v2.expr as E
import hacl.pdsketch.interface.v2.strips.strips_expr as SE
from hacl.pdsketch.interface.v2.value import BOOL, Value
from hacl.pdsketch.interface.v2.expr import split_simple_bool, flatten_expression, Expression, ExpressionDefinitionContext, ExpressionExecutionContext, FeatureDef
from hacl.pdsketch.interface.v2.state import State
from hacl.pdsketch.interface.v2.domain import OperatorApplier, Domain
from hacl.pdsketch.interface.v2.optimistic import is_optimistic_value
from hacl.pdsketch.interface.v2.planner.optimistic_planner import generate_all_partially_grounded_actions
from .grounded_expr import GStripsClassifier, GStripsAssignmentExpression, GStripsClassifierForwardDiffReturn
from .grounded_expr import GS_OPTIMISTIC_STATIC_OBJECT, GSOptimisticStaticObjectType
from .grounded_expr import GStripsBoolConstant, GStripsSimpleClassifier, gstrips_compose_classifiers, gs_is_simple_empty_classifier
from .grounded_expr import GStripsSimpleAssignment, GStripsSASAssignment, GStripsConditionalAssignment, GStripsDerivedPredicate
from .strips_expr import StripsExpression, StripsPredicateName, StripsProposition, StripsState


__all__ = [
    'GStripsOperator', 'GStripsTask',
    'GStripsTranslatorBase', 'GStripsTranslatorOptimistic', 'GStripsTranslatorSAS', 'GStripsTranslator',
    'relevance_analysis_v1', 'relevance_analysis_v2'
]


class GStripsOperator(object):
    def __init__(
        self,
        precondition: GStripsClassifier,
        effects: Sequence[Union[GStripsSimpleAssignment, GStripsConditionalAssignment]],
        raw_operator: OperatorApplier,
        implicit_propositions: Optional[Set[StripsProposition]] = None
    ):
        self.precondition = precondition
        self.implicit_propositions = implicit_propositions if implicit_propositions is not None else set()
        self.effects = effects
        self.raw_operator = raw_operator
        self.precondition_func: Optional[Callable[[StripsState], bool]] = None
        self.effects_func: Optional[List[Callable[[StripsState], StripsState]]] = None

    def compile(self):
        self.precondition_func = self.precondition.compile()
        self.effects_func = [e.compile() for e in self.effects]

    def applicable(self, state: Union[StripsState, Set[StripsProposition]]) -> Union[bool, GStripsClassifierForwardDiffReturn]:
        return self.precondition_func(state)

    def apply(self, state: StripsState) -> StripsState:
        for effect_func in self.effects_func:
            state = effect_func(state)
        return state

    def iter_propositions(self) -> Iterable[StripsProposition]:
        yield from self.precondition.iter_propositions()
        for e in self.effects:
            yield from e.iter_propositions()

    def filter_propositions(self, propositions: Set[StripsProposition], state: Optional[StripsState] = None) -> 'GStripsOperator':
        return GStripsOperator(
            self.precondition.filter_propositions(propositions, state=state),
            [e.filter_propositions(propositions, state=state) for e in self.effects],
            self.raw_operator
        )

    def __str__(self):
        return (
            type(self).__name__
            + '{' + f'{self.raw_operator}' + '}'
            + '{\n' + indent_text(self.precondition)
            + '\n' + '\n'.join(indent_text(x) for x in self.effects) if len(self.effects) > 0 else '<Empty Effect>'
            + '\n}'
        )

    __repr__ = jacinle.repr_from_str


class GStripsTask(object):
    def __init__(
        self,
        state: StripsState,
        goal: GStripsClassifier,
        operators: list[GStripsOperator],
        is_relaxed: bool = False,
        goal_implicit_propositions: Optional[Set[StripsProposition]] = None,
        derived_predicates: Optional[List[GStripsDerivedPredicate]] = None,
        facts: Optional[Set[StripsProposition]] = None,
    ):
        self.state = state
        self.goal = goal
        self.operators = operators
        self.derived_predicates = derived_predicates if derived_predicates is not None else []
        self.is_relaxed = is_relaxed
        self.goal_implicit_propositions = goal_implicit_propositions if goal_implicit_propositions is not None else set()
        self._facts = facts

    def compile(self) -> 'GStripsTask':
        for op in self.operators:
            op.compile()
        for op in self.derived_predicates:
            op.compile()
        return self

    @property
    def facts(self):
        return self._facts

    def __str__(self):
        operator_str = '\n'.join(str(op) for op in self.operators)
        derived_predicate_str = '\n'.join(str(dp) for dp in self.derived_predicates)
        return f"""{type(self).__name__}{{
  state: {self.state}
  goal: {self.goal}
  operators:
    {indent_text(operator_str, 2).strip()}
  derived_predicates:
    {indent_text(derived_predicate_str, 2).strip()}
  facts: {self._facts}
}}"""

    __repr__ = jacinle.repr_from_str


class GStripsTranslatorBase(object):
    def __init__(self, domain: Domain, use_string_name: bool = True, prob_goal_threshold: float = 0.5, use_derived_predicates: bool = False):
        self.domain = domain
        self.use_string_name = use_string_name
        self.prob_goal_threshold = prob_goal_threshold
        self.use_derived_predicates = use_derived_predicates
        self.predicate2index = dict()
        self._init_indices()

    def _init_indices(self):
        raise NotImplementedError()

    def register_grounded_predicate(self, name: str, modifier: Optional[str] = None):
        """Allocate a new identifier for the predicate (with modifier).

        Args:
            name (str): the name of the predicate.
            modifier (str, optional): an optional modifier (e.g., not)
        """
        if not self.use_string_name:
            identifier = len(self.predicate2index)
        else:
            identifier = name + (f'_{modifier}' if modifier is not None else '')
        self.predicate2index[(name, modifier)] = identifier
        return identifier

    def get_grounded_predicate_indentifier(self, name: str, modifier: Optional[str] = None):
        return self.predicate2index[(name, modifier)]

    def compile_expr(self, expr: Union[str, Expression], state: State) -> Tuple[GStripsClassifier, Set[StripsProposition]]:
        raise NotImplementedError()

    def compile_operator(self, op: OperatorApplier, state: State, is_relaxed=False) -> GStripsOperator:
        raise NotImplementedError()

    def compile_derived_predicate(self, dp: FeatureDef, state: State, is_relaxed=False) -> List[GStripsDerivedPredicate]:
        raise NotImplementedError()

    def compile_state(self, state: State, forward_derived: bool = False) -> StripsState:
        raise NotImplementedError()

    def relevance_analysis(self, task: GStripsTask, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsTask:
        raise NotImplementedError()

    def compile_task(
        self,
        state: State,
        goal_expr: Union[str, Expression],
        actions: Optional[Sequence[OperatorApplier]] = None,
        is_relaxed = False,
        forward_relevance_analysis: bool = True,
        backward_relevance_analysis: bool = True,
        verbose: bool = False
    ) -> GStripsTask:
        with jacinle.cond_with(jacinle.time('compile_task::actions'), verbose):
            if actions is None:
                actions = generate_all_partially_grounded_actions(self.domain, state, filter_static=True)

        with jacinle.cond_with(jacinle.time('compile_task::state'), verbose):
            strips_state = self.compile_state(state)
        with jacinle.cond_with(jacinle.time('compile_task::operators'), verbose):
            strips_operators = [self.compile_operator(op, state, is_relaxed=is_relaxed) for op in actions]
        derived_predicates = list()
        if self.use_derived_predicates:
            with jacinle.cond_with(jacinle.time('compile_task::derived_predicates'), verbose):
                for pred in self.domain.features.values():
                    if pred.group not in ('basic', 'augmented') and pred.cacheable and pred.output_type == BOOL and not pred.static:
                        derived_predicates.extend(self.compile_derived_predicate(pred, state, is_relaxed=is_relaxed))
        with jacinle.cond_with(jacinle.time('compile_task::goal'), verbose):
            strips_goal, strips_goal_ip = self.compile_expr(goal_expr, state)
        task = GStripsTask(strips_state, strips_goal, strips_operators, is_relaxed=is_relaxed, goal_implicit_propositions=strips_goal_ip, derived_predicates=derived_predicates)
        with jacinle.cond_with(jacinle.time('compile_task::relevance_analysis'), verbose):
            if forward_relevance_analysis or backward_relevance_analysis:
                task = self.relevance_analysis(task, forward=forward_relevance_analysis, backward=backward_relevance_analysis)
        return task.compile()

    def recompile_relaxed_task(self, task: GStripsTask, forward_relevance_analysis: bool = True, backward_relevance_analysis: bool = True) -> GStripsTask:
        new_operators = task.operators.copy()
        for i, op in enumerate(new_operators):
            new_effects = []
            for e in op.effects:
                new_e = e.relax()
                if isinstance(new_e, (tuple, list)):
                    new_effects.extend(new_e)
                else:
                    new_effects.append(new_e)
            new_operators[i] = GStripsOperator(op.precondition, new_effects, op.raw_operator)
        task = GStripsTask(task.state, task.goal, new_operators, is_relaxed=True, derived_predicates=[dp.relax() for dp in task.derived_predicates], facts=task.facts)
        if forward_relevance_analysis or backward_relevance_analysis:
            task = self.relevance_analysis(task, forward=forward_relevance_analysis, backward=backward_relevance_analysis)
        return task.compile()

    def recompile_task_new_state(
        self,
        task: GStripsTask, new_state: Union[State, StripsState],
        forward_relevance_analysis: bool = True, backward_relevance_analysis: bool = True,
        forward_derived: bool = False
    ) -> GStripsTask:
        """
        Compile a new GStripsTask from a new state.

        Args:
            task (GStripsTask): the original task.
            new_state (State or StripsState): the new state.
            forward_relevance_analysis (bool, optional): whether to perform forward relevance analysis. Defaults to True.
            backward_relevance_analysis (bool, optional): whether to perform backward relevance analysis. Defaults to True.
            forward_derived (bool, optional): whether to forward derived predicates. Defaults to False.

        Returns:
            GStripsTask: the new task.
        """
        if isinstance(new_state, State):
            new_state = self.compile_state(new_state.clone(), forward_derived=forward_derived)
        if task.facts is not None:
            new_state = new_state & task.facts
        task = GStripsTask(new_state, task.goal, task.operators, is_relaxed=task.is_relaxed, derived_predicates=task.derived_predicates, facts=task.facts)
        if forward_relevance_analysis or backward_relevance_analysis:
            task = self.relevance_analysis(task, forward=forward_relevance_analysis, backward=backward_relevance_analysis)
            return task.compile()
        return task


class GStripsTranslatorOptimistic(GStripsTranslatorBase):
    def __init__(
        self,
        domain: Domain,
        use_string_name: Optional[bool] = True,
        prob_goal_threshold: float = 0.5,
    ):
        super().__init__(domain, use_string_name, prob_goal_threshold)

    def _init_indices(self):
        for pred in _find_cached_predicates(self.domain):
            if pred.output_type == BOOL:
                self.register_grounded_predicate(pred.name)
                self.register_grounded_predicate(pred.name, 'not')
            else:
                self.register_grounded_predicate(pred.name, 'initial')
                self.register_grounded_predicate(pred.name, 'optimistic')

    def compose_grounded_predicate(
        self, ctx: ExpressionExecutionContext, feature_app: E.FeatureApplication,
        negated: bool = False, optimistic: Optional[bool] = None,
        allow_set: bool = False,
        return_argument_indices: bool = False
    ) -> Union[StripsProposition, Tuple[StripsProposition, List[int]]]:
        arguments = list()
        for arg_index, arg in enumerate(feature_app.arguments):
            assert isinstance(arg, (E.ObjectConstantExpression, E.VariableExpression))
            if isinstance(arg, E.ObjectConstantExpression):
                arg = ctx.state.get_typed_index(arg.name)
                if allow_set:
                    arg = [arg]
            else:
                if arg.variable.name == '??':
                    assert allow_set
                    arg = list(range(ctx.state.get_nr_objects_by_type(feature_app.feature_def.arguments[arg_index].typename)))
                else:
                    arg = ctx.get_bounded_variable(arg.variable)
                    if allow_set:
                        arg = [arg]
            assert isinstance(arg, list) if allow_set else isinstance(arg, int)
            arguments.append(arg)

        if feature_app.output_type == BOOL:
            assert optimistic is None
            if allow_set:
                rv = set(
                    _format_proposition((self.get_grounded_predicate_indentifier(feature_app.feature_def.name, 'not' if negated else None),) + tuple(args))
                    for args in itertools.product(*arguments)
                )
            else:
                rv = _format_proposition((self.get_grounded_predicate_indentifier(feature_app.feature_def.name, 'not' if negated else None),) + tuple(arguments))
        else:
            assert not negated and optimistic is not None
            modifier = 'optimistic' if optimistic else 'initial'
            if allow_set:
                rv = set(
                    _format_proposition((self.get_grounded_predicate_indentifier(feature_app.feature_def.name, modifier),) + tuple(args))
                    for args in itertools.product(*arguments)
                )
            else:
                rv = _format_proposition((self.get_grounded_predicate_indentifier(feature_app.feature_def.name, modifier),) + tuple(arguments))

        if return_argument_indices:
            return rv, arguments
        return rv

    # @jacinle.log_function
    def compose_grounded_classifier(
        self,
        ctx: ExpressionExecutionContext,
        expr: E.ValueOutputExpression,
        negated: bool = False
    ) -> Tuple[Union[GStripsClassifier, GSOptimisticStaticObjectType], Set[StripsProposition]]:
        feature_app, this_negated = split_simple_bool(expr, initial_negated=negated)
        if feature_app is not None:
            # jacinle.log_function.print(feature_app, this_negated)
            if feature_app.feature_def.static:
                if feature_app.output_type == BOOL:
                    if feature_app.feature_def.name in ctx.state.features:
                        _, arguments = self.compose_grounded_predicate(ctx, feature_app, this_negated, return_argument_indices=True)
                        init_value = ctx.state.features[feature_app.feature_def.name][tuple(arguments)]
                    else:
                        init_value = expr.forward(ctx)
                    return GStripsBoolConstant(bool(init_value) ^ this_negated), set()
                else:
                    return GS_OPTIMISTIC_STATIC_OBJECT, set()

            if feature_app.output_type == BOOL:
                return GStripsSimpleClassifier(self.compose_grounded_predicate(ctx, feature_app, this_negated, allow_set=True), is_disjunction=True), set()
            else:
                return GStripsSimpleClassifier(self.compose_grounded_predicate(ctx, feature_app, this_negated, optimistic=True, allow_set=True), is_disjunction=True), set()
        elif E.is_not_expr(expr):
            return self.compose_grounded_classifier(ctx, expr.arguments[0], negated=not negated)
        elif E.is_and_expr(expr) and not negated or E.is_or_expr(expr) and negated:
            classifiers = [self.compose_grounded_classifier(ctx, e, negated=negated) for e in expr.arguments]
            rv = gstrips_compose_classifiers(classifiers, is_disjunction=False)
            return rv
        elif E.is_and_expr(expr) and negated or E.is_or_expr(expr) and not negated:
            classifiers = [self.compose_grounded_classifier(ctx, e, negated=negated) for e in expr.arguments]
            return gstrips_compose_classifiers(classifiers, is_disjunction=True)
        elif E.is_forall_expr(expr) and not negated or E.is_exists_expr(expr) and negated:
            classifiers = list()
            for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                with ctx.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_grounded_classifier(ctx, expr.expr, negated=negated))
            return gstrips_compose_classifiers(classifiers, is_disjunction=False)
        elif E.is_forall_expr(expr) and negated or E.is_exists_expr(expr) and not negated:
            classifiers = list()
            for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                with ctx.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_grounded_classifier(ctx, expr.expr, negated=negated))
            return gstrips_compose_classifiers(classifiers, is_disjunction=True)
        elif isinstance(expr, E.DeicticSelectOp):
            classifiers = list()
            for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                with ctx.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_grounded_classifier(ctx, expr.expr, negated=negated))
            return gstrips_compose_classifiers(classifiers, is_disjunction=True)
        elif isinstance(expr, E.ConditionalSelectOp):
            classifiers = [
                self.compose_grounded_classifier(ctx, expr.condition),
                self.compose_grounded_classifier(ctx, expr.feature)
            ]
            return gstrips_compose_classifiers(classifiers, is_disjunction=False)
        elif isinstance(expr, E.FeatureEqualOp):
            argument_values = [self.compose_grounded_classifier(ctx, arg) for arg in [expr.feature, expr.value]]
            has_optimistic_object = any(c[0] == GS_OPTIMISTIC_STATIC_OBJECT for c in argument_values)
            if has_optimistic_object:
                if expr.output_type == BOOL:
                    return GStripsBoolConstant(True), _extract_all_propositions(argument_values)
                else:
                    return GS_OPTIMISTIC_STATIC_OBJECT, _extract_all_propositions(argument_values)

            argument_values = [argv for argv in argument_values if not gs_is_simple_empty_classifier(argv)]
            init_value = expr.forward(ctx)
            if init_value.item() > self.prob_goal_threshold and not negated or init_value.item() < 1 - self.prob_goal_threshold and negated:
                return GStripsBoolConstant(True), _extract_all_propositions(argument_values)
            else:
                return gstrips_compose_classifiers(argument_values, is_disjunction=True)
        elif isinstance(expr, E.FunctionApplication):
            argument_values = [self.compose_grounded_classifier(ctx, arg) for arg in expr.arguments]
            # jacinle.log_function.print(argument_values)
            if GS_OPTIMISTIC_STATIC_OBJECT in argument_values:
                if expr.output_type == BOOL:
                    return GStripsBoolConstant(True), _extract_all_propositions(argument_values)
                else:
                    return GS_OPTIMISTIC_STATIC_OBJECT, _extract_all_propositions(argument_values)

            argument_values = [argv for argv in argument_values if not gs_is_simple_empty_classifier(argv)]
            # jacinle.log_function.print('computing initial value.')
            if expr.output_type == BOOL:
                # Theoretically, we can compute these values bottom-up together with the transformation.
                # In practice, this requires much more code to do...
                init_value = expr.forward(ctx)
                # jacinle.log_function.print('computed initial value:', init_value)
                if init_value.item() > self.prob_goal_threshold and not negated or init_value.item() < 1 - self.prob_goal_threshold and negated:
                    return GStripsBoolConstant(True), _extract_all_propositions(argument_values)
                else:
                    return gstrips_compose_classifiers(argument_values, is_disjunction=True)
            else:
                return gstrips_compose_classifiers(argument_values, is_disjunction=True)
        elif isinstance(expr, E.VariableExpression):
            assert expr.output_type != BOOL
            assert isinstance(ctx.get_bounded_variable(expr.variable), Value), 'Most likely you are accessing a non-optimistic object.'
            assert is_optimistic_value(ctx.get_bounded_variable(expr.variable).item())
            return GS_OPTIMISTIC_STATIC_OBJECT, set()
        else:
            raise TypeError('Unsupported expression grounding: {}.'.format(expr))

    def compose_grounded_assignment(
        self,
        ctx: ExpressionExecutionContext,
        assignments: Sequence[E.VariableAssignmentExpression],
        is_relaxed: bool = False
    ) -> Tuple[List[GStripsAssignmentExpression], Set[StripsProposition]]:
        add_effects = set()
        del_effects = set()
        implicit_propositions = set()
        outputs = list()
        for assign_expr in assignments:
            if isinstance(assign_expr, E.AssignOp):
                feat = assign_expr.feature
                if feat.output_type == BOOL:
                    assert E.is_constant_bool_expr(assign_expr.value)
                    if assign_expr.value.value.item():
                        add_effects.add(self.compose_grounded_predicate(ctx, feat, negated=False))
                        if not is_relaxed:
                            del_effects.add(self.compose_grounded_predicate(ctx, feat, negated=True))
                    else:
                        add_effects.add(self.compose_grounded_predicate(ctx, feat, negated=True))
                        if not is_relaxed:
                            del_effects.add(self.compose_grounded_predicate(ctx, feat, negated=False))
                else:
                    # For customized feature types, the "feat(...)" means that "this state variable has been set to an optimistic value."
                    add_effects.add(self.compose_grounded_predicate(ctx, feat, optimistic=True))
                    if not is_relaxed:
                        del_effects.add(self.compose_grounded_predicate(ctx, feat, optimistic=False))
                    value, ip = self.compose_grounded_classifier(ctx, assign_expr.value)
                    implicit_propositions = ip | set(value.iter_propositions())
            elif isinstance(assign_expr, E.ConditionalAssignOp):
                assignment, ass_ip = self.compose_grounded_assignment(ctx, [E.AssignOp(assign_expr.feature, assign_expr.value)], is_relaxed=is_relaxed)
                condition_classifier, cond_ip = self.compose_grounded_classifier(ctx, assign_expr.condition)
                outputs.append(GStripsConditionalAssignment(condition_classifier, assignment[0]))
                implicit_propositions = cond_ip | ass_ip
            elif isinstance(assign_expr, E.DeicticAssignOp):
                for index in range(ctx.state.get_nr_objects_by_type(assign_expr.variable.typename)):
                    with ctx.new_bounded_variables({assign_expr.variable: index}):
                        assignments, ass_ip = self.compose_grounded_assignment(ctx, [assign_expr.expr], is_relaxed=is_relaxed)
                        for assignment in assignments:
                            if isinstance(assignment, GStripsSimpleAssignment):
                                add_effects.update(assignment.add_effects)
                                del_effects.update(assignment.del_effects)
                            else:
                                outputs.append(assignment)
                        implicit_propositions.update(ass_ip)
        if len(add_effects) > 0 or len(del_effects) > 0:
            outputs.append(GStripsSimpleAssignment(add_effects, del_effects))
        return outputs, implicit_propositions

    def compile_expr(self, expr: Union[str, Expression], state: State) -> Tuple[GStripsClassifier, Set[StripsProposition]]:
        expr = self.domain.parse(expr)
        expr = flatten_expression(expr)
        ctx = ExpressionExecutionContext(self.domain, state, {})
        return self.compose_grounded_classifier(ctx, expr)

    def compile_operator(self, op: OperatorApplier, state: State, is_relaxed=False) -> GStripsOperator:
        # print('compile_operator:: {}'.format(op))

        if getattr(op.operator, 'flatten_precondition', None) is None:
            ctx = ExpressionDefinitionContext(*op.operator.arguments, domain=self.domain)
            precondition = E.AndOp(*[flatten_expression(e.bool_expr, ctx=ctx) for e in op.operator.preconditions])
            op.operator.flatten_precondition = precondition
        else:
            precondition = op.operator.flatten_precondition

        if getattr(op.operator, 'flatten_effects', None) is None:
            ctx = ExpressionDefinitionContext(*op.operator.arguments, domain=self.domain)
            effects = [flatten_expression(e.assign_expr, ctx=ctx) for e in op.operator.effects]
            op.operator.flatten_effects = effects
        else:
            effects = op.operator.flatten_effects

        # print('  precondition: {}'.format(precondition))
        ctx = ExpressionExecutionContext(self.domain, state, state.compose_bounded_variables(
            op.operator.arguments,
            op.arguments
        ))
        precondition, pre_ip = self.compose_grounded_classifier(ctx, precondition)
        effects, eff_ip = self.compose_grounded_assignment(ctx, effects, is_relaxed=is_relaxed)
        # print('  compiled precondition: {}'.format(precondition))
        return GStripsOperator(precondition, effects, op, implicit_propositions=pre_ip | eff_ip)

    def compile_state(self, state: State, forward_derived: bool = False) -> StripsState:
        predicates = set()
        for name, feature in state.features.items():
            if self.domain.features[name].static:
                continue
            if self.domain.features[name].group not in ('basic', 'augmented'):
                continue
            if feature.dtype == BOOL:
                for args, v in _iter_value(feature):
                    if v > 0.5:
                        predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name),) + args))
                    else:
                        predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name, 'not'),) + args))
            else:
                for args, _ in _iter_value(feature):
                    predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name, 'initial'),) + args))
        return StripsState(predicates)

    def compile_derived_predicate(self, dp: FeatureDef, state: State, is_relaxed=False) -> List[GStripsDerivedPredicate]:
        raise NotImplementedError('Derived predicates are not supported in Optimistic GStrips translation.')

    def relevance_analysis(self, task: GStripsTask, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsTask:
        return relevance_analysis_v1(task, relaxed_relevance=relaxed_relevance, forward=forward, backward=backward)


GStripsTranslator = GStripsTranslatorOptimistic


class GStripsTranslatorSAS(GStripsTranslatorBase):
    def __init__(
        self,
        domain: Domain,
        use_string_name: Optional[bool] = True,
        prob_goal_threshold: float = 0.5,
        cache_bool_features: bool = False
    ):
        self.cache_bool_features = cache_bool_features
        super().__init__(domain, use_string_name, prob_goal_threshold, use_derived_predicates=cache_bool_features)

    def _init_indices(self):
        for pred in _find_cached_predicates(self.domain, allow_cacheable_bool=self.cache_bool_features):
            if pred.output_type == BOOL:
                self.register_grounded_predicate(pred.name)
                self.register_grounded_predicate(pred.name, 'not')
            else:
                for i in range(pred.ao_discretization.size):
                    self.register_grounded_predicate(f'{pred.name}@{i}')
                    self.register_grounded_predicate(f'{pred.name}@{i}', 'not')

    def compose_grounded_predicate_strips(
        self, ctx:ExpressionExecutionContext, feature_app: SE.StripsBoolPredicate,
        negated: bool = False
    ) -> Union[GStripsSimpleClassifier, GStripsBoolConstant]:
        argument_indices = list()
        for arg_index, arg in enumerate(feature_app.arguments):
            argument_indices.append(ctx.get_bounded_variable(arg))

        feature_name = feature_app.sas_name if isinstance(feature_app, SE.StripsSASPredicate) else feature_app.name
        feature_def = self.domain.features[feature_name]
        if feature_def.static:
            if isinstance(feature_app, SE.StripsSASPredicate):
                value = ctx.state.features[feature_name].tensor_indices[tuple(argument_indices)]
                return GStripsBoolConstant(value.item() == feature_app.sas_index ^ negated ^ feature_app.negated)
            else:
                value = ctx.state.features[feature_name][tuple(argument_indices)]
                return GStripsBoolConstant((value.item() > 0.5) ^ negated ^ feature_app.negated)

        predicate_name = self.get_grounded_predicate_indentifier(feature_app.name, 'not' if negated ^ feature_app.negated else None)
        return GStripsSimpleClassifier({_format_proposition((predicate_name, ) + tuple(argument_indices))})

    def compose_grounded_predicate(self, ctx:ExpressionExecutionContext, feature_app: E.FeatureApplication, negated: bool = False) -> Union[GStripsSimpleClassifier, GStripsBoolConstant]:
        argument_indices = list()
        for arg_index, arg in enumerate(feature_app.arguments):
            if isinstance(arg, E.ObjectConstantExpression):
                arg = ctx.state.get_typed_index(arg.name)
            else:
                assert isinstance(arg, E.VariableExpression)
                arg = ctx.get_bounded_variable(arg.variable)
            argument_indices.append(arg)
        feature_def = feature_app.feature_def
        feature_name = feature_def.name

        if feature_def.static:
            value = ctx.state.features[feature_name][tuple(argument_indices)]
            assert value.dtype == BOOL
            return GStripsBoolConstant((value.item() > 0.5) ^ negated)

        predicate_name = self.get_grounded_predicate_indentifier(feature_def.name, 'not' if negated else None)
        return GStripsSimpleClassifier({_format_proposition((predicate_name,) + tuple(argument_indices))})

    def _compose_grounded_classifier_strips(self, ctx: ExpressionExecutionContext, expr: StripsExpression, negated: bool = False) -> Union[GStripsClassifier, StripsProposition, GSOptimisticStaticObjectType]:
        if isinstance(expr, SE.StripsBoolConstant):
            return GStripsBoolConstant(expr.value ^ negated)
        elif isinstance(expr, SE.StripsBoolNot):
            return self._compose_grounded_classifier_strips(ctx, expr.expr, not negated)
        elif isinstance(expr, SE.StripsBoolAOFormula):
            classifiers = [self._compose_grounded_classifier_strips(ctx, e, negated) for e in expr.arguments]
            if expr.is_conjunction and not negated or expr.is_disjunction and negated:
                return gstrips_compose_classifiers(classifiers, is_disjunction=False, propagate_implicit_propositions=False)
            else:
                return gstrips_compose_classifiers(classifiers, is_disjunction=True, propagate_implicit_propositions=False)
        elif isinstance(expr, SE.StripsBoolFEFormula):
            classifiers = list()
            for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                with ctx.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self._compose_grounded_classifier_strips(ctx, expr.expr, negated))
            if expr.is_forall and not negated or expr.is_exists and negated:
                return gstrips_compose_classifiers(classifiers, is_disjunction=False, propagate_implicit_propositions=False)
            else:
                return gstrips_compose_classifiers(classifiers, is_disjunction=True, propagate_implicit_propositions=False)
        elif isinstance(expr, SE.StripsBoolPredicate):
            return self.compose_grounded_predicate_strips(ctx, expr, negated)
        else:
            raise TypeError('Unknown expression type: {}.'.format(expr))

    def compose_grounded_classifier(
        self,
        ctx: ExpressionExecutionContext,
        expr: E.ValueOutputExpression,
        negated: bool = False
    ) -> Union[GStripsClassifier, GSOptimisticStaticObjectType]:
        if isinstance(expr, E.FeatureApplication):
            feature_def = expr.feature_def
            assert feature_def.cacheable and feature_def.output_type == BOOL
            if feature_def.expr is None or self.cache_bool_features:  # a basic predicate.
                return self.compose_grounded_predicate(ctx, expr, negated)
            else:
                return self._compose_grounded_classifier_strips(ctx, feature_def.ao_discretization, negated)
        elif E.is_not_expr(expr):
            return self.compose_grounded_classifier(ctx, expr.arguments[0], negated=not negated)
        elif E.is_and_expr(expr) and not negated or E.is_or_expr(expr) and negated:
            classifiers = [self.compose_grounded_classifier(ctx, e, negated=negated) for e in expr.arguments]
            rv = gstrips_compose_classifiers(classifiers, is_disjunction=False, propagate_implicit_propositions=False)
            return rv
        elif E.is_and_expr(expr) and negated or E.is_or_expr(expr) and not negated:
            classifiers = [self.compose_grounded_classifier(ctx, e, negated=negated) for e in expr.arguments]
            return gstrips_compose_classifiers(classifiers, is_disjunction=True, propagate_implicit_propositions=False)
        elif E.is_forall_expr(expr) and not negated or E.is_exists_expr(expr) and negated:
            classifiers = list()
            for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                with ctx.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_grounded_classifier(ctx, expr.expr, negated=negated))
            return gstrips_compose_classifiers(classifiers, is_disjunction=False, propagate_implicit_propositions=False)
        elif E.is_forall_expr(expr) and negated or E.is_exists_expr(expr) and not negated:
            classifiers = list()
            for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                with ctx.new_bounded_variables({expr.variable: index}):
                    classifiers.append(self.compose_grounded_classifier(ctx, expr.expr, negated=negated))
            return gstrips_compose_classifiers(classifiers, is_disjunction=True, propagate_implicit_propositions=False)
        else:
            raise TypeError('Unsupported expression grounding: {}.'.format(expr))

    def _compose_grounded_assignment_strips(
        self,
        ctx: ExpressionExecutionContext,
        assignments: Sequence[SE.StripsVariableAssignmentExpression],
        is_relaxed: bool = False
    ) -> List[GStripsAssignmentExpression]:
        add_effects = set()
        del_effects = set()
        outputs = list()

        for expr in assignments:
            if isinstance(expr, SE.StripsDeicticAssignment):
                for index in range(ctx.state.get_nr_objects_by_type(expr.variable.typename)):
                    with ctx.new_bounded_variables({expr.variable: index}):
                        this_outputs = self._compose_grounded_assignment_strips(ctx, [expr.expr], is_relaxed)
                        for ass in this_outputs:
                            if isinstance(ass, GStripsSimpleAssignment):
                                add_effects.update(ass.add_effects)
                                del_effects.update(ass.del_effects)
                            else:
                                outputs.append(ass)
            elif isinstance(expr, SE.StripsConditionalAssignment):
                assignments = self._compose_grounded_assignment_strips(ctx, [expr.assign_op], is_relaxed)
                condition = self._compose_grounded_classifier_strips(ctx, expr.condition)
                for ass in assignments:
                    if isinstance(ass, GStripsSimpleAssignment):
                        outputs.append(GStripsConditionalAssignment(condition, ass))
                    elif isinstance(ass, GStripsConditionalAssignment):
                        outputs.append(GStripsConditionalAssignment(gstrips_compose_classifiers([condition, ass.condition], propagate_implicit_propositions=False), ass.assignment))
                    else:
                        raise TypeError('Invalid assignment type: {}.'.format(ass))
            elif isinstance(expr, SE.StripsAssignment):
                if isinstance(expr.feature, SE.StripsSASPredicate):
                    if is_relaxed:
                        raise NotImplementedError('Relaxed assignment to SAS predicate not supported during compilation. First compile it without is_relaxed, and re-run recompile_relaxed_operators.')
                    feature = expr.feature
                    feature_name = feature.sas_name
                    feature_def = self.domain.features[feature_name]
                    feature_sas_size = feature_def.ao_discretization.size
                    assert isinstance(expr.value, SE.StripsSASExpression)
                    argument_indices = list()
                    for arg_index, arg in enumerate(feature.arguments):
                        argument_indices.append(ctx.get_bounded_variable(arg))
                    expression = {k: self._compose_grounded_classifier_strips(ctx, v) for k, v in expr.value.mappings.items()}
                    sas_assignment = GStripsSASAssignment(feature_name, feature_sas_size, argument_indices, expression)
                    outputs.extend(sas_assignment.to_conditional_assignments())
                else:
                    feature = expr.feature
                    value = bool(expr.value)
                    if value:
                        add_effects.add(self.compose_grounded_predicate_strips(ctx, feature))
                        if not is_relaxed:
                            add_effects.add(self.compose_grounded_predicate_strips(ctx, feature, negated=True))
                    else:
                        add_effects.add(self.compose_grounded_predicate_strips(ctx, feature, negated=True))
                        if not is_relaxed:
                            add_effects.add(self.compose_grounded_predicate_strips(ctx, feature))

        if len(add_effects) > 0 or len(del_effects) > 0:
            outputs.append(GStripsSimpleAssignment(add_effects, del_effects))
        return outputs

    def compile_expr(self, expr: Union[str, Expression], state: State) -> Tuple[GStripsClassifier, Set[StripsProposition]]:
        expr = self.domain.parse(expr)
        expr = flatten_expression(expr, flatten_cacheable_bool=not self.cache_bool_features)
        ctx = ExpressionExecutionContext(self.domain, state, {})
        return self.compose_grounded_classifier(ctx, expr), set()

    def compile_operator(self, op: OperatorApplier, state: State, is_relaxed=False) -> GStripsOperator:
        ctx = ExpressionExecutionContext(self.domain, state, state.compose_bounded_variables(
            op.operator.arguments,
            op.arguments
        ))
        preconditions = list()
        for pred in op.operator.preconditions:
            preconditions.append(self._compose_grounded_classifier_strips(ctx, pred.ao_discretization))
        precondition = gstrips_compose_classifiers(preconditions, is_disjunction=False, propagate_implicit_propositions=False)
        effects = self._compose_grounded_assignment_strips(ctx, [eff.ao_discretization for eff in op.operator.effects], is_relaxed)
        return GStripsOperator(precondition, effects, op, implicit_propositions=set())

    def compile_derived_predicate(self, dp: FeatureDef, state: State, is_relaxed=False) -> List[GStripsDerivedPredicate]:
        arguments = list()
        for arg in dp.arguments:
            arguments.append(range(state.get_nr_objects_by_type(arg.type.typename)))

        rvs = list()
        for arg_indices in itertools.product(*arguments):
            bounded_variables = dict()
            for arg, arg_index in zip(dp.arguments, arg_indices):
                bounded_variables.setdefault(arg.typename, dict())[arg.name] = arg_index
            ctx = ExpressionExecutionContext(self.domain, state, bounded_variables)
            rvs.append(GStripsDerivedPredicate(
                dp.name, arg_indices,
                self._compose_grounded_classifier_strips(ctx, dp.ao_discretization),
                self._compose_grounded_classifier_strips(ctx, dp.ao_discretization, negated=True),
                is_relaxed=is_relaxed
            ))
        return rvs

    def compile_state(self, state: State, forward_derived: bool = False) -> StripsState:
        # Note: this function will change the original values of the state.
        # So be sure to make a copy of the state before calling this function.
        # This copying behavior is implemented in the compile_task function. If you are calling this function
        # directly, make sure to copy the state before calling this function.

        if forward_derived and self.cache_bool_features:
            self.domain.forward_features_and_axioms(state, forward_augmented=False, forward_derived=True, forward_axioms=False)

        for name, feature in state.features.items():
            feature_def = self.domain.features[name]
            if feature_def.group in ('basic', 'augmented') and not (feature_def.output_type == BOOL):
                state.features[name].tensor_indices = feature_def.ao_discretization.quantize(feature).tensor

        predicates = set()
        for name, feature in state.features.items():
            feature_def = self.domain.features[name]
            if feature_def.group in ('basic', 'augmented') or (self.cache_bool_features and feature_def.output_type == BOOL):
                if feature.dtype == BOOL:
                    for args, v in _iter_value(feature):
                        if v > 0.5:
                            predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name),) + args))
                        else:
                            predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(name, 'not'),) + args))
                else:
                    codebook = feature_def.ao_discretization
                    quantized_feature = codebook.quantize(feature)
                    for args, v in _iter_value(quantized_feature):
                        v = int(v)
                        for i in range(codebook.size):
                            if i == v:
                                predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(f'{name}@{i}'),) + args))
                            else:
                                predicates.add(_format_proposition((self.get_grounded_predicate_indentifier(f'{name}@{i}', 'not'),) + args))

        return StripsState(predicates)

    def compile_task(
        self,
        state: State,
        goal_expr: Union[str, Expression],
        actions: Optional[Sequence[OperatorApplier]] = None,
        is_relaxed = False,
        forward_relevance_analysis: bool = True,
        backward_relevance_analysis: bool = True,
        verbose: bool = False
    ) -> GStripsTask:
        state = state.clone()
        if self.cache_bool_features:
            self.domain.forward_features_and_axioms(state, forward_augmented=True, forward_derived=True, forward_axioms=False)
        return super().compile_task(
            state, goal_expr, actions, is_relaxed,
            forward_relevance_analysis=forward_relevance_analysis, backward_relevance_analysis=backward_relevance_analysis,
            verbose=verbose
        )

    def relevance_analysis(self, task: GStripsTask, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsTask:
        return relevance_analysis_v2(task, relaxed_relevance=relaxed_relevance, forward=forward, backward=backward)


def _find_cached_predicates(domain: Domain, allow_cacheable_bool: bool = False) -> Iterable[FeatureDef]:
    """
    Return the set of predicates that are either in the `basic` or the `augmented` group.
    When the flag allow_cacheable_bool is set to True, also return the set of boolean predicates that are cacheable.

    Args:
        domain: the domain to search for predicates
        allow_cacheable_bool: whether to return the set of boolean predicates that are cacheable

    Returns:
        the set of predicates that are either in the `basic` or the `augmented` group and optionally cacheable boolean predicates.
    """
    for f in domain.features.values():
        if f.group in ('basic', 'augmented') and f.cacheable:
            yield f
        elif allow_cacheable_bool and f.cacheable and f.output_type == BOOL:
            yield f


def _iter_value(value: Value) -> Iterator[Tuple[Tuple[int, ...], Union[bool, int, float]]]:
    indices = [list(range(value.tensor.size(i))) for i in range(value.total_batch_dims)]
    for args, x in zip(itertools.product(*indices), value.tensor.flatten().tolist()):
        yield args, x


def _extract_all_propositions(classifiers: Sequence[Tuple[GStripsClassifier, Set[StripsProposition]]]) -> Set[StripsProposition]:
    return set.union(*[c[1] for c in classifiers], *[c[0].iter_propositions() for c in classifiers if isinstance(c[0], GStripsClassifier)])


def _format_proposition(pred_application: tuple[Union[StripsPredicateName, int], ...]) -> StripsProposition:
    return ' '.join([str(x) for x in pred_application])


def relevance_analysis_v1(task: GStripsTask, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsTask:
    """
    Relevance analysis for a task.

    Args:
        task: the StripsTask to be analyzed.
        relaxed_relevance: whether to use relaxed relevance analysis. If this is set to False, we will not drop functions that are "implicitly" used by
        the operators. One example is that if one of the effect of the operator is a function (instead of simply Y/N).

    Returns:
        the updated task, removing all irrelevant propositions and operators.
    """

    # forward_analysis. a.k.a. static analysis.
    # useful when most objects are "non-movable". Then we don't need to track
    # their state variables/pose variables.

    # print('relevance_analysis::before')
    # print(task)
    if len(task.derived_predicates) > 0:
        raise RuntimeError('relevance_analysis::task.derived_predicates is not supported in relevance_analysis_v1.')

    if forward:
        task.compile()
        achievable_facts = set(task.state)
        changed = True
        visited = [False for _ in range(len(task.operators))]
        while changed:
            old_lengths = len(achievable_facts)
            for i, op in enumerate(task.operators):
                if not visited[i] and op.applicable(achievable_facts):
                    for e in op.effects:
                        achievable_facts.update(e.add_effects)
                    visited[i] = True
            changed = len(achievable_facts) != old_lengths
        new_operators = [op for i, op in enumerate(task.operators) if visited[i]]

        relevant_facts = set()
        for op in new_operators:
            for e in op.effects:
                relevant_facts.update(e.iter_propositions())

        # Note:: it seems that even when the task is not relaxed, we can also
        # do this drop...
        # Basically, if goal - relevant_facts isn't a subset of the initial
        # state, the problem is just unsolvable.
        # But when there are disjunctions, it's a bit hard to check.
        # relevant_facts.update(task.goal.iter_propositions())

        new_state = task.state.intersection(relevant_facts)
        new_operators = [op.filter_propositions(relevant_facts, state=task.state) for op in new_operators]
        new_goal = task.goal.filter_propositions(relevant_facts, state=task.state)

        # import ipdb; ipdb.set_trace()

        task = GStripsTask(new_state, new_goal, new_operators, is_relaxed=task.is_relaxed, facts=relevant_facts)
        task.compile()

        # print('relevance_analysis::forward')
        # print(task)

    # backward analysis.
    if backward:
        relevant_facts = set()
        relevant_facts.update(task.goal.iter_propositions())
        relevant_facts.update(task.goal_implicit_propositions)

        op_eff_facts = list()
        for op in task.operators:
            effects = set()
            for e in op.effects:
                effects.update(e.iter_propositions())
            op_eff_facts.append(effects)

        changed = True
        while changed:
            old_lengths = len(relevant_facts)
            for op, eff_facts in zip(task.operators, op_eff_facts):
                if set.intersection(eff_facts, relevant_facts):
                    relevant_facts |= set(op.precondition.iter_propositions())
                    if not relaxed_relevance:
                        relevant_facts |= set(op.implicit_propositions)
                    for e in op.effects:
                        if isinstance(e, GStripsConditionalAssignment):
                            relevant_facts |= set(e.condition.iter_propositions())
            changed = len(relevant_facts) != old_lengths

        new_operators = list()
        for op in task.operators:
            new_op = op.filter_propositions(relevant_facts, state=task.state)
            empty = True
            for e in new_op.effects:
                if len(e.add_effects.symmetric_difference(e.del_effects)) > 0:
                    empty = False
                    break
                if isinstance(e, GStripsConditionalAssignment) and len(e.condition.iter_propositions()) > 0:
                    empty = False
                    break
            if not empty:
                new_operators.append(new_op)

        new_state = task.state.intersection(relevant_facts)
        task = GStripsTask(new_state, task.goal, new_operators, is_relaxed=task.is_relaxed, facts=relevant_facts)

        # print('relevance_analysis::backward')
        # print(task)

    return task


def relevance_analysis_v2(task: GStripsTask, relaxed_relevance: bool = False, forward: bool = True, backward: bool = True) -> GStripsTask:
    """
    Relevance analysis for a task.

    Args:
        task: the StripsTask to be analyzed.
        relaxed_relevance: whether to use relaxed relevance analysis. If this is set to False, we will not drop functions that are "implicitly" used by
        the operators. One example is that if one of the effect of the operator is a function (instead of simply Y/N).
        forward: whether to run the forward pruning (forward reachability checking).
        backward: whether to run the backward pruning (goal regression).

    Returns:
        the updated task, removing all irrelevant propositions and operators.
    """

    # forward_analysis. a.k.a. static analysis.
    # useful when most objects are "non-movable". Then we don't need to track
    # their state variables/pose variables.

    # import ipdb; ipdb.set_trace()
    # print('relevance_analysis::before')
    # print(task)

    if forward:
        # collect all operators and derived predicates applicable.
        used_ops = set()
        used_dps = set()
        task.compile()
        achievable_facts = set(task.state)
        for i, dp in enumerate(task.derived_predicates):
            for j, eff in enumerate(dp.effects):
                if eff.applicable(achievable_facts):
                    achievable_facts.update(eff.add_effects)
                    used_dps.add((i, j))

        changed = True
        while changed:
            old_lengths = len(achievable_facts)
            for i, op in enumerate(task.operators):
                applicable = op.applicable(achievable_facts)
                if applicable:
                    for j, eff in enumerate(op.effects):
                        if (i, j) not in used_ops:
                            if isinstance(eff, GStripsSimpleAssignment) or isinstance(eff, GStripsConditionalAssignment) and eff.applicable(achievable_facts):
                                achievable_facts.update(eff.add_effects)
                                used_ops.add((i, j))
            for i, dp in enumerate(task.derived_predicates):
                for j, eff in enumerate(dp.effects):
                    if (i, j) not in used_dps and eff.applicable(achievable_facts):
                        achievable_facts.update(eff.add_effects)
                        used_dps.add((i, j))
            changed = len(achievable_facts) != old_lengths

        new_operators = list()
        for i, op in enumerate(task.operators):
            used_effects = list()
            for j, eff in enumerate(op.effects):
                if (i, j) in used_ops:
                    used_effects.append(eff)
            if len(used_effects) > 0:
                new_operators.append(GStripsOperator(op.precondition, used_effects, op.raw_operator, op.implicit_propositions))
        new_derived_predicates = list()
        for i, dp in enumerate(task.derived_predicates):
            used_effects = list()
            for j, eff in enumerate(dp.effects):
                if (i, j) in used_dps:
                    used_effects.append(eff)
            if len(used_effects) > 0:
                new_derived_predicates.append(GStripsDerivedPredicate(dp.name, dp.arguments, effects=used_effects))

        relevant_facts = set()
        for op in new_operators:
            for e in op.effects:
                relevant_facts.update(e.iter_propositions())
        for i, dp in enumerate(task.derived_predicates):
            for e in dp.effects:
                relevant_facts.update(e.assignment.iter_propositions())

        # Note:: it seems that even when the task is not relaxed, we can also
        # do this drop...
        # Basically, if goal - relevant_facts isn't a subset of the initial
        # state, the problem is just unsolvable.
        # But when there are disjunctions, it's a bit hard to check.
        # relevant_facts.update(task.goal.iter_propositions())

        new_state = task.state.intersection(relevant_facts)
        new_operators = [op.filter_propositions(relevant_facts, state=task.state) for op in new_operators]
        new_derived_predicates = [dp.filter_propositions(relevant_facts, state=task.state) for dp in task.derived_predicates]
        new_goal = task.goal.filter_propositions(relevant_facts, state=task.state)

        task = GStripsTask(new_state, new_goal, new_operators, is_relaxed=task.is_relaxed, derived_predicates=new_derived_predicates, facts=relevant_facts)
        task.compile()

        # import ipdb; ipdb.set_trace()
        # print('relevance_analysis::forward')
        # print(task)

    if backward:
        # backward analysis.
        relevant_facts = set()
        relevant_facts.update(task.goal.iter_propositions())
        relevant_facts.update(task.goal_implicit_propositions)

        changed = True
        while changed:
            old_lengths = len(relevant_facts)
            for i, dp in enumerate(task.derived_predicates):
                for j, eff in enumerate(dp.effects):
                    if set.intersection(set(eff.iter_propositions()), relevant_facts):
                        relevant_facts.update(eff.condition.iter_propositions())

            for i, op in enumerate(task.operators):
                for j, eff in enumerate(op.effects):
                    if set.intersection(set(eff.iter_propositions()), relevant_facts):
                        relevant_facts |= set(op.precondition.iter_propositions())
                        if not relaxed_relevance:
                            relevant_facts |= set(op.implicit_propositions)
                        if isinstance(eff, GStripsConditionalAssignment):
                            relevant_facts |= set(eff.condition.iter_propositions())
            changed = len(relevant_facts) != old_lengths

        new_operators = list()
        for op in task.operators:
            new_op = op.filter_propositions(relevant_facts, state=task.state)
            new_effects = list()
            for j, eff in enumerate(new_op.effects):
                if len(eff.add_effects.symmetric_difference(eff.del_effects)) > 0:
                    new_effects.append(eff)
            if len(new_effects) > 0:
                new_op = GStripsOperator(new_op.precondition, new_effects, new_op.raw_operator, new_op.implicit_propositions)
                new_operators.append(new_op)

        new_derived_predicates = list()
        for dp in task.derived_predicates:
            new_dp = dp.filter_propositions(relevant_facts, state=task.state)
            new_effects = list()
            for j, eff in enumerate(new_dp.effects):
                if len(eff.add_effects.symmetric_difference(eff.del_effects)) > 0:
                    new_effects.append(eff)
            if len(new_effects) > 0:
                new_dp = GStripsDerivedPredicate(new_dp.name, new_dp.arguments, effects=new_effects)
                new_derived_predicates.append(new_dp)

        new_state = task.state.intersection(relevant_facts)
        task = GStripsTask(new_state, task.goal, new_operators, is_relaxed=task.is_relaxed, derived_predicates=new_derived_predicates, facts=relevant_facts)

        # import ipdb; ipdb.set_trace()
        # print('relevance_analysis::backward')
        # print(task)

    return task
