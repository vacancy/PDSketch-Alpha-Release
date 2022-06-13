import collections
import itertools
import heapq as hq
import jacinle
import jactorch
import torch
from typing import Optional, Union, Callable, Tuple, Sequence, List, Mapping, Any

from hacl.pdsketch.interface.v2.value import ObjectType, NamedValueType, NamedValueTypeSlot, Value, wrap_value
from hacl.pdsketch.interface.v2.optimistic import EqualOptimisticConstraint, OptimisticValueContext, is_optimistic_value, optimistic_value_id, OptimisticValue, DeterminedValue
from hacl.pdsketch.interface.v2.state import State
from hacl.pdsketch.interface.v2.expr import ValueOutputExpression, ExpressionExecutionContext
from hacl.pdsketch.interface.v2.domain import Domain, OperatorApplier, ValueQuantizer
from hacl.pdsketch.interface.v2.csp_solver.simple_csp import simple_csp_solve, CSPNoGenerator
from hacl.algorithms.poc.heuristic_search import QueueNode
from .basic_planner import filter_static_grounding
from .prob_goal import MostPromisingTrajectoryTracker

__all__ = [
    'generate_all_partially_grounded_actions',
    'instantiate_action', 'ground_actions', 'apply_action', 'goal_test',
    'optimistic_search_domain_check', 'prepare_optimistic_search',
    'OptimisticSearchState',
    'solve_optimistic_plan', 'optimistic_search', 'optimistic_search_with_heuristic',
]


def generate_all_partially_grounded_actions(
    domain: Domain,
    state: State,
    action_names: Optional[Sequence[str]] = None,
    action_filter: Optional[Callable[[OperatorApplier], bool]] = None,
    filter_static: Optional[bool] = True
) -> List[OperatorApplier]:
    if action_names is not None:
        action_ops = [domain.operators[x] for x in action_names]
    else:
        action_ops = domain.operators.values()

    actions = list()
    for op in action_ops:
        argument_candidates = list()
        for arg in op.arguments:
            if isinstance(arg.type, ObjectType):
                argument_candidates.append(state.object_type2name[arg.type.typename])
            else:
                assert isinstance(arg.type, NamedValueType)
                argument_candidates.append([NamedValueTypeSlot(arg.type)])
        for comb in itertools.product(*argument_candidates):
            actions.append(op(*comb))

    if filter_static:
        actions = filter_static_grounding(domain, state, actions)
    if action_filter is not None:
        actions = list(filter(action_filter, actions))
    return actions


def instantiate_action(csp: OptimisticValueContext, action: OperatorApplier):
    new_argumnets = list()
    for arg in action.arguments:
        if isinstance(arg, NamedValueTypeSlot):
            new_argumnets.append(Value(arg.type, [], torch.tensor(csp.new_actionable_var(arg.type), dtype=torch.int64), quantized=True))
        else:
            new_argumnets.append(arg)
    return OperatorApplier(action.operator, *new_argumnets)


def ground_actions(domain, actions: Sequence[OperatorApplier], assignments: Mapping[int, Any]):
    output_actions = list()
    for action in actions:
        new_arguments = list()
        for arg in action.arguments:
            if isinstance(arg, Value):
                if arg.quantized:
                    arg = arg.item()
                    assert isinstance(arg, int)
                    if is_optimistic_value(arg):
                        new_arguments.append(assignments[arg].d)
                    else:
                        assert isinstance(arg.type, NamedValueType)
                        new_arguments.append(domain.value_quantizer.unquantize(arg.type.typename, arg))
                else:
                    new_arguments.append(arg)
            else:
                assert isinstance(arg, str)
                new_arguments.append(arg)
        output_actions.append(OperatorApplier(action.operator, *new_arguments))
    return output_actions


def apply_action(domain: Domain, state: State, action: OperatorApplier, csp: OptimisticValueContext):
    csp = csp.clone()
    return action(state, csp), csp


def goal_test(
    domain: Domain, state: State, goal_expr: ValueOutputExpression, csp: OptimisticValueContext,
    trajectory, csp_max_generator_trials=3,
    mpt_tracker: Optional[MostPromisingTrajectoryTracker] = None, verbose: bool = False
):
    csp = csp.clone()
    ctx = ExpressionExecutionContext(domain, state, {}, optimistic_context=csp)
    with ctx.as_default():
        rv = goal_expr.forward(ctx).item()
        if is_optimistic_value(rv):
            csp.add_constraint(EqualOptimisticConstraint.from_bool(rv, True))
            try:
                if verbose:
                    print("  optimistic_search::final_csp_solve", *trajectory, sep='\n    ')
                assignments = simple_csp_solve(domain, csp, max_generator_trials=csp_max_generator_trials)
            except CSPNoGenerator:
                return None
            if assignments is not None:
                return assignments
            else:
                return None
        else:
            rv = float(rv)
            threshold = mpt_tracker.threshold if mpt_tracker is not None else 0.5
            if rv > threshold or mpt_tracker is not None:
                if rv <= threshold:
                    if not mpt_tracker.check(rv):
                        return None

                try:
                    assignments = simple_csp_solve(domain, csp, max_generator_trials=csp_max_generator_trials)
                except CSPNoGenerator:
                    raise None
                if assignments is not None:
                    plan = ground_actions(domain, trajectory, assignments)
                    if mpt_tracker is not None:
                        mpt_tracker.update(rv, plan)
                    if rv > threshold:
                        return ground_actions(domain, trajectory, assignments)
                else:
                    return None
            else:
                return None


def optimistic_search_domain_check(domain: Domain):
    for op in domain.operators.values():
        if op.is_axiom:
            raise NotImplementedError('Optimistic search does not support axioms.')


def prepare_optimistic_search(
    func,
    domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    actions=None, action_filter=None,
    verbose=False, forward_augmented=True, forward_derived=True
) -> Tuple[State, ValueOutputExpression, Sequence[OperatorApplier]]:
    state = domain.forward_features_and_axioms(state, forward_augmented, False, forward_derived)

    quantizer = ValueQuantizer(domain)
    if actions is None:
        actions = generate_all_partially_grounded_actions(domain, state, action_filter=action_filter)

    state = quantizer.quantize_state(state)
    goal_expr = domain.parse(goal_expr)

    if verbose:
        print(func.__name__ + '::initial_state', state)
        print(func.__name__ + '::actions nr', len(actions))
        print(func.__name__ + '::goal_expr', goal_expr)

    return state, goal_expr, actions


@jactorch.no_grad_func
def solve_optimistic_plan(domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression], actions: Sequence[OperatorApplier], csp_max_generator_trials=3):
    optimistic_search_domain_check(domain)
    quantizer = ValueQuantizer(domain)
    state = quantizer.quantize_state(state)
    csp = OptimisticValueContext()
    if isinstance(goal_expr, str):
        goal_expr = domain.parse(goal_expr)
    else:
        assert isinstance(goal_expr, ValueOutputExpression)

    action_groundings = list()
    for a in actions:
        action_grounding = instantiate_action(csp, a)
        action_groundings.append(action_grounding)
        (succ, state), csp = apply_action(domain, state, action_grounding, csp)
        if succ:
            pass
        else:
            raise ValueError(f'Unable to perform action {action_grounding} at state {state}.')

    plan = goal_test(domain, state, goal_expr, csp, trajectory=action_groundings, csp_max_generator_trials=csp_max_generator_trials)
    if plan is not None:
        return state, csp, plan
    return state, csp, None


@jactorch.no_grad_func
def optimistic_search(
    domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression],
    max_depth=5, use_tuple_desc=True, use_csp_pruning=True, verbose=False,
    actions=None, action_filter=None, forward_augmented=True, forward_derived=False,
    csp_max_generator_trials=3
):
    optimistic_search_domain_check(domain)
    state, goal_expr, actions = prepare_optimistic_search(
        optimistic_search, domain, state, goal_expr,
        actions=actions, action_filter=action_filter,
        verbose=verbose, forward_augmented=forward_augmented, forward_derived=forward_derived
    )

    assert not forward_derived, 'Not implemented.'

    states = [(state, tuple(), OptimisticValueContext())]
    visited = set()
    if use_tuple_desc:
        visited.add(state.generate_tuple_description(domain))

    for depth in range(max_depth):
        next_states = list()

        for s, traj, csp in states:
            for a in actions:
                ncsp = csp.clone()
                action_grounding = instantiate_action(ncsp, a)

                (succ, ns), ncsp = apply_action(domain, s, action_grounding, ncsp)
                nt = traj + (action_grounding, )

                if succ:
                    if use_csp_pruning:
                        try:
                            if not simple_csp_solve(domain, ncsp, solvable_only=True, max_generator_trials=csp_max_generator_trials):
                                continue
                        except CSPNoGenerator:
                            pass
                    if use_tuple_desc:
                        nst = ns.generate_tuple_description(domain)
                        if nst not in visited:
                            next_states.append((ns, nt, ncsp))
                            visited.add(nst)
                    else:  # unconditionally expand
                        next_states.append((ns, nt, ncsp))

                    plan = goal_test(domain, ns, goal_expr, ncsp, trajectory=nt, csp_max_generator_trials=csp_max_generator_trials, verbose=verbose)
                    if plan is not None:
                        return plan

        states = next_states

        if verbose:
            print(f'optimistic_search::depth={depth}, states={len(states)}')
    return None


class OptimisticSearchState(collections.namedtuple('_OptimisticSearchState', ['state', 'strips_state', 'trajectory', 'csp', 'g'])):
    pass


@jactorch.no_grad_func
def optimistic_search_with_heuristic(
    domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression],
    strips_heuristic: str = 'hff', *,
    max_expansions=100000, max_depth=100,  # search related parameters.
    heuristic_weight: float = 1, heuristic_use_forward_diff: bool = True,  # heuristic related parameters.
    use_tuple_desc=True, use_csp_pruning=True,  # pruning related parameters.
    actions=None, action_filter=None, forward_augmented=True, forward_derived=False,  # initialization related parameters.
    csp_max_generator_trials=3,  # csp solver related parameters.
    track_most_promising_trajectory=False, prob_goal_threshold=0.5,  # non-optimal trajectory tracking related parameters.
    verbose=False
):
    assert not forward_derived, 'Not implemented.'
    optimistic_search_domain_check(domain)
    state, goal_expr, actions = prepare_optimistic_search(
        optimistic_search, domain, state, goal_expr,
        actions=actions, action_filter=action_filter,
        verbose=verbose, forward_augmented=forward_augmented, forward_derived=forward_derived
    )

    from hacl.pdsketch.interface.v2.strips.heuristics import StripsHeuristic
    from hacl.pdsketch.interface.v2.strips.grounding import GStripsTranslator

    # TODO: Relevance analysis for optimistic planning tasks.
    strips_translator = GStripsTranslator(domain, use_string_name=verbose)
    strips_task = strips_translator.compile_task(
        state, goal_expr, actions,
        is_relaxed=False, forward_relevance_analysis=False, backward_relevance_analysis=False
    )
    heuristic = StripsHeuristic.from_type(strips_heuristic, strips_task, strips_translator, use_forward_diff=heuristic_use_forward_diff)

    mpt_tracker = None
    if track_most_promising_trajectory:
        mpt_tracker = MostPromisingTrajectoryTracker(True, prob_goal_threshold)

    initial_state = OptimisticSearchState(state, strips_task.state, tuple(), OptimisticValueContext(), 0)
    queue: List[QueueNode] = list()
    visited = set()

    def push_node(node: OptimisticSearchState):
        added = False
        if use_tuple_desc:
            nst = node.state.generate_tuple_description(domain)
            if nst not in visited:
                added = True
                visited.add(nst)
                hq.heappush(queue, QueueNode(heuristic.compute(node.strips_state) * heuristic_weight + node.g, node))
        else:  # unconditionally expand
            added = True
            hq.heappush(queue, QueueNode(heuristic.compute(node.strips_state) * heuristic_weight + node.g, node))

        if optimistic_search_with_heuristic.DEBUG and added:
            print('  opt::push_node:', *node.trajectory, sep='\n    ')
            print('   ', 'heuristic =', heuristic.compute(node.strips_state), 'g =', node.g)

    push_node(initial_state)
    nr_expansions = 0

    is_initial_state = True
    while len(queue) > 0 and nr_expansions < max_expansions:
        priority, node = hq.heappop(queue)
        nr_expansions += 1

        s, ss, traj, csp, _ = node
        if optimistic_search_with_heuristic.DEBUG:
            print('opt::pop_node:')
            print('  trajectory:', *traj, sep='\n  ')
            print('  priority =', priority, 'h =', priority - node.g, 'g =', node.g)
            input('  Continue?')

        if track_most_promising_trajectory and is_initial_state:
            is_initial_state = False
        else:
            plan = goal_test(
                domain, s, goal_expr, csp,
                trajectory=traj,
                csp_max_generator_trials=csp_max_generator_trials,
                verbose=verbose,
                mpt_tracker=mpt_tracker
            )
            if plan is not None:
                if verbose:
                    print('optimistic_search_with_heuristic::total_expansions:', nr_expansions)
                return plan

        if len(traj) >= max_depth:
            continue

        for sa in strips_task.operators:
            a = sa.raw_operator
            ncsp = csp.clone()
            action_grounding = instantiate_action(ncsp, a)

            (succ, ns), ncsp = apply_action(domain, s, action_grounding, ncsp)
            nt = traj + (action_grounding, )

            if succ:
                if use_csp_pruning:
                    try:
                        if optimistic_search_with_heuristic.DEBUG:
                            print('  opt::running CSP pruning...', *nt, sep='\n    ')
                        if not simple_csp_solve(domain, ncsp, solvable_only=True, max_generator_trials=csp_max_generator_trials):
                            if optimistic_search_with_heuristic.DEBUG:
                                print('  opt::pruned:', *nt, sep='\n    ')
                            continue
                    except CSPNoGenerator:
                        pass

                nss = sa.apply(ss)
                nnode = OptimisticSearchState(ns, nss, nt, ncsp, node.g + 1)
                push_node(nnode)

    if verbose:
        print('optimistic_search_with_heuristic::search failed.')
        print('optimistic_search_with_heuristic::total_expansions:', nr_expansions)

    if mpt_tracker is not None:
        return mpt_tracker.solution

    return None


optimistic_search_with_heuristic.DEBUG = False
optimistic_search_with_heuristic.set_debug = lambda x = True: setattr(optimistic_search_with_heuristic, 'DEBUG', x)
