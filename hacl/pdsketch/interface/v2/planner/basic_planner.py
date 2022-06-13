import warnings
import jacinle
import jactorch
import torch
import itertools
import collections
import heapq as hq
from typing import Any, Optional, Union, Tuple, Sequence, Mapping, List, Dict, Callable

from hacl.pdsketch.interface.v2.value import wrap_value
from hacl.pdsketch.interface.v2.state import State, ObjectType, NamedValueType
from hacl.pdsketch.interface.v2.expr import ExpressionExecutionContext, ValueOutputExpression, is_simple_bool, get_simple_bool_def
from hacl.pdsketch.interface.v2.domain import Domain, OperatorApplier, ValueQuantizer
from hacl.algorithms.poc.heuristic_search import QueueNode
from .prob_goal import MostPromisingTrajectoryTracker

__all__ = [
    'generate_continuous_values', 'expand_continuous_values',
    'generate_all_grounded_actions', 'filter_static_grounding',
    'unquantize_actions', 'apply_action', 'prepare_search',
    'HeuristicSearchState',
    'validate_plan', 'brute_force_search', 'heuristic_search', 'heuristic_search_strips'
]


def generate_continuous_values(domain: Domain, state: State, nr_iterations: Optional[int] = 1, nr_samples: Optional[int] = 5) -> Mapping[str, Sequence[torch.Tensor]]:
    continuous_values = dict()
    for type_def in domain.types.values():
        if isinstance(type_def, NamedValueType):
            continuous_values[type_def.typename] = list()

    for key, value in domain.features.items():
        if key in state.features.all_feature_names and isinstance(value.output_type, NamedValueType):
            type_def = value.output_type
            feat = state.features[key].tensor
            feat = feat.reshape((-1, ) + feat.shape[-type_def.ndim():])
            continuous_values[type_def.typename].extend([wrap_value(x, type_def) for x in feat])

    for i in range(nr_iterations):
        expand_continuous_values(domain, continuous_values, nr_samples=nr_samples)
    return continuous_values


def expand_continuous_values(domain: Domain, current: Dict[str, List[Any]], nr_samples: Optional[int] = 5):
    for gen_name, gen_def in domain.generators.items():
        arguments = list()
        for arg in gen_def.context:
            assert isinstance(arg.output_type, NamedValueType)
            arguments.append(current[arg.output_type.typename])
        for comb in itertools.product(*arguments):
            for i in range(nr_samples):
                outputs = domain.get_external_function(f'generator::{gen_name}')(*comb)
                for output, output_def in zip(outputs, gen_def.generates):
                    assert isinstance(output_def.output_type, NamedValueType)
                    current[output_def.output_type.typename].append(wrap_value(output, output_def.output_type))


def generate_all_grounded_actions(domain: Domain, state: State, continuous_values=None, action_filter: Optional[Callable[[OperatorApplier], bool]] = None, filter_static: Optional[bool] = True) -> List[OperatorApplier]:
    actions = list()
    for op in domain.operators.values():
        argument_candidates = list()
        for arg in op.arguments:
            if isinstance(arg.type, ObjectType):
                argument_candidates.append(state.object_type2name[arg.type.typename])
            else:
                assert isinstance(arg.type, NamedValueType)
                argument_candidates.append(continuous_values[arg.type.typename])
        for comb in itertools.product(*argument_candidates):
            actions.append(op(*comb))
    if filter_static:
        actions = filter_static_grounding(domain, state, actions)
    if action_filter is not None:
        actions = list(filter(action_filter, actions))
    return actions


def filter_static_grounding(domain, state, actions):
    output_actions = list()
    for action in actions:
        ctx = ExpressionExecutionContext(domain, state, bounded_variables=state.compose_bounded_variables(action.operator.arguments, action.arguments))
        flag = True
        with ctx.as_default():
            for pre in action.operator.preconditions:
                if is_simple_bool(pre.bool_expr) and get_simple_bool_def(pre.bool_expr).static:
                    rv = pre.bool_expr.forward(ctx).item()
                    if rv < 0.5:
                        flag = False
                        break
        if flag:
            output_actions.append(action)
    return output_actions


def unquantize_actions(domain: Domain, actions: Sequence[OperatorApplier]):
    output = list()
    for action in actions:
        arguments = list()
        for arg_def, arg in zip(action.operator.arguments, action.arguments):
            if isinstance(arg_def.type, ObjectType):
                arguments.append(arg)
            else:
                assert isinstance(arg_def.type, NamedValueType)
                if arg.quantized:
                    arguments.append(domain.value_quantizer.unquantize(arg_def.type.typename, arg.item()).tensor)
                else:
                    arguments.append(arg)
        output.append(action.operator(*arguments))
    return output


def apply_action(domain: Domain, state: State, action: OperatorApplier, forward_derived=True):
    succ, ns = action(state)
    if succ:
        ns = domain.forward_features_and_axioms(ns, False, True, forward_derived)
    return succ, ns


def goal_test(
    domain: Domain, state: State, goal_expr: ValueOutputExpression,
    trajectory, mpt_tracker: Optional[MostPromisingTrajectoryTracker] = None, verbose=False
):
    score = domain.eval(goal_expr, state).item()

    threshold = 0.5
    if mpt_tracker is not None:
        if mpt_tracker.check(score):
            mpt_tracker.update(score, trajectory)
        threshold = mpt_tracker.threshold

    return score > threshold


def prepare_search(
    func,
    domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    use_quantized_state: bool = True,
    actions=None, continuous_values=None, action_filter=None, forward_augmented=True, forward_derived=True,
    verbose=False
) -> Tuple[State, ValueOutputExpression, Sequence[OperatorApplier]]:
    state = domain.forward_features_and_axioms(state, forward_augmented, False, forward_derived)

    quantizer = ValueQuantizer(domain)
    if actions is None:
        if continuous_values is not None and use_quantized_state:
            continuous_values = quantizer.quantize_dict_list(continuous_values)
        actions = generate_all_grounded_actions(domain, state, continuous_values, action_filter=action_filter)
    if use_quantized_state:
        state = quantizer.quantize_state(state)
    goal_expr = domain.parse(goal_expr)

    if verbose:
        # print(func.__name__ + '::initial_state', state)
        print(func.__name__ + '::actions nr', len(actions))
        print(func.__name__ + '::goal_expr', goal_expr)

    return state, goal_expr, actions


@jactorch.no_grad_func
def validate_plan(
    domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression], actions: Sequence[OperatorApplier],
    use_quantized_state: bool = True,
    forward_augmented=True, forward_derived=True,
):
    state = domain.forward_features_and_axioms(state, forward_augmented, False, forward_derived)

    quantizer = ValueQuantizer(domain)
    if use_quantized_state:
        state = quantizer.quantize_state(state)

    if isinstance(goal_expr, str):
        goal_expr = domain.parse(goal_expr)
    else:
        assert isinstance(goal_expr, ValueOutputExpression)

    for action in actions:
        succ, state = apply_action(domain, state, action, forward_derived=forward_derived)
        assert succ

    score = domain.eval(goal_expr, state)
    return state, score


@jactorch.no_grad_func
def brute_force_search(
    domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression], *,
    max_depth=5,
    use_tuple_desc=True, use_quantized_state: bool = True,
    actions=None, continuous_values=None, action_filter=None, forward_augmented=True, forward_derived=True,
    verbose=False
):
    if not use_quantized_state:
        assert not use_tuple_desc, 'Tuple desc cannot be used without quantized states.'
    state, goal_expr, actions = prepare_search(
        brute_force_search, domain, state, goal_expr,
        actions=actions, continuous_values=continuous_values, action_filter=action_filter,
        verbose=verbose, forward_augmented=forward_augmented, forward_derived=forward_derived,
        use_quantized_state=use_quantized_state
    )

    states = [(state, tuple())]
    visited = set()
    if use_tuple_desc:
        visited.add(state.generate_tuple_description(domain))

    pbar = None
    if verbose:
        pbar = jacinle.tqdm_pbar(desc='brute_force_search::depth=0')
    with jacinle.cond_with(pbar, verbose):
        for depth in range(max_depth):
            next_states = list()

            for s, traj in states:
                # print(traj, domain.eval(goal_expr, s))
                if goal_test(domain, s, goal_expr, traj):
                    return unquantize_actions(domain, traj)

                for a in actions:
                    succ, ns = apply_action(domain, s, a, forward_derived=forward_derived)
                    if verbose:
                        pbar.update()
                    nt = traj + (a, )
                    if succ:
                        if use_tuple_desc:
                            nst = ns.generate_tuple_description(domain)
                            if nst not in visited:
                                next_states.append((ns, nt))
                                visited.add(nst)
                        else:  # unconditionally expand
                            next_states.append((ns, nt))

            states = next_states

            if verbose:
                pbar.set_description(f'brute_force_search::depth={depth}, states={len(states)}')
    return None


@jactorch.no_grad_func
def heuristic_search(
    domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression],
    heuristic, *,
    max_expansions=10000, max_depth=1000,
    use_tuple_desc=True, use_quantized_state: bool = True,
    actions=None, continuous_values=None, action_filter=None, forward_augmented=False, forward_derived=False,
    verbose=False
):
    warnings.warn('heuristic_search is deprecated. Use heuristic_search_strips instead.')
    assert use_tuple_desc and use_quantized_state, 'Tuple desc and quantized states are required.'
    assert not forward_derived, 'Not implemented.'

    state, goal_expr, actions = prepare_search(
        heuristic_search, domain, state, goal_expr,
        use_quantized_state=use_quantized_state,
        actions=actions, continuous_values=continuous_values, action_filter=action_filter,
        forward_augmented=forward_augmented, forward_derived=forward_derived,
        verbose=verbose
    )

    visited = dict()
    idx2state = list()

    pbar = None
    if verbose:
        pbar = jacinle.tqdm_pbar(desc='heuristic_search::expanding')

    def check_goal(state: int):
        state = idx2state[state]
        return goal_test(domain, state, goal_expr, None)

    def get_priority(state: int, g: int):  # gbf search heuristic.
        state = idx2state[state]
        return heuristic.compute(state, goal_expr, actions)

    def get_successors(state: int):
        state = idx2state[state]
        for action in actions:
            if verbose:
                pbar.update()
            succ, ns = apply_action(domain, state, action, forward_derived)
            if succ:
                td = ns.generate_tuple_description(domain)
                if td not in visited:
                    visited[td] = len(visited)
                    idx2state.append(ns)
                yield action, len(visited) - 1, 1

    td = state.generate_tuple_description(domain)
    visited[td] = 0
    idx2state.append(state)

    with jacinle.cond_with(pbar, verbose):
        from hacl.algorithms.poc.heuristic_search import run_heuristic_search
        return run_heuristic_search(
            0,
            check_goal,
            get_priority,
            get_successors,
            check_visited=False,
            max_expansions=max_expansions,
            max_depth=max_depth
        )[1]


class HeuristicSearchState(collections.namedtuple('_HeuristicSearchState', ['state', 'strips_state', 'trajectory', 'g'])):
    pass


@jactorch.no_grad_func
def heuristic_search_strips(
    domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression],
    strips_heuristic: str = 'hff', *,
    max_expansions=100000, max_depth=100,  # search related parameters.
    heuristic_weight: float = 1,  # heuristic related parameters.
    external_heuristic_function = None,  # external heuristic related parameters.
    strips_forward_relevance_analysis: bool = False, strips_backward_relevance_analysis: bool = True,
    strips_use_sas: bool = False,  # whether to use SAS Strips compiler (AODiscretization)
    use_strips_op: bool = False,
    use_tuple_desc: bool = True, use_quantized_state: bool = True,  # pruning related parameters.
    actions=None, action_filter=None, continuous_values=None, forward_augmented=True, forward_derived=False,  # initialization related parameters.
    track_most_promising_trajectory=False, prob_goal_threshold=0.5,  # non-optimal trajectory tracking related parameters.
    verbose=False, return_extra_info=False
):
    if not use_quantized_state:
        assert not use_tuple_desc, 'Tuple desc cannot be used without quantized states.'

    state, goal_expr, actions = prepare_search(
        heuristic_search, domain, state, goal_expr,
        use_quantized_state=use_quantized_state,
        actions=actions, action_filter=action_filter,
        continuous_values=continuous_values,
        forward_augmented=forward_augmented, forward_derived=forward_derived,
        verbose=verbose
    )

    from hacl.pdsketch.interface.v2.strips.heuristics import StripsHeuristic
    from hacl.pdsketch.interface.v2.strips.grounding import GStripsTranslator
    from hacl.pdsketch.interface.v2.strips.grounding import GStripsTranslatorSAS

    # import ipdb; ipdb.set_trace()
    if strips_use_sas:
        strips_translator = GStripsTranslatorSAS(domain, use_string_name=True, prob_goal_threshold=prob_goal_threshold, cache_bool_features=True)
    else:
        strips_translator = GStripsTranslator(domain, use_string_name=True, prob_goal_threshold=prob_goal_threshold)
    strips_task = strips_translator.compile_task(
        state, goal_expr, actions,
        is_relaxed=False,
        forward_relevance_analysis=strips_forward_relevance_analysis,
        backward_relevance_analysis=strips_backward_relevance_analysis,
    )

    if strips_heuristic == 'external' and external_heuristic_function is not None:
        pass
    else:
        heuristic = StripsHeuristic.from_type(
            strips_heuristic, strips_task, strips_translator,
            forward_relevance_analysis=strips_forward_relevance_analysis,
            backward_relevance_analysis=strips_backward_relevance_analysis,
        )

    # from IPython import embed; embed()
    # import ipdb; ipdb.set_trace()

    # print(strips_task.goal)
    # print(strips_task.operators)
    # print(heuristic.relaxed.goal)
    # print(heuristic.relaxed.operators)

    mpt_tracker = None
    if track_most_promising_trajectory:
        mpt_tracker = MostPromisingTrajectoryTracker(True, prob_goal_threshold)

    initial_state = HeuristicSearchState(state, strips_task.state, tuple(), 0)
    queue: List[QueueNode] = list()
    visited = set()

    def push_node(node: HeuristicSearchState):
        added = False
        if use_tuple_desc:
            nst = node.state.generate_tuple_description(domain)
            if nst not in visited:
                added = True
                visited.add(nst)
        else:  # unconditionally expand
            added = True

        if added:
            if strips_heuristic == 'external' and external_heuristic_function is not None:
                hq.heappush(queue, QueueNode(external_heuristic_function(node.state, goal_expr) * heuristic_weight + node.g, node))
            else:
                hq.heappush(queue, QueueNode(heuristic.compute(node.strips_state) * heuristic_weight + node.g, node))
                if heuristic_search_strips.DEBUG and added:
                    print('  hsstrips::push_node:', *node.trajectory, sep='\n    ')
                    print('   ', 'heuristic =', heuristic.compute(node.strips_state), 'g =', node.g)

        # if added:
        #     print('  hsstrips::push_node:', *node.trajectory)
        #     print('   ', 'heuristic =', heuristic.compute(node.strips_state), 'g =', node.g)

    push_node(initial_state)
    nr_expansions = 0

    def wrap_extra_info(traj):
        if return_extra_info:
            return traj, {'nr_expansions': nr_expansions}
        return traj

    pbar = None
    if verbose:
        pbar = jacinle.tqdm_pbar(desc='heuristic_search::expanding')

    while len(queue) > 0 and nr_expansions < max_expansions:
        priority, node = hq.heappop(queue)
        nr_expansions += 1

        """
        Name convention:
            - node: current node.
            - nnode: next node.
            - s: the state of the search tree.
            - ns: the state of the search tree after the action is applied.
            - a: the action applied.
            - ss: the strips state of the search tree.
            - nss: the strips state of the search tree after the action is applied.
            - traj: the path from the root to the node in the search tree.
            - nt: the path in the search tree after the action is applied.
            - g: the cost of the path from the root to the node in the search tree.
        """

        s, ss, traj, _ = node
        if heuristic_search_strips.DEBUG:
            print('hsstrips::pop_node:')
            print('  trajectory:', *traj, sep='\n  ')
            print('  priority =', priority, 'h =', priority - node.g, 'g =', node.g)
            input('  Continue?')
        if verbose:
            pbar.set_description(f'heuristic_search::expanding: priority = {priority} g = {node.g}')
            pbar.update()
        # print('hsstrips::pop_node:', *traj)
        # print('  priority =', priority, 'h =', priority - node.g, 'g =', node.g)

        goal_reached = goal_test(
            domain, s, goal_expr,
            trajectory=traj,
            verbose=verbose,
            mpt_tracker=mpt_tracker
        )
        if goal_reached:
            if verbose:
                print('hsstrips::total_expansions:', nr_expansions)
            return wrap_extra_info(traj)

        if len(traj) >= max_depth:
            continue

        for sa in strips_task.operators:
            a = sa.raw_operator
            succ, ns = apply_action(domain, s, a)
            nt = traj + (a, )

            if succ:
                nss = sa.apply(ss) if use_strips_op else strips_translator.compile_state(ns.clone(), forward_derived=False)
                nnode = HeuristicSearchState(ns, nss, nt, node.g + 1)
                push_node(nnode)

    if verbose:
        print('hsstrips::search failed.')
        print('hsstrips::total_expansions:', nr_expansions)

    if mpt_tracker is not None:
        return wrap_extra_info(mpt_tracker.solution)

    return wrap_extra_info(None)


heuristic_search_strips.DEBUG = False
heuristic_search_strips.set_debug = lambda x = True: setattr(heuristic_search_strips, 'DEBUG', x)
