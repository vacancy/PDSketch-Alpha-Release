import jacinle
import jactorch
from typing import Callable
from .strips_expr import StripsState
from .grounding import GStripsTask
from .heuristics import StripsHeuristic

__all__ = ['strips_brute_force_search', 'strips_heuristic_search', 'get_priority_func']


def print_task(task: GStripsTask, func):
    func_name = func.__name__

    print(f'{func_name}::task.goal={task.goal}')
    print(f'{func_name}::task.facts={len(task.facts) if task.facts is not None else "N/A"}')
    print(f'{func_name}::task.operators={len(task.operators)}')


@jactorch.no_grad_func
def strips_brute_force_search(
    task: GStripsTask, max_depth=5, verbose=True
):
    if verbose:
        print_task(task, strips_brute_force_search)

    goal_func = task.goal.compile()
    states = [(task.state, tuple())]
    visited = set()
    visited.add(task.state)

    pbar = None
    if verbose:
        pbar = jacinle.tqdm_pbar(desc='strips_brute_force_search::depth=0')
    with jacinle.cond_with(pbar, verbose):
        for depth in range(max_depth):
            next_states = list()
            for s, traj in states:
                for a in task.operators:
                    if verbose:
                        pbar.update()
                    if a.applicable(s):
                        ns = a.apply(s)

                        if ns not in visited:
                            visited.add(ns)
                        nt = traj + (a.raw_operator,)
                        next_states.append((ns, nt))
                        if goal_func(ns):
                            return nt
            states = next_states
            if verbose:
                pbar.set_description(f'strips_brute_force_search::depth={depth}, states={len(states)}')
    return None


def get_priority_func(heuristic: StripsHeuristic, weight: float) -> Callable[[StripsState, int], float]:
    if weight == 1:
        def priority_fun(state, g, heuristic=heuristic):
            return heuristic.compute(state) + g
    elif weight == float('inf'):
        def priority_fun(state, g, heuristic=heuristic):
            return heuristic.compute(state)
    else:
        def priority_fun(state, g, heuristic=heuristic, weight=weight):
            return heuristic.compute(state) + g * weight
    return priority_fun


@jactorch.no_grad_func
def strips_heuristic_search(
    task: GStripsTask, heuristic: StripsHeuristic, *,
    max_expansions=int(1e9), verbose=False,
    heuristic_weight=float('inf')
):
    if verbose:
        print_task(task, strips_heuristic_search)
        print('strips_heuristic_search::init_heuristic={}'.format(heuristic.compute(task.state)))

    goal_func = task.goal.compile()

    pbar = None
    if verbose:
        pbar = jacinle.tqdm_pbar(desc='strips_heuristic_search::expanding')

    def check_goal(state: StripsState, gf=goal_func):
        return gf(state)

    def get_successors(state: GStripsTask, actions=task.operators):
        for action in actions:
            if verbose:
                pbar.update()
            if action.applicable(state):
                yield action, action.apply(state), 1

    with jacinle.cond_with(pbar, verbose):
        from hacl.algorithms.poc.heuristic_search import run_heuristic_search
        return run_heuristic_search(
            task.state,
            check_goal,
            get_priority_func(heuristic, heuristic_weight),
            get_successors,
            max_expansions=max_expansions
        )[1]
