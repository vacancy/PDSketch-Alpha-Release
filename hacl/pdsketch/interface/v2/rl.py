import time
import torch
import jacinle

from typing import Optional, Union, Sequence
from .value import unwrap_value
from .optimistic import OptimisticValueContext, EqualOptimisticConstraint, is_optimistic_value
from .state import State
from .expr import Expression, ValueOutputExpression, ExpressionExecutionContext
from .domain import OperatorApplier, Domain, ValueQuantizer
from .planner import optimistic_planner as optplan
from .planner.basic_planner import generate_all_grounded_actions
from .planner.optimistic_planner import generate_all_partially_grounded_actions, optimistic_search_with_heuristic
from .csp_solver.simple_csp import simple_csp_solve

__all__ = ['RLEnvAction', 'TreeNode', 'simple_tree_search', 'simple_unroll_policy', 'optimistic_tree_search', 'optimistic_unroll_policy', 'customize_unroll_policy', 'customize_follow_policy']


class RLEnvAction(object):
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def __str__(self):
        return '{}({})'.format(self.name, ', '.join(str(arg) for arg in self.args))

    def __repr__(self):
        return str(self)


class TreeNode(object):
    def __init__(self, state, action=None, csp=None, **kwargs):
        self.state = state
        self.action = action
        self.csp = csp
        self.kwargs = kwargs
        self.children = list()

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def append(self, node):
        self.children.append(node)
        return node

    def format(self):
        if self.action is None:
            fmt = 'ROOT'
        else:
            fmt = '- ' + str(self.action)
        fmt += ': ' + ', '.join([f'{k}={v}' for k, v in self.kwargs.items()])
        fmt += '\n'
        for c in self.children:
            fmt += jacinle.indent_text(c.format().strip()) + '\n'
        return fmt

    def print(self):
        print(self.format())

    def compute(self):
        for c in self.children:
            c.compute()
        p = self.kwargs['goal_prob']
        if self.is_leaf:
            v = (1 - p) * (-1)
        else:
            v = p * 0 + (1 - p) * (-1 + torch.max(
                torch.tensor([c.kwargs['value'] for c in self.children])
            ))
        self.kwargs['value'] = v

        if self.is_leaf:
            self.kwargs['action'] = None
        else:
            maxc = max(self.children, key=lambda c: c.kwargs['value'])
            self.kwargs['action'] = maxc.action

            if 'assignments' in maxc.kwargs:
                self.kwargs['assignments'] = maxc.kwargs['assignments']


def simple_tree_search(domain: Domain, state: State, goal: Expression, actions: Optional[Sequence[OperatorApplier]] = None, max_depth=2) -> OperatorApplier:
    if actions is None:
        actions = generate_all_grounded_actions(domain, state)
    root = TreeNode(state, action=None, goal_prob=domain.eval(goal, state).tensor)

    def dfs(node: TreeNode, depth: int):
        for a in actions:
            succ, new_state = a(node.state)
            if succ:
                goal_prob = domain.eval(goal, new_state)
                new_node = node.append(TreeNode(new_state, a, goal_prob=goal_prob.tensor.item()))
                if depth + 1 < max_depth:
                    dfs(new_node, depth + 1)

    dfs(root, 0)
    root.compute()
    # root.print()
    return root.kwargs['action']


def simple_unroll_policy(domain: Domain, env, max_episode_length):
    obs = env.reset()
    state = obs['state']
    goal = obs['mission']
    all_actions = generate_all_grounded_actions(domain, state)

    states = [state]
    actions = list()
    dones = [False]
    with torch.no_grad():
        for i in range(max_episode_length):
            action = simple_tree_search(domain, state, goal, all_actions, max_episode_length - i)
            env_action = RLEnvAction(action.name, *[unwrap_value(v) for v in action.arguments])
            obs, reward, done, _ = env.step(env_action)
            state = obs['state']
            states.append(state)
            actions.append(action)
            dones.append(done)
            if done:
                break

    return states, actions, torch.tensor(dones, dtype=torch.int64), goal


def _goal_test(domain, state, goal_expr, csp: OptimisticValueContext):
    csp = csp.clone()
    ctx = ExpressionExecutionContext(domain, state, {}, optimistic_context=csp)
    with ctx.as_default():
        rv = goal_expr.forward(ctx).item()
        if is_optimistic_value(rv):
            raise NotImplementedError()
        else:
            assignments = simple_csp_solve(domain, csp)
            if assignments is not None:
                return rv, assignments
            else:
                return rv, None


def optimistic_tree_search(
    domain: Domain, state: State, goal_expr: Union[str, ValueOutputExpression],
    max_depth=5, actions=None, use_tuple_desc=True
):
    optplan.optimistic_search_domain_check(domain)

    quantizer = ValueQuantizer(domain)
    if actions is None:
        actions = generate_all_partially_grounded_actions(domain, state)
    state = quantizer.quantize_state(state)

    if isinstance(goal_expr, str):
        goal_expr = domain.parse(goal_expr)
    else:
        assert isinstance(goal_expr, ValueOutputExpression)

    root = node = TreeNode(state, None, OptimisticValueContext())
    rv, assignments = _goal_test(domain, state, goal_expr, node.csp)
    node.kwargs['goal_prob'] = rv
    node.kwargs['assignments'] = assignments

    states = [node]
    visited = set()
    if use_tuple_desc:
        visited.add(state.generate_tuple_description(domain))

    for depth in range(max_depth):
        next_states = list()
        for node in states:
            for a in actions:
                ncsp = node.csp.clone()
                action_grounding = optplan.instantiate_action(ncsp, a)
                (succ, ns), ncsp = optplan.apply_action(domain, node.state, action_grounding, ncsp)
                if succ:
                    if use_tuple_desc:
                        nst = ns.generate_tuple_description(domain)
                        if nst not in visited:
                            visited.add(nst)
                        else:
                            continue

                    nnode = TreeNode(ns, action_grounding, ncsp)
                    rv, assignments = _goal_test(domain, ns, goal_expr, ncsp)
                    if assignments is not None:
                        nnode.kwargs['goal_prob'] = rv
                        nnode.kwargs['assignments'] = assignments
                    else:
                        continue

                    next_states.append(nnode)
                    node.append(nnode)

        states = next_states

    root.compute()
    return optplan.ground_actions(domain, [root.kwargs['action']], root.kwargs['assignments'])[0]


def optimistic_unroll_policy(domain: Domain, env, max_episode_length, all_actions=None, action_filter=None):
    obs = env.reset()
    state = obs['state']
    goal = obs['mission']

    if all_actions is None:
        all_actions = generate_all_partially_grounded_actions(domain, state)
        if action_filter is not None:
            all_actions = list(filter(action_filter, all_actions))

    states = [state]
    actions = list()
    dones = [False]
    with torch.no_grad():
        for i in range(max_episode_length):
            action = optimistic_tree_search(domain, state, goal, max_depth=max_episode_length - i, actions=all_actions)
            env_action = RLEnvAction(action.name, *[unwrap_value(v) for v in action.arguments])
            obs, reward, done, _ = env.step(env_action)
            state = obs['state']
            states.append(state)
            actions.append(action)
            dones.append(done)
            if done:
                break

    return states, actions, torch.tensor(dones, dtype=torch.int64), goal


def customize_unroll_policy(
    domain: Domain,
    env,
    max_episode_length,
    actions=None, action_filter=None,
    search_algo=optimistic_search_with_heuristic,
    max_expansions=500, **kwargs
):
    obs = env.reset()
    state = obs['state']
    goal = obs['mission']
    state = domain.forward_features_and_axioms(state, forward_augmented=True, forward_axioms=False, forward_derived=False)

    if actions is None:
        actions = generate_all_partially_grounded_actions(domain, state, action_filter=action_filter)

    print('unroll_policy::goal =', goal)

    t_states = [state]
    t_actions = list()
    t_dones = [False]
    with torch.no_grad():
        for i in range(max_episode_length):
            plan = search_algo(
                domain, state, goal,
                max_depth=max_episode_length - i,
                max_expansions=max_expansions,
                actions=actions,
                forward_augmented=False, forward_derived=False,
                track_most_promising_trajectory=True,
                **kwargs
            )
            env.debug_print()
            print('unroll_policy::step =', i, plan)
            if plan is None or len(plan) == 0:
                raise RuntimeError('No plan found.')
            else:
                action = plan[0]

            env_action = RLEnvAction(action.name, *[unwrap_value(v) for v in action.arguments])
            obs, reward, done, _ = env.step(env_action)
            state = obs['state']
            state = domain.forward_features_and_axioms(
                state, forward_augmented=True, forward_axioms=False,
                forward_derived=False
            )
            t_states.append(state)
            t_actions.append(action)
            t_dones.append(done)
            if done:
                break

    print('unroll_policy::output =', t_actions)
    return t_states, t_actions, torch.tensor(t_dones, dtype=torch.int64), goal


def customize_follow_policy(
    domain: Domain,
    env,
    max_episode_length,
    actions=None, action_filter=None,
    search_algo=optimistic_search_with_heuristic,
    max_expansions=500, extra_monitors=None, **kwargs
):
    end_time = time.time()

    obs = env.reset()
    state = obs['state']
    goal = obs['mission']
    state = domain.forward_features_and_axioms(state, forward_augmented=True, forward_axioms=False, forward_derived=False)

    if actions is None:
        actions = generate_all_partially_grounded_actions(domain, state, action_filter=action_filter)

    if extra_monitors is not None:
        extra_monitors['time/search/init'] = time.time() - end_time
        end_time = time.time()

    with torch.no_grad():
        plan = search_algo(
            domain, state, goal,
            max_depth=max_episode_length,
            max_expansions=max_expansions,
            actions=actions,
            forward_augmented=False, forward_derived=False,
            track_most_promising_trajectory=True,
            **kwargs
        )

    if extra_monitors is not None:
        extra_monitors['time/search/core'] = time.time() - end_time
        end_time = time.time()

    if plan is None:
        raise RuntimeError('No plan found.')

    # print('follow_policy::output =', plan)

    t_states = [state]
    t_actions = list()
    t_dones = [False]
    succ = False
    for action in plan:
        env_action = RLEnvAction(action.name, *[unwrap_value(v) for v in action.arguments])
        obs, reward, done, _ = env.step(env_action)
        state = obs['state']
        state = domain.forward_features_and_axioms(
            state, forward_augmented=True, forward_axioms=False, forward_derived=False
        )
        t_states.append(state)
        t_actions.append(action)
        t_dones.append(done)

        # def str_obj(o):
        #     if isinstance(o, tuple):
        #         o = o[2]
        #     return f'({o.type} {o.color} {o.is_open})'
        # print('follow_policy::step =', action, done, env.compute_done(), str_obj(env.goal_obj), list(str_obj(o) for o in env.grid.iter_objects() if o[2].type == 'door'))

        if done:
            succ = True
            break

    # print('succ', succ)

    if extra_monitors is not None:
        extra_monitors['time/search/simulate'] = time.time() - end_time
        end_time = time.time()

    return t_states, t_actions, torch.tensor(t_dones, dtype=torch.int64), goal, succ
