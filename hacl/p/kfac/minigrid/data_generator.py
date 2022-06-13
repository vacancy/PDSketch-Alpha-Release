import time
import torch
import random
import hacl.pdsketch as pds

from copy import deepcopy
from hacl.envs.gridworld.minigrid.gym_minigrid.path_finding import find_path_to_obj

__all__ = ['MGState', 'OfflineDataGenerator', 'worker_offline', 'worker_search']


class MGState(object):
    def __init__(self, agent_pos, agent_dir, grid, carrying):
        self.agent_pos = agent_pos
        self.agent_dir = agent_dir
        self.grid = grid
        self.carrying = carrying

    @classmethod
    def save(cls, env):
        return cls(deepcopy(env.agent_pos), deepcopy(env.agent_dir), deepcopy(env.grid), deepcopy(env.carrying))

    def restore(self, env):
        env.agent_pos = self.agent_pos
        env.agent_dir = self.agent_dir
        env.agent_grid = self.grid
        env.carrying = self.carrying


class OfflineDataGenerator(object):
    def __init__(self, succ_prob):
        self.succ_prob = succ_prob

    def plan(self, env):
        if env.task in ('goto', 'goto2', 'gotosingle'):
            return self.plan_goto(env)
        elif env.task == 'pickup':
            return self.plan_pickup(env)
        else:
            raise NotImplementedError('Unknown task: {}.'.format(self.task))

    def plan_goto(self, env):
        goal_pos = (-1, -1)
        if random.random() < self.succ_prob:
            goal_obj = env.goal_obj
            for x, y, obj in env.iter_objects():
                if obj.color == goal_obj.color and obj.type == goal_obj.type:
                    goal_pos = (x, y)
                    break
        else:
            objects = list()
            for x, y, obj in env.iter_objects():
                if obj.type != 'wall':
                    objects.append(((x, y), obj))
            goal_pos, goal_obj = random.choice(objects)

        plan = find_path_to_obj(env, goal_pos)
        if plan is None:
            return None

        if env.task == 'goto2':
            plan.extend([env.Actions.forward, env.Actions.left, env.Actions.forward, env.Actions.left, env.Actions.forward])
        return plan

    def plan_pickup(self, env):
        plan = self.plan_goto(env)
        if plan is None:
            return None
        plan.append(env.Actions.pickup)
        return plan


def worker_offline(args, domain, env, action_filter):
    extra_monitors = dict()
    end = time.time()

    obs = env.reset()
    state, goal = obs['state'], obs['mission']
    data_gen = OfflineDataGenerator(0.5)
    plan = data_gen.plan(env)

    if plan is None:
        plan = list()

    states = [state]
    actions = []
    dones = [False]
    succ = False

    action_to_operator = {'left': 'lturn', 'right': 'rturn', 'forward': 'forward', 'pickup': 'pickup', 'toggle': 'toggle'}
    structure_mode = getattr(args, 'structure_mode', 'basic')
    for action in plan:
        if structure_mode in ('basic', 'robokin', 'abskin'):
            pddl_action = domain.operators[action_to_operator[action.name]]('r')
        elif structure_mode == 'full':
            if action.name == 'pickup':
                fwd_pos = env.front_pos
                fwd_cell = env.grid.get(*fwd_pos)
                assert fwd_cell is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', fwd_cell.name)
            else:
                pddl_action = domain.operators[action_to_operator[action.name]]('r')
        else:
            raise ValueError('Unknown structure mode: {}.'.format(structure_mode))

        rl_action = pds.rl.RLEnvAction(pddl_action.name)
        obs, reward, done, _ = env.step(rl_action)

        states.append(obs['state'])
        actions.append(pddl_action)
        dones.append(done)

        if done:
            succ = True
            break
    dones = torch.tensor(dones, dtype=torch.int64)

    extra_monitors['time/generate'] = time.time() - end
    data = (states, actions, dones, goal, succ, extra_monitors)
    return data


def worker_search(args, domain, env, action_filter):
    extra_monitors = dict()
    end = time.time()
    states, actions, dones, goal, succ = pds.rl.customize_follow_policy(
        domain, env,
        max_episode_length=10, max_expansions=200,
        action_filter=action_filter,
        search_algo=pds.heuristic_search_strips,
        extra_monitors=extra_monitors,
        # search-algo-arguments
        use_tuple_desc=False,
        use_quantized_state=False,
        prob_goal_threshold=0.5,
        strips_heuristic=args.heuristic,
        strips_backward_relevance_analysis=args.relevance_analysis,
    )
    extra_monitors['time/search'] = time.time() - end
    data = (states, actions, dones, goal, succ, extra_monitors)
    return data

