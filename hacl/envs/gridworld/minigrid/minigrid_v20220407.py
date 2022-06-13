import os.path as osp
import numpy as np
from typing import Optional
import time
import torch
import jactorch
import hacl.pdsketch as pds
import hacl.pdsketch.interface.v2.rl as pdsrl
import hacl.envs.gridworld.minigrid.gym_minigrid as minigrid
from .gym_minigrid.path_finding import find_path_to_obj

__all__  = [
    'get_domain', 'set_domain_mode',
    'SUPPORTED_ACTION_MODES', 'SUPPORTED_STRUCTURE_MODES', 'SUPPORTED_TASKS',
    'MiniGridEnvV20220407', 'MiniGridStateV20220407', 'make',
    'visualize_planner', 'visualize_plan'
]


def _map_int(x):
    if isinstance(x, tuple):
        return map(int, x)
    if isinstance(x, np.ndarray):
        return map(int, x)
    if isinstance(x, torch.Tensor):
        return map(int, jactorch.as_numpy(x))


g_domain_action_mode = 'absaction'
g_domain_structure_mode = 'full'
g_domain: Optional[pds.Domain] = None

SUPPORTED_ACTION_MODES = ['absaction', 'envaction']
SUPPORTED_STRUCTURE_MODES = ['basic', 'abskin', 'robokin', 'full']
SUPPORTED_TASKS = ['gotosingle', 'goto', 'goto2', 'pickup', 'open', 'generalization']


def set_domain_mode(action_mode, structure_mode):
    global g_domain
    assert g_domain is None, 'Domain has been loaded.'
    assert action_mode in SUPPORTED_ACTION_MODES, f'Unsupported action mode: {action_mode}.'
    assert structure_mode in SUPPORTED_STRUCTURE_MODES, f'Unsupported structure mode: {structure_mode}.'

    global g_domain_action_mode
    global g_domain_structure_mode

    g_domain_action_mode = action_mode
    g_domain_structure_mode = structure_mode


def get_domain(force_reload=False):
    global g_domain
    if g_domain is None or force_reload:
        g_domain = pds.load_domain_file(osp.join(
            osp.dirname(__file__),
            'pds_files',
            f'minigrid-v20220407-{g_domain_action_mode}-{g_domain_structure_mode}.pdsketch'
        ))
    return g_domain


class MiniGridEnvV20220407(minigrid.MiniGridEnv):
    def __init__(self, task='pickup'):
        assert task in SUPPORTED_TASKS, f'Unknown task: {task}.'
        self.task = task
        self.options = dict()
        super().__init__(grid_size=7, max_steps=64, seed=1337, require_obs=False)

    def set_options(self, **kwargs):
        self.options.update(kwargs)

    def get_option(self, name, default=None):
        return self.options.get(name, default)

    def _gen_grid(self, width, height):
        if self.task == 'gotosingle':
            self._gen_grid_goto_single(width, height)
        elif self.task in ('goto', 'goto2'):
            self._gen_grid_goto(width, height)
        elif self.task == 'pickup':
            self._gen_grid_pickup(width, height)
        elif self.task == 'open':
            self._gen_grid_open(width, height)
        elif self.task == 'generalization':
            self._gen_grid_generalization(width, height)
        else:
            raise ValueError(f'Unknown task: {self.task}.')

    def _gen_grid_goto_single(self, width, height):
        self.grid = minigrid.Grid(width, height)
        self.agent_pos = (3, 3)
        self.agent_dir = 0
        self.grid.horz_wall(0, 0, 7)
        self.grid.horz_wall(0, 6, 7)
        self.grid.vert_wall(0, 0, 7)
        self.grid.vert_wall(6, 0, 7)

        objects = list()
        for i in range(1):
            shape = np.random.choice([minigrid.Key, minigrid.Box, minigrid.Ball])
            color = np.random.choice(minigrid.COLOR_NAMES)

            while True:
                pose = np.random.randint(1, 6, size=2)
                if self.grid.get(*pose) is None and not np.all(pose == 3) and not np.all(pose == (4, 3)):  # not initially facing.
                    break

            object = shape(color)
            objects.append(object)
            self.grid.set(*pose, object)

        self.goal_obj = goal = np.random.choice(objects)
        self.mission = get_domain().parse(f'(exists (?o - item) (and (robot-is-facing r ?o) (is-{goal.type} ?o)))')

    def _gen_grid_goto(self, width, height):
        for _ in range(self.get_option('max_trials', 100)):
            self.grid = minigrid.Grid(width, height)
            self.agent_pos = (3, 3)
            self.agent_dir = 0
            self.grid.horz_wall(0, 0, 7)
            self.grid.horz_wall(0, 6, 7)
            self.grid.vert_wall(0, 0, 7)
            self.grid.vert_wall(6, 0, 7)

            objects = list()
            object_poses = list()
            for i in range(self.get_option('nr_objects', 4)):
                shape = np.random.choice([minigrid.Key, minigrid.Box, minigrid.Ball])
                color = np.random.choice(minigrid.COLOR_NAMES)

                while True:
                    pose = np.random.randint(1, 6, size=2)
                    if self.grid.get(*pose) is None and not np.all(pose == 3) and not np.all(pose == (4, 3)):  # not initially facing.
                        break

                object = shape(color)
                objects.append(object)
                object_poses.append(pose)
                self.grid.set(*pose, object)

            self.goal_obj = goal = np.random.choice(objects)
            self.goal_pose = object_poses[objects.index(goal)]
            self.mission = get_domain().parse(f'(exists (?o - item) (and (robot-is-facing r ?o) (is-{goal.type} ?o) (is-{goal.color} ?o)))')

            path = find_path_to_obj(self, tuple(self.goal_pose))
            if path is not None:
                break

    def _gen_grid_pickup(self, width, height):
        for _ in range(self.get_option('max_trials', 100)):
            self.grid = minigrid.Grid(width, height)
            self.agent_pos = (3, 3)
            self.agent_dir = 0
            self.grid.horz_wall(0, 0, 7)
            self.grid.horz_wall(0, 6, 7)
            self.grid.vert_wall(0, 0, 7)
            self.grid.vert_wall(6, 0, 7)

            objects = list()
            object_poses = list()
            for i in range(self.get_option('nr_objects', 4)):
                shape = np.random.choice([minigrid.Key, minigrid.Box, minigrid.Ball])
                color = np.random.choice(minigrid.COLOR_NAMES)

                while True:
                    pose = np.random.randint(1, 6, size=2)
                    if self.grid.get(*pose) is None and not np.all(pose == 3):
                        break

                object = shape(color)
                objects.append(object)
                object_poses.append(pose)
                self.grid.set(*pose, object)
            self.goal_obj = goal = np.random.choice(objects)
            self.goal_pose = object_poses[objects.index(goal)]
            self.mission = get_domain().parse(f'(exists (?o - item) (and (robot-holding r ?o) (is-{goal.type} ?o) (is-{goal.color} ?o)))')

            path = find_path_to_obj(self, tuple(self.goal_pose))
            if path is not None:
                break

    def _gen_grid_open(self, width, height):
        self.grid = minigrid.Grid(width, height)
        self.agent_pos = (3, 3)
        self.agent_dir = 0
        self.grid.horz_wall(0, 0, 7)
        self.grid.horz_wall(0, 6, 7)
        self.grid.vert_wall(0, 0, 7)
        self.grid.vert_wall(6, 0, 7)

        objects = list()
        for i in range(4):
            color = np.random.choice(minigrid.COLOR_NAMES)

            while True:
                pose = np.random.randint(1, 6)
                dir = i
                # dir = np.random.randint(0, 4)
                if dir == 0:
                    pose = (pose, 0)
                elif dir == 1:
                    pose = (pose, 6)
                elif dir == 2:
                    pose = (0, pose)
                elif dir == 3:
                    pose = (6, pose)

                if self.grid.get(*pose).type != 'door':
                    break

            object = minigrid.Door(color)
            objects.append(object)
            self.grid.set(*pose, object)

        self.goal_obj = goal = np.random.choice(objects)
        self.mission = get_domain().parse(
            f'(exists (?o - item) (and (is-open ?o) (is-{goal.color} ?o)))'
        )

    def _gen_grid_generalization(self, width, height):
        for _ in range(self.get_option('max_trials', 100)):
            self.grid = minigrid.Grid(width, height)
            self.agent_pos = (3, 3)
            self.agent_dir = 0
            self.grid.horz_wall(0, 0, 7)
            self.grid.horz_wall(0, 6, 7)
            self.grid.vert_wall(0, 0, 7)
            self.grid.vert_wall(6, 0, 7)

            objects = list()
            object_poses = list()
            for i in range(self.get_option('nr_objects', 4)):
                shape = np.random.choice([minigrid.Key, minigrid.Box, minigrid.Ball])
                color = np.random.choice(minigrid.COLOR_NAMES)

                while True:
                    pose = np.random.randint(1, 6, size=2)
                    if self.grid.get(*pose) is None and not np.all(pose == 3):
                        break

                object = shape(color)
                objects.append(object)
                object_poses.append(pose)
                self.grid.set(*pose, object)

            self.goal_obj = goal = np.random.choice(objects, size=2, replace=False)
            self.goal_pose = object_poses[objects.index(goal[0])], object_poses[objects.index(goal[1])]
            self.mission = get_domain().parse(f"""(and
                (exists (?o - item) (and (robot-holding r ?o) (is-{goal[0].type} ?o) (is-{goal[0].color} ?o)))
                (exists (?o - item) (and (robot-is-facing r ?o) (is-{goal[1].type} ?o) (is-{goal[1].color} ?o)))
            )""")

            path = find_path_to_obj(self, tuple(self.goal_pose[0]))
            if path is not None:
                break
            path = find_path_to_obj(self, tuple(self.goal_pose[1]))
            if path is not None:
                break

    def compute_obs(self):
        return {'state': MiniGridStateV20220407.from_env(self), 'mission': self.mission}

    def compute_done(self):
        if self.task in ('goto', 'goto2', 'gotosingle'):
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is not None and fwd_cell.type == self.goal_obj.type and fwd_cell.color == self.goal_obj.color:
                return True
        elif self.task == 'pickup':
            if self.carrying is not None and self.carrying.color == self.goal_obj.color and self.carrying.type == self.goal_obj.type:
                return True
        elif self.task == 'open':
            for _, _, obj in self.iter_objects():
                if obj.color == self.goal_obj.color and obj.type == self.goal_obj.type and obj.is_open:
                    return True
        elif self.task == 'generalization':
            if self.carrying is not None and self.carrying.color == self.goal_obj[0].color and self.carrying.type == self.goal_obj[0].type:
                fwd_pos = self.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None and fwd_cell.type == self.goal_obj[1].type and fwd_cell.color == self.goal_obj[1].color:
                    return True
        else:
            raise ValueError(f'Unknown task: {self.task}.')
        return False

    def reset(self):
        super().reset()
        return self.compute_obs()

    def step_inner(self, action):
        super().step(action)

    def step(self, action: pdsrl.RLEnvAction):
        if action.name == 'move':
            self.step_move_to(action.args[1], action.args[2])
        elif action.name == 'forward':
            self.step_forward()
        elif action.name == 'lturn':
            self.step_lturn()
        elif action.name == 'rturn':
            self.step_rturn()
        elif action.name == 'pickup':
            self.step_pickup()
        elif action.name == 'place':
            self.step_drop()
        elif action.name == 'toggle':
            self.step_inner(self.Actions.toggle)
        else:
            raise ValueError(f'Unknown action: {action}.')

        obs = self.compute_obs()
        done = self.compute_done()
        return obs, -1, done, {}

    def step_move_to(self, pose, dir, traj=None):
        x, y = _map_int(pose)
        dir, = _map_int(dir)
        if self.grid.get(x, y) is None or self.grid.get(x, y).can_overlap():
            self.agent_pos = (x, y)
            self.agent_dir = dir

    def step_pickup(self):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell and fwd_cell.can_pickup():
            if self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = (-1, -1)
                self.grid.set(*fwd_pos, None)

    def step_forward(self):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)

    def step_lturn(self):
        self.agent_dir = (self.agent_dir - 1 + 4) % 4

    def step_rturn(self):
        self.agent_dir = (self.agent_dir + 1) % 4

    def step_drop(self):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None and self.carrying:
            self.grid.set(*fwd_pos, self.carrying)
            self.carrying.cur_pos = tuple(fwd_pos)
            self.carrying = None

    def step_toggle(self):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell:
            fwd_cell.toggle(self, fwd_pos)

    def debug_print(self):
        print(self)


class MiniGridStateV20220407(pds.State):
    @classmethod
    def from_env(cls, env: MiniGridEnvV20220407, ignore_walls: bool = True):
        if g_domain_structure_mode == 'full':
            return cls.from_env_full(env, ignore_walls=ignore_walls)
        elif g_domain_structure_mode == 'basic':
            return cls.from_env_basic(env, ignore_walls=ignore_walls)
        elif g_domain_structure_mode == 'robokin':
            return cls.from_env_robokin(env, ignore_walls=ignore_walls)
        elif g_domain_structure_mode == 'abskin':
            return cls.from_env_robokin(env, ignore_walls=ignore_walls)
        else:
            raise ValueError('Unknown domain structure mode: {}.'.format(g_domain_structure_mode))

    @classmethod
    def from_env_full(cls, env: MiniGridEnvV20220407, ignore_walls: bool = False):
        object_names = ['r']
        object_types = ['robot']
        object_type2id = dict()
        for k in minigrid.OBJECT_TO_IDX:
            object_type2id[k] = 0

        object_images = list()
        object_poses = list()
        objects = list()
        for x, y, obj in env.iter_objects():
            if ignore_walls and obj.type == 'wall':
                continue
            obj.name = f'{obj.type}:{object_type2id[obj.type]}'
            object_names.append(obj.name)
            object_types.append('item')
            object_images.append(obj.encode())
            object_poses.append((x, y))
            object_type2id[obj.type] += 1
            objects.append(obj)

        domain = get_domain()
        state = cls([domain.types[t] for t in object_types], pds.ValueDict(), object_names)
        ctx = pds.TensorDictDefHelper(domain, state)

        predicates = list()
        for obj, obj_name in zip(objects, object_names[1:]):
            if obj.type == 'wall':
                pass
            else:
                predicates.append(ctx.pickable(obj_name))
            if obj.type == 'door':
                predicates.append(ctx.toggleable(obj_name))
        if env.carrying is not None:
            predicates.append(ctx.robot_holding('r', env.carrying.name))

        ctx.define_predicates(predicates)
        ctx.define_feature('robot-pose', torch.tensor([env.agent_pos], dtype=torch.float32))
        ctx.define_feature('robot-direction', torch.tensor([[env.agent_dir]], dtype=torch.int64))
        ctx.define_feature('item-pose', torch.tensor(object_poses, dtype=torch.float32))
        ctx.define_feature('item-image', torch.tensor(object_images, dtype=torch.float32))
        return state

    @classmethod
    def from_env_basic(cls, env: MiniGridEnvV20220407, ignore_walls: bool = False):
        object_names = ['r']
        object_types = ['robot']
        object_type2id = dict()
        for k in minigrid.OBJECT_TO_IDX:
            object_type2id[k] = 0

        robot_images = list()
        robot_images.append(
            env.agent_pos + (env.agent_dir, )
        )

        object_images = list()
        objects = list()
        for x, y, obj in env.iter_objects():
            if ignore_walls and obj.type == 'wall':
                continue
            obj.name = f'{obj.type}:{object_type2id[obj.type]}'
            object_names.append(obj.name)
            object_types.append('item')
            object_images.append(obj.encode() + (x, y))
            object_type2id[obj.type] += 1
            objects.append(obj)

        domain = get_domain()
        state = cls([domain.types[t] for t in object_types], pds.ValueDict(), object_names)
        ctx = pds.TensorDictDefHelper(domain, state)
        ctx.define_feature('robot-image', torch.tensor(robot_images, dtype=torch.float32))
        ctx.define_feature('item-image', torch.tensor(object_images, dtype=torch.float32))
        return state

    @classmethod
    def from_env_robokin(cls, env: MiniGridEnvV20220407, ignore_walls: bool = False):
        object_names = ['r']
        object_types = ['robot']
        object_type2id = dict()
        for k in minigrid.OBJECT_TO_IDX:
            object_type2id[k] = 0

        object_images = list()
        object_poses = list()
        objects = list()
        for x, y, obj in env.iter_objects():
            if ignore_walls and obj.type == 'wall':
                continue
            obj.name = f'{obj.type}:{object_type2id[obj.type]}'
            object_names.append(obj.name)
            object_types.append('item')
            object_images.append(obj.encode())
            object_poses.append((x, y))
            object_type2id[obj.type] += 1
            objects.append(obj)

        domain = get_domain()
        state = cls([domain.types[t] for t in object_types], pds.ValueDict(), object_names)
        ctx = pds.TensorDictDefHelper(domain, state)
        ctx.define_feature('robot-pose', torch.tensor([env.agent_pos], dtype=torch.float32))
        ctx.define_feature('robot-direction', torch.tensor([[env.agent_dir]], dtype=torch.int64))
        ctx.define_feature('item-pose', torch.tensor(object_poses, dtype=torch.float32))
        ctx.define_feature('item-image', torch.tensor(object_images, dtype=torch.float32))
        return state


def make(*args, **kwargs):
    return MiniGridEnvV20220407(*args, **kwargs)


def visualize_planner(env: MiniGridEnvV20220407, planner):
    torch.set_grad_enabled(False)
    while True:
        init_obs = env.reset()
        state, mission = init_obs['state'], init_obs['mission']
        assert planner is not None
        plan = planner(state, mission)

        cmd = visualize_plan(env, plan)
        if cmd == 'q':
            break


def visualize_plan(env: MiniGridEnvV20220407, plan):
    env.render()
    print('Plan: ' + ', '.join([str(x) for x in plan]))
    print('Press <Enter> to visualize.')
    _ = input('> ').strip()

    for action in plan:
        print('Executing action: ' + str(action))
        if action.name == 'move':
            pose = action.arguments[1].tensor.tolist()
            dir = action.arguments[2].tensor.item()
            for action in minigrid.find_path(env, pose, dir):
                env.step_inner(action)
                env.render()
                time.sleep(0.5)
        elif action.name == 'forward':
            env.step_inner(MiniGridEnvV20220407.Actions.forward)
        elif action.name == 'lturn':
            env.step_inner(MiniGridEnvV20220407.Actions.left)
        elif action.name == 'rturn':
            env.step_inner(MiniGridEnvV20220407.Actions.right)
        elif action.name == 'pickup':
            env.step_inner(MiniGridEnvV20220407.Actions.pickup)
        elif action.name == 'toggle':
            env.step_inner(MiniGridEnvV20220407.Actions.toggle)
        else:
            raise NotImplementedError(action)
        env.render()
        time.sleep(0.5)

    print('Visualization finished.')
    print('Press <Enter> to continue. Type q to quit.')
    cmd = input('> ').strip()
    return cmd
