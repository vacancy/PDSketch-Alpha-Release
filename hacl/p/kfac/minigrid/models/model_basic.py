import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
import hacl.pdsketch as pds
from hacl.p.kfac.nn.int_embedding import IntEmbedding, FallThroughEmbedding, ConcatIntEmbedding

__all__ = ['ModelBasic']


class ModelBasic(pds.nn.PDSketchMBRLModel):
    USE_GT_FACING = False
    USE_GT_CLASSIFIER = False

    def set_debug_options(self, use_gt_facing=None, use_gt_classifier=None):
        if use_gt_facing is not None:
            type(self).USE_GT_FACING = use_gt_facing
        if use_gt_classifier is not None:
            type(self).USE_GT_CLASSIFIER = use_gt_classifier

    def init_networks(self, domain):
        self.register_buffer('dir_to_vec', torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=torch.float32))

        self.functions.robot_feature = ConcatIntEmbedding({
            # 2: IntEmbedding(64, input_dim=2, value_range=(-1, 15)),
            2: FallThroughEmbedding(input_dim=2),
            1: IntEmbedding(16, input_dim=1, value_range=4, concat_input=True),
            # 1: FallThroughEmbedding(input_dim=1),
        })
        robot_embedding_dim = self.functions.robot_feature.output_dim
        self.functions.item_feature = ConcatIntEmbedding({
            3: IntEmbedding(64, input_dim=3, value_range=(0, 16), attach_input=type(self).USE_GT_CLASSIFIER),
            # 2: IntEmbedding(64, input_dim=2, value_range=(-1, 15)),
            2: FallThroughEmbedding(input_dim=2)
        })
        item_embedding_dim = self.functions.item_feature.output_dim

        # self.functions.action_forward = self.action_forward
        self.functions.action_forward_f = jacnn.MLPLayer(robot_embedding_dim, robot_embedding_dim, [128], flatten=False)
        self.functions.action_forward_g = jacnn.MLPLayer(robot_embedding_dim + item_embedding_dim, 1, [128], flatten=False)
        self.functions.action_pickup_f = jacnn.MLPLayer(robot_embedding_dim + item_embedding_dim, item_embedding_dim, [128], flatten=False)
        self.functions.action_pickup_g = jacnn.MLPLayer(robot_embedding_dim + item_embedding_dim, 1, [128], flatten=False)

        self.functions.action_lturn = pds.nn.AutoBatchWrapper(jacnn.MLPLayer(robot_embedding_dim, robot_embedding_dim, [128], flatten=False))
        self.functions.action_rturn = pds.nn.AutoBatchWrapper(jacnn.MLPLayer(robot_embedding_dim, robot_embedding_dim, [128], flatten=False))

        domain.register_external_function_implementation('feature::robot-feature::f', self.functions.robot_feature)
        domain.register_external_function_implementation('feature::item-feature::f', self.functions.item_feature)
        domain.register_external_function_implementation('action::forward::f', self.action_forward)
        domain.register_external_function_implementation('action::lturn::f', self.functions.action_lturn)
        domain.register_external_function_implementation('action::rturn::f', self.functions.action_rturn)
        domain.register_external_function_implementation('action::pickup::f', self.action_pickup)

        for k in ['is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey', 'is-key', 'is-ball', 'is-box', 'is-door', 'is-open']:
            identifier = 'derived::' + k + '::f'
            if identifier in domain.external_functions:
                if type(self).USE_GT_CLASSIFIER:
                    domain.register_external_function_implementation(identifier, self.generate_groundtruth_classifier(k))
                else:
                    module = nn.Sequential(
                        pds.nn.UnpackValue(), nn.Linear(item_embedding_dim, 1),
                        pds.nn.Squeeze(-1),
                        nn.Sigmoid()
                    )
                    self.functions.add_module(k, module)
                    domain.register_external_function_implementation(identifier, module)

        self.functions.robot_holding = pds.nn.AutoBatchWrapper(nn.Sequential(
            jacnn.MLPLayer(robot_embedding_dim + item_embedding_dim, 1, [128], flatten=False),
            nn.Sigmoid()
        ), squeeze=-1)
        domain.register_external_function_implementation('derived::robot-holding::f', self.functions.robot_holding)
        self.functions.robot_is_facing = pds.nn.AutoBatchWrapper(nn.Sequential(
            jacnn.MLPLayer(robot_embedding_dim + item_embedding_dim, 1, [128], flatten=False),
            nn.Sigmoid()
        ), squeeze=-1)

        if type(self).USE_GT_FACING:
            domain.register_external_function_implementation('derived::robot-is-facing::f', self.robot_is_facing_gt)
        else:
            domain.register_external_function_implementation('derived::robot-is-facing::f', self.functions.robot_is_facing)

    def generate_groundtruth_classifier(self, property):
        if property in ['is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey']:
            color_name = property[3:]
            from hacl.envs.gridworld.minigrid.gym_minigrid.minigrid import COLOR_TO_IDX
            target = COLOR_TO_IDX[color_name]
            def classifier(x, target=target):
                return x.tensor[..., 1] == target
            return classifier
        elif property in ['is-key', 'is-ball', 'is-box', 'is-door']:
            from hacl.envs.gridworld.minigrid.gym_minigrid.minigrid import OBJECT_TO_IDX
            target = OBJECT_TO_IDX[property[3:]]
            def classifier(x, target=target):
                return x.tensor[..., 0] == target
            return classifier
        elif property == 'is-open':
            def classifier(x):
                return x.tensor[..., 2] == 0
            return classifier

    def action_forward(self, robot, item):
        robot, item = robot.tensor, item.tensor
        robot = robot[0]
        assert item.dim() == 2

        f = self.functions.action_forward_f(robot.unsqueeze(0))[0]
        g = self.functions.action_forward_g(torch.cat([
            jactorch.add_dim(robot, 0, item.size(0)), item
        ], dim=-1)).sigmoid().min()
        # rv = f * g + robot * (1 - g)
        rv = robot + f * g
        # print(robot, f)
        return pds.Value(self.domain.features["robot-feature"].output_type, [], rv)

    def action_pickup(self, robot, item):
        robot, item = robot.tensor, item.tensor

        input = torch.cat((robot, item), dim=-1)
        f = self.functions.action_pickup_f(input)
        g = self.functions.action_pickup_g(input).sigmoid()

        f[..., -2:] = -1
        mask = torch.zeros_like(item)
        mask[..., -2:] = 1

        rv = item * (1 - mask) + (f * g + item * (1 - g)) * mask
        return rv

    def _dir_to_vec_fn(self, d):
        return self.dir_to_vec[d.long().flatten()].reshape(d.shape[:-1] + (2, ))

    def _facing_tensor(self, p, d):
        return p + self._dir_to_vec_fn(d)

    def _pose_equal_tensor(self, p1, p2):
        return (torch.abs(p1 - p2) < 0.5).all(dim=-1)

    def robot_is_facing_gt(self, robot, item):
        robot = pds.unwrap_value(robot)
        item = pds.unwrap_value(item)
        pose = robot[..., 0:2]
        dir = robot[..., 64:65]
        item_pose = item[..., 64:66]

        rv = self._pose_equal_tensor(self._facing_tensor(pose, dir), item_pose)
        return rv


