import torch
import torch.nn as nn
import jactorch.nn as jacnn
import hacl.pdsketch as pds
from hacl.p.kfac.nn.int_embedding import IntEmbedding, ConcatIntEmbedding

__all__ = ['ModelRobokin']


class ModelRobokin(pds.nn.PDSketchMBRLModel):
    USE_GT_FACING = False
    USE_GT_CLASSIFIER = False

    def set_debug_options(self, use_gt_facing=None, use_gt_classifier=None):
        if use_gt_facing is not None:
            type(self).USE_GT_FACING = use_gt_facing
        if use_gt_classifier is not None:
            type(self).USE_GT_CLASSIFIER = use_gt_classifier

    def init_networks(self, domain):
        self.functions.robot_direction_feature = IntEmbedding(16, input_dim=1, value_range=4, concat_input=True)
        self.functions.item_feature = ConcatIntEmbedding({
            3: IntEmbedding(64, input_dim=3, value_range=(0, 16), attach_input=type(self).USE_GT_CLASSIFIER),
        })
        robot_embedding_dim = 2 + self.functions.robot_direction_feature.embedding_dim  # pose + dir
        item_embedding_dim = self.functions.item_feature.output_dim

        self.register_buffer('empty_pose', torch.tensor([-1, -1], dtype=torch.float32))
        self.register_buffer('dir_to_vec', torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=torch.float32))
        domain.register_external_function_implementation('feature::empty-pose', self.empty_pose_fn)
        domain.register_external_function_implementation('feature::facing', self.facing)
        domain.register_external_function_implementation('type::pose::equal', self.pose_equal)
        domain.register_external_function_implementation('type::direction::equal', self.pose_equal)

        if 'feature::direction-left' in domain.external_functions:
            domain.register_external_function_implementation('feature::direction-left', self.direction_left)
        if 'feature::direction-right' in domain.external_functions:
            domain.register_external_function_implementation('feature::direction-right', self.direction_right)

        # self.functions.action_forward = self.action_forward
        self.functions.action_pickup_f = jacnn.MLPLayer(robot_embedding_dim + item_embedding_dim + 2, item_embedding_dim, [128], flatten=False)
        self.functions.action_pickup_g = jacnn.MLPLayer(robot_embedding_dim + item_embedding_dim + 2, 1, [128], flatten=False)
        domain.register_external_function_implementation('derived::robot-feature::f', self.robot_feature)
        domain.register_external_function_implementation('feature::item-feature::f', self.functions.item_feature)
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
            jacnn.MLPLayer(robot_embedding_dim + item_embedding_dim + 2, 1, [128], flatten=False),
            nn.Sigmoid()
        ), squeeze=-1)
        domain.register_external_function_implementation('derived::robot-holding::f', self.functions.robot_holding)

        self.functions.robot_is_facing = pds.nn.AutoBatchWrapper(nn.Sequential(
            jacnn.MLPLayer(robot_embedding_dim + item_embedding_dim + 2, 1, [128], flatten=False),
            nn.Sigmoid()
        ), squeeze=-1)
        if type(self).USE_GT_FACING:
            domain.register_external_function_implementation('derived::robot-is-facing::f', self.robot_is_facing_gt)
        else:
            domain.register_external_function_implementation('derived::robot-is-facing::f', self.functions.robot_is_facing)

        identifier = 'action::toggle::f'
        if identifier in domain.external_functions:
            module = pds.nn.AutoBatchWrapper( nn.Linear(96, 32) )
            self.add_module(identifier, module)
            domain.register_external_function_implementation(identifier, module)

    def empty_pose_fn(self):
        return self.empty_pose

    def dir_to_vec_fn(self, d):
        return self.dir_to_vec[d.flatten()].reshape(d.shape[:-1] + (2, ))

    def facing(self, p, d):
        return p.tensor + self.dir_to_vec_fn(d.tensor)

    def pose_equal(self, p1, p2):
        return (torch.abs(p1.tensor - p2.tensor) < 0.5).all(dim=-1)

    def gen_direction(self):
        return torch.randint(4, size=(1, )),

    def gen_pose_neq(self, pose1):
        return torch.zeros_like(pose1.tensor) + 3,

    def gen_facing_robot(self, target):
        i = torch.randint(4, size=(1, ))[0]
        return target.tensor + self.dir_to_vec[i], ((i + 2) % 4).unsqueeze(0)

    def direction_left(self, d):
        return (d.tensor - 1) % 4

    def direction_right(self, d):
        return (d.tensor + 1) % 4

    def robot_is_facing_gt(self, robot, item):
        robot = pds.unwrap_value(robot)
        item = pds.unwrap_value(item)
        pose = robot[..., 0:2]
        dir = robot[..., 64:65]
        item_pose = item[..., 64:66]

        rv = self._pose_equal_tensor(self._facing_tensor(pose, dir), item_pose)
        return rv

    def robot_feature(self, robot_pose, robot_direction):
        robot_pose = pds.unwrap_value(robot_pose)
        robot_direction = pds.unwrap_value(robot_direction)
        return torch.cat([robot_pose, self.functions.robot_direction_feature(robot_direction)], dim=-1)

    def action_pickup(self, robot_feature, item_pose, item_feature):
        robot_feature = pds.unwrap_value(robot_feature)
        item_pose = pds.unwrap_value(item_pose)
        item_feature = pds.unwrap_value(item_feature)

        input = torch.cat([robot_feature, item_pose, item_feature], dim=-1)
        f = torch.zeros_like(item_pose) - 1
        g = self.functions.action_pickup_g(input)
        return item_pose * (1 - g) + f * g

