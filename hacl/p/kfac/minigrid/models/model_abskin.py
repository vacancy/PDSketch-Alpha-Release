import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
import hacl.pdsketch as pds
from typing import List
from hacl.p.kfac.nn.int_embedding import IntEmbedding, ConcatIntEmbedding

__all__ = ['ModelAbskin']


class ModelAbskin(pds.nn.PDSketchMBRLModel):
    USE_GT_CLASSIFIER = False

    def set_debug_options(self, use_gt_classifier=None):
        if use_gt_classifier is not None:
            type(self).USE_GT_CLASSIFIER = use_gt_classifier

    def init_networks(self, domain):
        self.register_buffer('empty_pose', torch.tensor([-1, -1], dtype=torch.float32))
        self.register_buffer('dir_to_vec', torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=torch.float32))
        domain.register_external_function_implementation('type::direction::equal', self.pose_equal)

        if 'feature::direction-left' in domain.external_functions:
            domain.register_external_function_implementation('feature::direction-left', self.direction_left)
        if 'feature::direction-right' in domain.external_functions:
            domain.register_external_function_implementation('feature::direction-right', self.direction_right)

        self.functions.item_feature = ConcatIntEmbedding({
            3: IntEmbedding(64, input_dim=3, value_range=(0, 16), attach_input=type(self).USE_GT_CLASSIFIER),
        })
        domain.register_external_function_implementation('feature::item-feature::f', self.functions.item_feature)

        self.functions.robot_direction_feature = IntEmbedding(16, input_dim=1, value_range=4, concat_input=True)
        domain.register_external_function_implementation('derived::direction-feature::f', self.functions.robot_direction_feature)

        robot_embedding_dim = 2 + self.functions.robot_direction_feature.embedding_dim  # pose + dir
        item_embedding_dim = self.functions.item_feature.output_dim

        self.functions.is_facing = pds.nn.AutoBatchWrapper(nn.Sequential(
            jacnn.MLPLayer(robot_embedding_dim + 2, 1, [128], flatten=False),
            nn.Sigmoid()
        ), squeeze=-1)
        domain.register_external_function_implementation('derived::is-facing::f', self.functions.is_facing)

        self.functions.robot_holding = pds.nn.AutoBatchWrapper(nn.Sequential(
            jacnn.MLPLayer(robot_embedding_dim + 2 + item_embedding_dim, 1, [128], flatten=False),
            nn.Sigmoid()
        ), squeeze=-1)
        domain.register_external_function_implementation('derived::robot-holding::f', self.functions.robot_holding)

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

        # self.functions.action_forward = self.action_forward
        self.functions.action_forward_f = jacnn.MLPLayer(item_embedding_dim + 1, 1, [128], flatten=False)
        self.functions.action_forward_g = jacnn.MLPLayer(robot_embedding_dim, 2, [128], flatten=False)
        domain.register_external_function_implementation('action::forward::f', self.action_forward_f, auto_broadcast=False)
        domain.register_external_function_implementation('action::forward::g', self.action_forward_g)

        domain.register_external_function_implementation('action::pickup::f', self.action_pickup_f, auto_broadcast=False)

    def pose_equal(self, p1, p2):
        return (torch.abs(p1.tensor - p2.tensor) < 0.5).all(dim=-1)

    def direction_left(self, d):
        return (d.tensor - 1) % 4

    def direction_right(self, d):
        return (d.tensor + 1) % 4

    def action_forward_f(self, robot_pose, robot_direction, item_feature):
        mask = item_feature.tensor_mask > 0.5  # this line breaks the gradient, but it's fine.
        feature = torch.sigmoid(
            self.functions.action_forward_f(torch.cat([item_feature.tensor, item_feature.tensor_mask.unsqueeze(-1)], dim=-1))
        ).squeeze(-1)
        blocked = torch.min(mask, feature)
        blocked = blocked.float()
        blocked_any = blocked.amax(dim=0).unsqueeze(0)
        # print(feature)
        # return torch.ones(1, dtype=torch.float32, device=robot_pose.tensor.device)
        # print(feature, item_feature.tensor_mask)
        # return feature.amax(dim=0).unsqueeze(0)
        # rv = 1 - item_feature.tensor_mask.amax(dim=0).unsqueeze(0)
        # print(item_feature.tensor_mask)
        # print(rv)
        # return rv
        # return torch.tensor(1, dtype=torch.float32, device=robot_pose.tensor.device)
        return 1 - blocked_any

    def action_forward_g(self, robot_pose, robot_direction):
        rv = self.functions.action_forward_g(torch.cat([robot_pose.tensor, robot_direction.tensor], dim=-1))
        # print(robot_pose.tensor, robot_direction.tensor[-1], rv)
        return robot_pose.tensor + rv

    def action_pickup_f(self, robot_pose, robot_direction, item_pose, item_feature):
        f = torch.zeros_like(item_pose.tensor) - 1
        return f


class HeuristicModelAbskin(nn.Module):
    def __init__(self, domain: pds.Domain, predicates: List[str]):
        super().__init__()

        self.domain = domain
        self.item_feature = ConcatIntEmbedding({
            3: IntEmbedding(64, input_dim=3, value_range=(0, 16)),
        })
        self.robot_direction_feature = IntEmbedding(16, input_dim=1, value_range=4)

        robot_embedding_dim = 2 + self.robot_direction_feature.embedding_dim  # pose + dir
        item_embedding_dim = 2 + self.item_feature.output_dim

        self.predicate2index = {predicate: i + 1 for i, predicate in enumerate(predicates)}
        self.predicate2index['<PAD>'] = 0
        self.goal_embedding = nn.Embedding(len(predicates) + 1, 32, padding_idx=0)
        self.state_encoder = jacnn.NeuralLogicMachine(
            3, 1, [robot_embedding_dim + 64, item_embedding_dim], [128, 1], 'mlp', 128, activation='relu'
        )
        self.value_network = nn.Linear(128, 1)
        self.loss = nn.SmoothL1Loss()

    def forward(self, state: pds.State, goal):
        robot_feature = torch.cat(
            [state['robot-pose'].tensor, self.robot_direction_feature(state['robot-direction'].tensor)],
            dim=-1
        )
        robot_feature = robot_feature[0]
        robot_feature = torch.cat(
            [robot_feature, self.forward_goal(goal).flatten()]
        )
        object_feature = torch.cat(
            [state['item-pose'].tensor, self.item_feature(state['item-image'].tensor)],
            dim=-1
        )

        output = self.state_encoder([robot_feature.unsqueeze(0), object_feature.unsqueeze(0)])[0]
        output = self.value_network(output).squeeze(-1).squeeze(0)
        return output

    def bc(self, states: List[pds.State], goal):
        batched_states = pds.BatchState.from_states(self.domain, states)
        robot_feature = torch.cat(
            [batched_states['robot-pose'].tensor, self.robot_direction_feature(batched_states['robot-direction'].tensor)],
            dim=-1
        )
        robot_feature = robot_feature[:, 0]
        robot_feature = torch.cat(
            [robot_feature, jactorch.add_dim(self.forward_goal(goal).flatten(), 0, len(states))],
            dim=-1
        )
        object_feature = torch.cat(
            [batched_states['item-pose'].tensor, self.item_feature(batched_states['item-image'].tensor)],
            dim=-1
        )
        output = self.state_encoder([robot_feature, object_feature])[0]
        output = self.value_network(output).squeeze(-1)

        returns = len(states) - torch.arange(len(states), device=output.device)
        returns = returns.float()
        loss = self.loss(output, returns)
        return loss, {'loss': loss}, {}

    def forward_goal(self, goal):
        predicate_names = list()
        for predicate in goal.expr.arguments:
            if predicate.feature_def.name in self.predicate2index:
                predicate_names.append(self.predicate2index[predicate.feature_def.name])
        assert len(predicate_names) in (1, 2)
        if len(predicate_names) == 1:
            predicate_names.append(0)
        embedding = self.goal_embedding(torch.tensor(predicate_names, dtype=torch.long, device=self.goal_embedding.weight.device))
        return embedding
