import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
from typing import List
from hacl.pdsketch.interface.v2.state import State, BatchState
from hacl.pdsketch.interface.v2.domain import Domain
from hacl.p.kfac.nn.int_embedding import IntEmbedding, FallThroughEmbedding, ConcatIntEmbedding

__all__ = ['GeneralPolicyNetwork']


class GeneralPolicyNetwork(nn.Module):
    def __init__(self, domain: Domain, action_space, predicates: List[str], updown=False):
        super().__init__()
        self.domain = domain
        self.action_space = action_space
        self.updown = updown

        self.robot_feature = ConcatIntEmbedding({
            # 2: IntEmbedding(64, input_dim=2, value_range=(-1, 15)),
            2: FallThroughEmbedding(input_dim=2),
            1: IntEmbedding(16, input_dim=1, value_range=4, concat_input=True),
            # 1: FallThroughEmbedding(input_dim=1),
        })
        robot_embedding_dim = self.robot_feature.output_dim
        self.item_feature = ConcatIntEmbedding({
            3: IntEmbedding(64, input_dim=3, value_range=(0, 16)),
            # 2: IntEmbedding(64, input_dim=2, value_range=(-1, 15)),
            2: FallThroughEmbedding(input_dim=2)
        })
        item_embedding_dim = self.item_feature.output_dim

        if self.updown:
            self.succ_embedding = nn.Embedding(2, 32)
            robot_embedding_dim += 32
        else:
            self.add_module('succ_embedding', None)

        self.goal_embedding = nn.Embedding(len(predicates) + 1, 32, padding_idx=0)
        self.state_encoder = jacnn.NeuralLogicMachine(
            3, 1, [robot_embedding_dim + 64, item_embedding_dim], [128, 1], 'mlp', 128, activation='relu'
        )
        self.predicate2index = {predicate: i + 1 for i, predicate in enumerate(predicates)}
        self.predicate2index['<PAD>'] = 0
        self.action2index = {get_action_desc(action): i for i, action in enumerate(action_space)}
        self.action_classifier = nn.Linear(128, len(action_space))
        self.loss = nn.CrossEntropyLoss()

    def forward_state(self, state: State, goal, succ: int = 1):
        robot_tensors = list()
        object_tensors = list()

        robot_tensors.append(self.robot_feature(state['robot-image'])[0])
        goal_tensor = self.forward_goal(goal).flatten()
        robot_tensors.append(goal_tensor)
        if self.updown:
            succ_tensor = self.succ_embedding.weight[int(succ)]
            robot_tensors.append(succ_tensor)
        object_tensors.append(self.item_feature(state['item-image']))

        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = torch.cat(object_tensors, dim=-1)

        output = self.state_encoder([robot_tensor.unsqueeze(0), object_tensor.unsqueeze(0)])[0]
        output = self.action_classifier(output)
        return output.squeeze(0)

    def forward_goal(self, goal):
        predicate_names = list()
        for predicate in goal.expr.arguments:
            if predicate.feature_def.name in self.predicate2index:
                predicate_names.append(predicate.feature_def.name)
        assert len(predicate_names) in (1, 2)

        predicate_names = [self.predicate2index[n] for n in predicate_names]
        if len(predicate_names) == 1:
            predicate_names.append(0)
        embedding = self.goal_embedding(torch.tensor(predicate_names, dtype=torch.long, device=self.goal_embedding.weight.device))
        return embedding

    def forward_actions(self, actions):
        actions = [self.action2index[get_action_desc(action)] for action in actions]
        return torch.tensor(actions, dtype=torch.long, device=self.action_classifier.weight.device)

    def bc(self, states: List[State], actions, goal, succ):
        if len(states) <= 1:
            return 0, {'loss': 0}, {}

        states = states[:-1]
        batched_states = BatchState.from_states(self.domain, states)

        robot_tensors = list()
        object_tensors = list()

        robot_tensors.append(self.robot_feature(batched_states['robot-image'])[:, 0])
        goal_tensor = jactorch.add_dim(self.forward_goal(goal).flatten(), 0, len(states))
        robot_tensors.append(goal_tensor)
        object_tensors.append(self.item_feature(batched_states['item-image']))

        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = torch.cat(object_tensors, dim=-1)

        output = self.state_encoder([robot_tensor, object_tensor])[0]
        output = self.action_classifier(output)
        loss = self.loss(output, self.forward_actions(actions))
        return loss, {'loss': loss}, {}

    def dt_full(self, states: List[State], actions, goal, succ):
        if len(states) <= 1:
            return 0, {'loss': 0}, {}

        states = states[:-1]
        batched_states = BatchState.from_states(self.domain, states)

        robot_tensors = list()
        object_tensors = list()

        robot_tensors.append(self.robot_feature(batched_states['robot-image'])[:, 0])
        goal_tensor = jactorch.add_dim(self.forward_goal(goal).flatten(), 0, len(states))
        robot_tensors.append(goal_tensor)
        assert self.updown
        succ_tensor = jactorch.add_dim(self.succ_embedding.weight[int(succ)], 0, len(states))
        robot_tensors.append(succ_tensor)
        object_tensors.append(self.item_feature(batched_states['item-image']))

        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = torch.cat(object_tensors, dim=-1)

        output = self.state_encoder([robot_tensor, object_tensor])[0]
        output = self.action_classifier(output)
        loss = self.loss(output, self.forward_actions(actions))
        return loss, {'loss': loss}, {}


def get_action_desc(action):
    return '{}({})'.format(action.name, ', '.join(map(str, action.arguments)))

