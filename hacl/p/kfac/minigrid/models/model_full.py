import torch
import torch.nn as nn
import hacl.pdsketch as pds
from hacl.p.kfac.nn.int_embedding import IntEmbedding

__all__ = ['ModelFull']


class ModelFull(pds.nn.PDSketchMBRLModel):
    def init_networks(self, domain):
        self.register_buffer('empty_pose', torch.tensor([-1, -1], dtype=torch.float32))
        self.register_buffer('dir_to_vec', torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=torch.float32))

        domain.register_external_function_implementation('feature::empty-pose', self.empty_pose_fn)
        domain.register_external_function_implementation('feature::facing', self.facing)
        domain.register_external_function_implementation('type::pose::equal', self.pose_equal)
        domain.register_external_function_implementation('type::direction::equal', self.pose_equal)

        domain.register_external_function_implementation('generator::gen-direction', self.gen_direction)
        domain.register_external_function_implementation('generator::gen-pose-neq', self.gen_pose_neq)
        domain.register_external_function_implementation('generator::gen-facing-robot', self.gen_facing_robot)

        if 'feature::direction-left' in domain.external_functions:
            domain.register_external_function_implementation('feature::direction-left', self.direction_left)
        if 'feature::direction-right' in domain.external_functions:
            domain.register_external_function_implementation('feature::direction-right', self.direction_right)

        for k in ['item-type', 'item-color', 'item-state']:
            module = IntEmbedding(32)
            self.add_module(k, module)
            domain.register_external_function_implementation('feature::' + k + '::f', module)

        for k in ['is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey', 'is-key', 'is-ball', 'is-box', 'is-door', 'is-open']:
            identifier = 'derived::' + k + '::f'
            if identifier in domain.external_functions:
                module = nn.Sequential(
                    pds.nn.UnpackValue(), nn.Linear(32, 1), pds.nn.Squeeze(-1),
                    nn.Sigmoid()
                )
                self.add_module(k, module)
                domain.register_external_function_implementation(identifier, module)

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


