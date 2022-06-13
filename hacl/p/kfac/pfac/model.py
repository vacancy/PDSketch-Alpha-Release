import os.path as osp
import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
import hacl.pdsketch as pds
from hacl.nn.lenet import LeNetRGB32
from hacl.nn.quantization.vector_quantizer import VectorQuantizer
from hacl.p.kfac.nn.int_embedding import IntEmbedding, ConcatIntEmbedding

__all__ = ['ModelPFac', 'load_domain']


def load_domain():
    return pds.load_domain_file(osp.join(osp.dirname(__file__), 'pfac-v20220517.pdsketch'))


class ModelPFac(pds.nn.PDSketchMBRLModel):
    USE_GT_CLASSIFIER = False

    def set_debug_options(self, use_gt_classifier=None):
        if use_gt_classifier is not None:
            type(self).USE_GT_CLASSIFIER = use_gt_classifier

    def init_networks(self, domain):
        self.options['bptt'] = True

        dimensions = 32
        self.functions.item_feature = nn.Sequential(pds.nn.UnpackValue(), LeNetRGB32(dimensions))
        jactorch.mark_freezed(self.functions.item_feature)
        domain.register_external_function_implementation('feature::item-feature::f', self.forward_item_feature)
        self.functions.container_feature = nn.Sequential(pds.nn.UnpackValue(), LeNetRGB32(dimensions))
        domain.register_external_function_implementation('feature::container-feature::f', self.forward_container_feature)

        for k in ['is-red', 'is-green', 'is-yellow', 'is-purple', 'is-pink', 'is-cyan', 'is-brown', 'is-orange', 'is-target']:
            identifier = 'derived::' + k + '::f'
            if identifier in domain.external_functions:
                module = pds.nn.AutoBatchWrapper(
                    nn.Sequential(jacnn.MLPLayer(32, 1, [32], activation='relu'), nn.Sigmoid()),
                    squeeze=-1
                )
                self.functions.add_module(k, module)
                if k != 'is-target':
                    jactorch.mark_freezed(module)
                domain.register_external_function_implementation(identifier, module)

        for k in ['is-left', 'is-right', 'is-on', 'is-in']:
            identifier = 'derived::' + k + '::f'
            self.functions.register_parameter(identifier, nn.Parameter(torch.zeros(3)))
            identifier = 'derived::' + k + '::thresh'
            self.functions.register_parameter(identifier, nn.Parameter(torch.tensor(0.05, dtype=torch.float32)))
            identifier = 'derived::' + k + '::gamma'
            self.functions.register_parameter(identifier, nn.Parameter(torch.tensor(10.0, dtype=torch.float32)))

            identifier = 'derived::' + k + '::f'
            domain.register_external_function_implementation(identifier, self.gen_forward_relation(k))

        self.functions.move_into_f = pds.nn.AutoBatchWrapper(nn.Linear(3, 3))
        self.functions.move_into_g = pds.nn.AutoBatchWrapper(nn.Sequential(nn.Linear(dimensions, 1), nn.Sigmoid()), squeeze=-1)
        self.functions.move_into_h = pds.nn.AutoBatchWrapper(jacnn.MLPLayer(dimensions, dimensions, [64], activation='relu', last_activation=False))
        domain.register_external_function_implementation('action::move-into::f', self.functions.move_into_f)
        domain.register_external_function_implementation('action::move-into::g', self.functions.move_into_g)
        domain.register_external_function_implementation('action::move-into::h', self.functions.move_into_h)
        self.functions.move_to_f = pds.nn.AutoBatchWrapper(nn.Linear(3, 3))
        domain.register_external_function_implementation('action::move-to::f', self.functions.move_to_f)

    def forward_relation(self, k, pose1, pose2):
        pose1 = pose1.tensor
        pose2 = pose2.tensor
        param = getattr(self.functions, 'derived::' + k + '::f')
        thresh = getattr(self.functions, 'derived::' + k + '::thresh')
        gamma = getattr(self.functions, 'derived::' + k + '::gamma')
        norm = torch.norm((pose2 + param) - pose1, p=2, dim=-1)
        out = torch.sigmoid((thresh - norm) * gamma)

        if k == 'is-in':
            return (torch.norm(pose1 - pose2, p=1, dim=-1) < 0.08).float()

        return out

    def gen_forward_relation(self, k):
        def func(pose1, pose2, k=k):
            return self.forward_relation(k, pose1, pose2)
        return func

    def forward_item_feature(self, images):
        return pds.nn.AutoBatchWrapper(self.functions.item_feature[1])(images)

    def forward_container_feature(self, images):
        return pds.nn.AutoBatchWrapper(self.functions.container_feature[1])(images)
