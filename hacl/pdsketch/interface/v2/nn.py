import collections
import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
import jactorch.nn.functional as jacf

from typing import Sequence
from tabulate import tabulate
from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from jactorch.graph.context import ForwardContext, get_forward_context
from hacl.nn.quantization.gumbel_quantizer import GumbelQuantizer
from hacl.nn.quantization.vector_quantizer import VectorQuantizer

from .value import BOOL, VectorValueType, Value, QuantizedTensorValue
from .state import BatchState, concat_batch_states
from .expr import QINDEX, FunctionDef, ExpressionExecutionContext, ConstantExpression, AssignOp
from .domain import AugmentedFeatureStage, Domain, Operator, OperatorApplier

logger = get_logger(__file__)

__all__ = ['UnpackValue', 'Concat', 'Squeeze', 'AutoBatchWrapper', 'QuantizerMode', 'SimpleQuantizedEncodingModule', 'SimpleQuantizationWrapper',
    'PDSketchMultiStageModel', 'PDSketchMBRLModel']


class UnpackValue(nn.Module):
    """A simple module that unpacks PyTorch tensors from Value objects."""

    def forward(self, *tensors):
        if len(tensors) == 1:
            return tensors[0].tensor
        return tuple([t.tensor for t in tensors])


class Concat(nn.Module):
    """Concatenate all inputs along the last axis."""
    def forward(self, *tensors):
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-1)


class Squeeze(nn.Module):
    """Squeeze a give axis."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.squeeze(self.dim)


class AutoBatchWrapper(nn.Module):
    def __init__(self, wrapped, concat=True, squeeze=None):
        super().__init__()
        self.wrapped = wrapped
        self.concat = concat
        self.squeeze = squeeze

    def forward(self, *values: Value):
        v = values[0]
        total_batch_dims = v.batch_dims + len(v.batch_variables)
        tensors = tuple(v.tensor for v in values)

        if total_batch_dims == 0:
            tensors = tuple(t.unsqueeze(0) for t in tensors)
        else:
            tensors = tuple(t.reshape((-1,) + t.shape[total_batch_dims:]) for t in tensors)

        if self.concat:
            if len(tensors) == 1:
                input_tensor = tensors[0]
            else:
                input_tensor = torch.cat(tensors, dim=-1)
            rv = self.wrapped(input_tensor)
        else:
            rv = self.wrapped(*tensors)

        if total_batch_dims == 0:
            rv = rv.squeeze(0)
        else:
            rv = rv.reshape(v.tensor.shape[:total_batch_dims] + rv.shape[1:])

        if self.squeeze is not None:
            assert self.squeeze == -1, 'Only last-dim squeezing is implemented.'
            rv = rv.squeeze(self.squeeze)
        return rv


class QuantizerMode(JacEnum):
    NONE = 'none'
    VQ = 'vq'
    VQ_MULTISTAGE = 'vq-multistage'
    GUMBEL = 'gumbel'


class SimpleQuantizedEncodingModule(nn.Module):
    """A simple feature extractor that supports quantization."""
    def __init__(self, input_dim, hidden_dim, nr_quantizer_states, quantizer_mode='vq-multistage', forward_key=None, q_lambda=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nr_quantizer_states = nr_quantizer_states
        self.quantizer_mode = QuantizerMode.from_string(quantizer_mode)
        self.q_lambda = q_lambda
        self.forward_key = forward_key

        if self.quantizer_mode is QuantizerMode.NONE:
            self.mlp = jacnn.MLPLayer(input_dim, hidden_dim, [], activation='tanh', flatten=False, last_activation=True)
            self.add_module('quantizer', None)
        elif self.quantizer_mode is QuantizerMode.VQ:
            self.mlp = jacnn.MLPLayer(input_dim, hidden_dim, [hidden_dim], activation='tanh', flatten=False, last_activation=False)
            self.quantizer = VectorQuantizer(self.nr_quantizer_states, self.hidden_dim, beta=1.0)
        elif self.quantizer_mode is QuantizerMode.VQ_MULTISTAGE:
            self.mlp = jacnn.MLPLayer(input_dim, hidden_dim, [hidden_dim], activation='tanh', flatten=False, last_activation=False)
            self.quantizer = VectorQuantizer(self.nr_quantizer_states, self.hidden_dim, beta=1.0)
        elif self.quantizer_mode is QuantizerMode.GUMBEL:
            self.mlp = jacnn.MLPLayer(input_dim, hidden_dim, [], activation='tanh', flatten=False, last_activation=True)
            self.quantizer = GumbelQuantizer(self.hidden_dim, self.nr_quantizer_states, self.hidden_dim, hard=False)
        else:
            raise ValueError('Unknown quantizer mode: {}.'.format(self.quantizer_mode))

        self.vq_stage = 1

    def set_vq_stage(self, stage: int):
        assert stage in (1, 2, 3)
        self.vq_stage = stage

    def extra_state_dict(self):
        return {'vq_stage': self.vq_stage}

    def load_extra_state_dict(self, state_dict):
        self.vq_stage = state_dict.pop('vq_stage', 1)

    def embedding_weight(self):
        if self.quantizer is not None:
            return self.quantizer.embedding_weight
        return None

    def quantize(self, z, loss_key):
        ctx = get_forward_context()
        if loss_key is not None:
            ctx.monitor_rms({f'q/{loss_key}/0': z})

        if self.quantizer_mode is QuantizerMode.NONE:
            return z, None
        elif self.quantizer_mode in (QuantizerMode.VQ, QuantizerMode.VQ_MULTISTAGE):
            if self.vq_stage == 1:
                return z, None
            elif self.vq_stage == 2:
                self.quantizer.save_tensor(z)
                return z, None
            else:
                assert self.vq_stage == 3

                if self.training:
                    z, z_id, loss = self.quantizer(z)
                    ctx.add_loss(loss * self.q_lambda, f'loss/q/{loss_key}')
                else:
                    z, z_id = self.quantizer(z)

                ctx.monitor_rms({f'q/{loss_key}/1': z})
                return z, z_id
        elif self.quantizer_mode is QuantizerMode.GUMBEL:
            z, z_id = self.quantizer(z)
            return z, z_id.argmax(dim=-1)
        else:
            raise ValueError('Unknown quantizer mode: {}.'.format(self.quantizer_mode))

    def forward(self, x, loss_key=None):
        if isinstance(x, Value):
            x = x.tensor
        if loss_key is None:
            loss_key = self.forward_key
        if loss_key is None:
            raise ValueError('Should specify either self.forward_key or loss_key when calling the function.')

        z = self.mlp(x)
        z, z_id = self.quantize(z, loss_key)
        return QuantizedTensorValue(z, z_id)


class SimpleQuantizationWrapper(nn.Module):
    """A simple wrapper for mapping functions that supports quantization.

    This function assumes that all inputs to the module are quantized vectors.
    Thus, during training, the actual inputs to the inner module is a list of
    vectors, and the output is either a Boolean value, or a quantized vector.

    Usage:

        >>> module.train()
        >>> # train the module using the standard supervised learning.
        >>> module.eval()
        >>> module.enable_quantization_recorder()
        >>> for data in dataset:
        >>>     _ = module(data)
        >>> module.quantize_mapping_module()  # this ends the quantization encoder and creates the quantized mapping.
        >>> # now use the model in eval mode only, and use quantized values as its input.
    """
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.save_tensors = False
        self.saved_tensors = list()
        self.quantized_mapping = None

    def forward_quantized(self, *args):
        if self.quantized_mapping is not None:
            assert not self.training

            args = [arg.tensor for arg in args]
            output_shape = args[0].shape
            args = [arg.reshape(-1) for arg in args]

            output = list()
            for i in range(args[0].shape[0]):
                argv = tuple(int(arg[i]) for arg in args)
                if argv in self.quantized_mapping:
                    output.append( max(self.quantized_mapping[argv], key=self.quantized_mapping[argv].get) )
                else:
                    output.append(0)
            return torch.tensor(output, dtype=torch.int64).reshape(output_shape)
        raise ValueError('Run quantize_mapping_module first.')

    def forward(self, *args, **kwargs):
        ret = self.wrapped(*args, **kwargs)
        if self.save_tensors:
            self.saved_tensors.append((args, kwargs, ret))
        return ret

    def extra_state_dict(self):
        return {'quantized_mapping': self.quantized_mapping}

    def load_extra_state_dict(self, state_dict):
        if 'quantized_mapping' in state_dict:
            self.quantized_mapping = state_dict['quantized_mapping']

    def enable_quantization_recorder(self):
        assert self.quantized_mapping is None, 'Quantized mapping has already been created. Manually reset it to retrain.'
        self.save_tensors = True
        self.saved_tensors = list()

    def finalize_quantization_recorder(self):
        try:
            self.save_tensors = False
            return self.saved_tensors
        finally:
            self.saved_tensors = list()

    def quantize_mapping_module(self, function_def: FunctionDef):
        """Generate the quantized mapping given the recorded input output pairs.

        As described in the top-level docstring, this function assumes that

            - All arguments are vector-typed and quantized.
            - The output type is either Boolean or quantized vector.

        Based on the recorded IO pairs, it creates a mapping function, stored as a dictionary:
            Mapping[ Sequence[int], Mapping[Union[int, bool], int] ]

        It's a nested mapping. The top level key is the argument values. The second level key is
        the output value (thus, either an integer, or a Boolean value). The value is the number
        of occurrances of the (args, rv) pair.

        Args:
            function_def (FunctionDef): The corresponding function definition of this module.
        """
        saved_tensors = self.finalize_quantization_recorder()

        inputs_quantized = [list() for _ in range(len(function_def.arguments))]
        output_quantized = list()
        for inputs, _, output in saved_tensors:
            for i, (arg, arg_def) in enumerate(zip(inputs, function_def.arguments)):
                assert isinstance(arg_def, VectorValueType) and arg_def.quantized
                inputs_quantized[i].extend(arg.tensor_indices.flatten().tolist())
            if function_def.output_type == BOOL:
                output_quantized.extend((output > 0.5).flatten().tolist())
            else:
                assert isinstance(function_def.output_type, VectorValueType) and function_def.output_type.choices == 0
                output_quantized.extend(output.argmax(-1).flatten().tolist())

        mapping = collections.defaultdict(lambda: collections.defaultdict(int))
        for i in range(len(output_quantized)):
            args = tuple(arg[i] for arg in inputs_quantized)
            output = output_quantized[i]
            mapping[args][output] += 1

        self.quantized_mapping = mapping

    def print_quantized_mapping(self):
        """Prints the quantized mapping table."""
        rows = list()
        for key, values in self.quantized_mapping.items():
            row = list()
            row.extend(key)

            total_counts = sum(values.values())
            normalized = dict()
            for v, c in values.items():
                normalized[v] = c / total_counts
            row.append(' '.join([ f'{v}({c:.4f})' for v, c in sorted(normalized.items()) ]))
            rows.append(row)

        headers = list()
        headers.extend([f'arg{i}' for i in range(len(rows[0][:-1]))])
        headers.append('rv')
        print(tabulate(rows, headers=headers))


class PDSketchMultiStageModel(nn.Module):
    def __init__(self, domain: Domain, stages: Sequence[AugmentedFeatureStage]):
        super().__init__()
        self.domain = domain
        self.stages = stages
        self.functions = nn.ModuleDict()

        self.stage_index = None
        self.vq_stage_index = None

        self.init_networks()
        self.bce = nn.BCELoss()
        self.xent = nn.CrossEntropyLoss()

    def init_networks(self):
        raise NotImplementedError()

    def set_stage_index(self, stage_index, vq_stage_index):
        self.stage_index = stage_index
        self.vq_stage_index = vq_stage_index

        trainable_modules = self.stages[stage_index].all_functions
        for name, module in self.functions.items():
            if name in trainable_modules:
                logger.info('  Unfreeze: {}.'.format(name))
                jactorch.mark_unfreezed(module)
                for subname, submodule in module.named_modules():
                    if isinstance(submodule, SimpleQuantizedEncodingModule):
                        logger.info('    Setting VQ stage for {}.{} to {}.'.format(name, subname, vq_stage_index))
                        submodule.set_vq_stage(vq_stage_index)
            else:
                logger.info('  Freeze: {}.'.format(name))
                jactorch.mark_freezed(module)
                for subname, submodule in module.named_modules():
                    if isinstance(submodule, SimpleQuantizedEncodingModule):
                        logger.info('    Setting VQ stage for {}.{} to {}.'.format(name, subname, 3))
                        submodule.eval()  # do not add loss functions for quantization.
                        submodule.set_vq_stage(3)

    def train(self, mode=True):
        super().train(mode=mode)

        if not mode:
            for module in self.modules():
                if isinstance(module, SimpleQuantizedEncodingModule):
                    module.set_vq_stage(3)

    def init_vq_embeddings(self, feed_dict):
        with torch.no_grad():
            self(feed_dict)
        for function_name in self.stages[self.stage_index].all_functions:
            if function_name in self.functions.keys():
                module = self.functions[function_name]
                for subname, submodule in module.named_modules():
                    if isinstance(submodule, SimpleQuantizedEncodingModule):
                        logger.info('Running init_embeddings for {}.{}.'.format(function_name, subname))
                        submodule.quantizer.init_embeddings()

    def forward(self, feed_dict):
        batch_state1, batch_state2 = feed_dict['state1'].clone(), feed_dict['state2'].clone()

        forward_ctx = ForwardContext(self.training)
        with forward_ctx.as_default():
            batch_state1 = self.domain.forward_features_and_axioms(batch_state1)
            batch_state2 = self.domain.forward_features_and_axioms(batch_state2)
            batch_all_states = concat_batch_states(self.domain, batch_state1, batch_state2)
            batch_all_states_ref = concat_batch_states(self.domain, feed_dict['state1'], feed_dict['state2'])
            for stage_index in range(self.stage_index + 1):
                for supervision in self.stages[stage_index].supervisions:
                    if 'goal' in supervision:
                        pred = batch_all_states[supervision['goal']]
                        target = batch_all_states_ref[supervision['goal']]
                        if self.training and stage_index == self.stage_index:
                            loss = self.bce(pred, target.float())
                            forward_ctx.add_loss(loss, supervision['goal'])
                        forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), supervision['goal'])
                    else:
                        assert 'action' in supervision
                        op = self.domain.operators[supervision['action']]
                        pred, target = self.forward_action(op, supervision['effect_index'], batch_state1, batch_state2, feed_dict[f'action/{op.name}/arguments'])
                        if self.training and stage_index == self.stage_index:
                            loss = self.xent(pred, target.long())
                            forward_ctx.add_loss(loss, supervision['action'] + ':' + str(supervision['effect_index']))
                        forward_ctx.add_accuracy((pred.argmax(-1) == target).float().mean(), supervision['action'] + ':' + str(supervision['effect_index']))

        return forward_ctx.finalize()

    def forward_action(self, operator: Operator, effect_index: int, state1, state2, arguments):
        bounded_variables = dict()
        for i, arg in enumerate(operator.arguments):
            bounded_variables.setdefault(arg.typename, dict())[arg.name] = arguments[i]

        ctx1 = ExpressionExecutionContext(self.domain, state1, bounded_variables)
        ctx2 = ExpressionExecutionContext(self.domain, state2, bounded_variables)
        effect = operator.effects[effect_index].assign_expr

        # TODO: Implement DeicticAssignOp and ConditionalAssignOp.
        assert isinstance(effect, AssignOp)
        output_type = effect.feature.function_def.output_type
        assert isinstance(output_type, VectorValueType) and output_type.quantized
        with ctx1.as_default():
            pred = effect.value.forward(ctx1).tensor
        with ctx2.as_default():
            target = effect.feature.forward(ctx2).tensor_indices
        return pred, target


class PDSketchMBRLModel(nn.Module):
    DEFAULT_OPTIONS = {
        'bptt': False
    }

    def __init__(self, domain: Domain, goal_loss_weight=1.0, action_loss_weight=1.0, **options):
        super().__init__()
        self.domain = domain
        self.functions = nn.ModuleDict()
        self.bce = nn.BCELoss()
        self.xent = nn.CrossEntropyLoss()
        # self.mse = nn.MSELoss(reduction='sum')
        self.mse = nn.SmoothL1Loss(reduction='sum')

        self.goal_loss_weight = goal_loss_weight
        self.action_loss_weight = action_loss_weight
        self.options = options

        for key, value in type(self).DEFAULT_OPTIONS.items():
            self.options.setdefault(key, value)

        self.init_networks(domain)

    def init_networks(self, domain):
        raise NotImplementedError()

    def forward(self, feed_dict, forward_augmented=False):
        forward_ctx = ForwardContext(self.training)
        with forward_ctx.as_default():
            goal_expr = feed_dict['goal_expr']
            states, actions, done = feed_dict['states'], feed_dict['actions'], feed_dict['dones']

            assert forward_augmented
            if forward_augmented:
                for state in states:
                    self.domain.forward_augmented_features(state)

            batch_state = BatchState.from_states(self.domain, states)

            actions: Sequence[OperatorApplier]

            self.domain.forward_derived_features(batch_state)

            if goal_expr is not None:
                pred = self.domain.forward_expr(batch_state, [], goal_expr).tensor
                target = done
                if self.training:
                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target.float())
                    # loss = self.bce(pred, target.float())
                    forward_ctx.add_loss(loss, 'goal', accumulate=self.goal_loss_weight)
                forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), 'goal')

            if 'subgoals' in feed_dict:
                subgoals = feed_dict['subgoals']
                subgoals_done = feed_dict['subgoals_done']

                for i, (subgoal, subgoal_done) in enumerate(zip(subgoals, subgoals_done)):
                    pred = self.domain.forward_expr(batch_state, [], subgoal).tensor
                    target = subgoal_done
                    if self.training:
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target.float())
                        # loss = self.bce(pred, target.float())
                        forward_ctx.add_loss(loss, f'subgoal/{i}', accumulate=self.goal_loss_weight)
                    forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), f'subgoal/{i}')

            if self.action_loss_weight > 0:
                for i, action in enumerate(actions):
                    state = states[i]
                    next_state_pred = action.apply_effect(state)
                    next_state_target = states[i + 1]

                    has_learnable_parameters = False
                    for eff in action.operator.effects:
                        feature_def = eff.unwrapped_assign_expr.feature.feature_def
                    # for feature_def in self.domain.features.values():
                        if feature_def.group != 'augmented':
                            continue

                        has_learnable_parameters = True
                        feature_name = feature_def.name

                        # if action.operator.name == 'pickup':
                        #     print('prev', state[feature_name].tensor[..., -2:])
                        #     print('pred', next_state_pred[feature_name].tensor[..., -2:])
                        #     print('trgt', next_state_target[feature_name].tensor[..., -2:])

                        this_loss = self.mse(
                            input=next_state_pred[feature_name].tensor,
                            target=next_state_target[feature_name].tensor
                        )

                        forward_ctx.add_loss(this_loss, f'a', accumulate=False)
                        forward_ctx.add_loss(this_loss, f'a/{action.operator.name}/{feature_name}', accumulate=self.action_loss_weight)

                        # if action.operator.name.__contains__('pickup') and this_loss.item() > 0.1:
                        #     print('\n' + '-' * 80)
                        #     print(action)
                        #     print('prev', state[feature_name].tensor[..., -10:])
                        #     print('pred', next_state_pred[feature_name].tensor[..., -10:])
                        #     print('trgt', (next_state_target[feature_name].tensor[..., -10:] - next_state_pred[feature_name].tensor[..., -10:]).abs())
                        #     print(this_loss, self.action_loss_weight, 'loss/a/' + action.operator.name + '/' + feature_name, forward_ctx.monitors.raw()['loss/a/' + action.operator.name + '/' + feature_name])

                    if has_learnable_parameters and self.options['bptt']:
                        self.domain.forward_derived_features(next_state_pred)  # forward again the derived features.
                        if goal_expr is not None:
                            pred = self.domain.forward_expr(next_state_pred, [], goal_expr).tensor
                            target = done[i + 1]
                            loss = self.bce(pred, target.float())
                            forward_ctx.add_loss(loss, 'goal_bptt', accumulate=self.goal_loss_weight * 0.1)
                            forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), 'goal_bptt')

                        if 'subgoals' in feed_dict:
                            subgoals = feed_dict['subgoals']
                            subgoals_done = feed_dict['subgoals_done']

                            for j, (subgoal, subgoal_done) in enumerate(zip(subgoals, subgoals_done)):
                                pred = self.domain.forward_expr(next_state_pred, [], subgoal).tensor
                                target = subgoal_done[i + 1]
                                if self.training:
                                    loss = self.bce(pred, target.float())
                                    # loss = self.bce(pred, target.float())
                                    forward_ctx.add_loss(loss, f'subgoal_bptt/{j}', accumulate=self.goal_loss_weight * 0.1)
                                forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), f'subgoal_bptt/{j}')

        return forward_ctx.finalize()
