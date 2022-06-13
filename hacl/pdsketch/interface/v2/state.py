import itertools
from copy import deepcopy

import torch
import jacinle

from jacinle.utils.printing import indent_text, kvformat
from typing import Any, Optional, Union, Tuple, Sequence, Set, Mapping, Dict
from .value import NamedValueType, NamedValueTypeSlot, ObjectType, BOOL, Variable, Value, concat_values
from .optimistic import OPTIM_MAGIC_NUMBER

__all__ = [
    'MultidimensionalArrayInterface', 'ValueDict', 'MixedValueDict',
    'StateTensorAccessor', 'StateLike', 'SingleStateLike', 'State', 'MixedState',
    'BatchState', 'concat_batch_states',
    'TensorDictDefHelper'
]


class MultidimensionalArrayInterface(object):
    """
    A multi-dimensional array inferface. At a high-level, this can be interpreted as a dictionary that maps
    feature names (keys) to multi-diemsntional tensors (value).
    """
    def __init__(self, all_feature_names):
        self.all_feature_names = set(all_feature_names)

    def clone(self):
        raise NotImplementedError()

    def get_feature(self, name: Union[str, int]) -> Value:
        raise NotImplementedError()

    def _set_feature_impl(self, name: str, feature: Value):
        raise NotImplementedError()

    def set_feature(self, name: str, feature: Value):
        if name not in self.all_feature_names:
            self.all_feature_names.add(name)
        self._set_feature_impl(name, feature)

    def update_feature(self, other_tensor_dict: Mapping[str, Value]):
        for key, value in other_tensor_dict.items():
            self.set_feature(key, value)

    def __contains__(self, item: str) -> bool:
        return item in self.all_feature_names

    def __getitem__(self, name: str) -> Value:
        return self.get_feature(name)

    def __setitem__(self, key, value):
        return self.set_feature(key, value)

    def keys(self):
        yield from self.all_feature_names

    def values(self):
        for key in self.all_feature_names:
            yield self.get_feature(key)

    def items(self):
        for key in self.all_feature_names:
            yield key, self.get_feature(key)


class ValueDict(MultidimensionalArrayInterface):
    """Basic tensor dict implementation."""
    def __init__(self, tensor_dict: Optional[Dict[str, Value]] = None):
        if tensor_dict is None:
            tensor_dict = dict()

        all_feature_names = set(tensor_dict.keys())
        super().__init__(all_feature_names)
        self.tensor_dict = tensor_dict

    def clone(self):
        return type(self)({k: v.clone() for k, v in self.tensor_dict.items()})

    def get_feature(self, name: Union[str, int]):
        return self.tensor_dict[name]

    def _set_feature_impl(self, name, feature: Value):
        self.tensor_dict[name] = feature

    def __contains__(self, item):
        return item in self.tensor_dict


class MixedValueDict(MultidimensionalArrayInterface):
    """For MixedState only. """

    # TODO:: Implement this.
    def __init__(self, feature_dict: Optional[Mapping[str, Union[Mapping[str, Set[int]], Set[int]]]] = None):
        if feature_dict is None:
            feature_dict = dict()

        all_feature_names = list(feature_dict.keys())
        super().__init__(all_feature_names)
        self.feature_dict = feature_dict
        raise NotImplementedError()

    def clone(self):
        return type(self)(deepcopy(self.feature_dict))

    def index(self, feature_name: Union[str, int], indices: Sequence[int]) -> Set[int]:
        assert isinstance(feature_name, str)
        if len(indices) == 0:
            return self.feature_dict[feature_name]
        elif len(indices) == 1:
            indices = indices[0]
            return self.feature_dict[feature_name][indices]
        else:
            return self.feature_dict[feature_name][tuple(indices)]

    def set_index(self, feature_name: Union[str, int], indices: Sequence[int], value: int):
        if feature_name not in self.feature_dict:
            self.feature_dict[feature_name] = set() if len(indices) == 0 else dict()

        if len(indices) == 0:
            self.feature_dict[feature_name].add(value)
        else:
            if len(indices) == 1:
                indices = indices[0]
            if indices not in self.feature_dict[feature_name]:
                self.feature_dict[feature_name][indices] = set()
            self.feature_dict[feature_name][tuple(indices)].add(value)

    def get_feature(self, name: Union[str, int]):
        return self.feature_dict[name]

    def _set_feature_impl(self, name, feature):
        raise NotImplementedError()


class StateTensorAccessor(object):
    def __init__(self, state):
        self.state = state

    def __getitem__(self, item):
        return self.state.features[item].tensor


class StateLike(object):
    @property
    def batch_dims(self) -> int:
        raise NotImplementedError()

    def get_typename(self, name):
        raise NotImplementedError()

    def get_typed_index(self, name):
        raise NotImplementedError()

    def get_nr_objects_by_type(self,typename):
        raise NotImplementedError()

    @property
    def features(self) -> MultidimensionalArrayInterface:
        raise NotImplementedError()

    @property
    def tensors(self) -> StateTensorAccessor:
        return StateTensorAccessor(self)

    def clone(self):
        raise NotImplementedError()

    def index(self, feature_name, arguments):
        return self.features[feature_name][arguments]

    def set_index(self, feature_name, arguments, value):
        self.features[feature_name][arguments] = value

    def get_feature(self, name: str):
        return self.features[name]

    def __getitem__(self, name: str):
        return self.get_feature(name)

    def __str__(self):
        raise NotImplementedError()

    __repr__ = jacinle.repr_from_str


class SingleStateLike(StateLike):
    def __init__(self, object_types: Sequence[ObjectType], features: Optional[MultidimensionalArrayInterface] = None, object_names: Optional[Sequence[str]] = None):
        self._object_types = object_types
        self._features = features
        self._object_names = object_names

        if self._features is None:
            self._features = ValueDict()

        if self._object_names is not None:
            assert len(self._object_names) == len(self._object_types)
            self._object_type2name = dict()
            self._object_name2index = dict()
            for name, type in zip(self._object_names, self._object_types):
                typename = type.typename
                if typename not in self._object_type2name:
                    self._object_type2name[typename] = list()
                self._object_name2index[name] = (typename, len(self._object_type2name[typename]))
                self._object_type2name[typename].append(name)
        else:
            self._object_name2index = dict()
            self._object_type2name = dict()
        self.internals = dict()

    @property
    def batch_dims(self) -> int:
        """The number of batchified dimensions. For the basic State, it should be 0."""
        return 0

    @property
    def nr_objects(self) -> int:
        """The number of objects in the current state."""
        return len(self._object_types)

    @property
    def object_types(self) -> Sequence[ObjectType]:
        """A list of object types."""
        return self._object_types

    @property
    def object_names(self) -> Sequence[str]:
        """A lsit of object names."""
        return self._object_names

    @property
    def object_type2name(self) -> Mapping[str, Sequence[str]]:
        """Return a mapping from typename (in string) to a list of objects of this type. For example:
            `state.object_type2name['location'][i]` returns the object name of the i-th location in the state.

        Returns:
            Mapping[str, Sequence[str]]: the mapping.
        """
        return self._object_type2name

    @property
    def object_name2index(self) -> Mapping[str, Tuple[str, int]]:
        """Return a mapping from the object name to a tuple of (typename, the index under that typename).
            That is, `state.object_name2index[name] == (typename, index)` iff. `state.object_type2name[typename][index] = nam.e`
        """
        return self._object_name2index

    def get_typename(self, name):
        return self._object_name2index[name][0]

    def get_typed_index(self, name):
        return self._object_name2index[name][1]

    def get_nr_objects_by_type(self, typename):
        return len(self.object_type2name[typename])

    @property
    def features(self):
        return self._features

    def clone(self):
        return type(self)(self._object_types, self._features.clone(), self._object_names)

    def compose_bounded_variables(self, arguments_def: Sequence[Variable], arguments_value: Sequence[Union[str, int, torch.Tensor]]):
        assert len(arguments_def) == len(arguments_value)
        bounded_variables = dict()
        for d, v in zip(arguments_def, arguments_value):
            if isinstance(d.type, ObjectType):
                if isinstance(v, int):
                    v = self.object_names[v]
                typename, typed_index = self.object_name2index[v]
                assert d.type.typename == typename
                bounded_variables.setdefault(typename, {})[d.name] = typed_index
            else:
                assert isinstance(d.type, NamedValueType)
                if isinstance(v, NamedValueTypeSlot):  # partially grounded actions.
                    v = Value(d.type, [], torch.tensor(OPTIM_MAGIC_NUMBER, dtype=torch.int64), quantized=True)
                if not isinstance(v, Value):
                    v = Value(d.type, [], v, quantized=False)
                bounded_variables.setdefault(d.type.typename, {})[d.name] = v
        return bounded_variables

    def __str__(self):
        if self.object_names is not None:
            objects_str = [f'{name} - {type.typename}' for name, type in zip(self.object_names, self.object_types)]
        else:
            objects_str = self.object_names
        fmt = f'''{type(self).__name__}{{
  states:
'''
        for p in self.features.all_feature_names:
            tensor = self.features[p]
            fmt += f'    - {p}'
            fmt += ': ' + indent_text(str(tensor), level=2).strip() + '\n'
        fmt += f"  objects: {', '.join(objects_str)}\n"
        fmt += self.extra_state_str()
        fmt += '}'
        return fmt

    def extra_state_str(self):
        return ''


class State(SingleStateLike):
    def clone(self):
        rv = type(self)(self._object_types, self._features.clone(), self._object_names)
        rv.internals = self.clone_internals()
        return rv

    def clone_internals(self):
        return dict()

    def make_quantized(self, domain, features=None):
        assert isinstance(self.features, ValueDict), 'Only TensorDict is supported for automatic quantization.'

        if features is None:
            features = [name for name in self.features.all_feature_names if not domain.feature_in_group(name, ['augmented-input'])]

        new_tensor_dict = ValueDict()
        for feature_name in features:
            new_tensor_dict[feature_name] = self.features[feature_name].make_quantized()
        return type(self)(self.object_types, new_tensor_dict, self.object_names)

    def define_context(self, domain) -> 'TensorDictDefHelper':
        return TensorDictDefHelper(domain, self)

    def generate_tuple_description(self, domain):
        assert isinstance(self.features, ValueDict), 'Only TensorDict is supported for automatic tuple description.'

        rv = list()
        for feature_name in sorted(self.features.all_feature_names):
            if domain.feature_in_group(feature_name, ['basic', 'augmented']):
                feature = self.features[feature_name]
                assert feature.quantized, 'Can only generate tuple description for quantized states.'
                rv.extend(self.features[feature_name].tensor.flatten().tolist())
        return tuple(rv)


class MixedState(SingleStateLike):
    # TODO: Implement this.
    def __init__(self, object_types, features, object_names):
        super().__init__(object_types, features, object_names)
        raise NotImplementedError()

    @classmethod
    def from_state(cls, domain, state: State):
        state = state.make_quantized(domain)
        assert isinstance(state.features, ValueDict), 'Only TensorDict is supported for MixedState.'
        new_feature_dict = dict()
        for feature_name in state.features.all_feature_names:
            feature_def = domain.features[feature_name]

            if len(feature_def.arguments) == 0:
                new_feature_dict[feature_name] = {state.features[feature_name].item()}
            else:
                new_feature_dict[feature_name] = dict()
                for index, value in _iter_tensor_indices_and_values(state.features[feature_name]):
                    new_feature_dict[feature_name][index] = {value.item()}
        return cls(state.object_types, MixedValueDict(new_feature_dict), object_names=state.object_names)


class BatchState(StateLike):
    def __init__(self, nr_objects_per_type, features, max_nr_objects_per_type=None, object_name2index=None):
        self._nr_objects_per_type = nr_objects_per_type
        self._max_nr_objects_per_type = max_nr_objects_per_type
        self._object_name2index = object_name2index
        self._features = features

        if self._max_nr_objects_per_type is None:
            self._max_nr_objects_per_type = {key: value.max().item() for key, value in nr_objects_per_type.items()}

    @classmethod
    def from_states(cls, domain, states: Sequence[State]):
        all_typenames = list(domain.types.keys())
        all_features = list(states[0].features.all_feature_names)

        # 1. Sanity checks.
        for state in states[1:]:
            assert len(all_features) == len(state.features.all_feature_names)
            for feature in all_features:
                assert feature in state.features.all_feature_names

        # 2. Construct the nr_objects_pre_type dict.
        nr_objects_per_type = {typename: list() for typename in all_typenames}
        for state in states:
            for typename in all_typenames:
                if typename in state.object_type2name:
                    nr_objects_per_type[typename].append(len(state.object_type2name[typename]))
                else:
                    nr_objects_per_type[typename].append(0)

        # 3. Compute the max_nr_objects_per_type.
        for typename, nr_list in nr_objects_per_type.items():
            nr_objects_per_type[typename] = torch.tensor(nr_list, dtype=torch.int64)
        max_nr_objects_per_type = {key: value.max().item() for key, value in nr_objects_per_type.items()}

        # 4. Put the same feature into a list.
        features = {feature_name: list() for feature_name in all_features}
        for state in states:
            assert isinstance(state.features, ValueDict), 'Only TensorDict is implemented for BatchState.from_states.'
            for key, value in state.features.tensor_dict.items():
                features[key].append(value)

        # 5. Actually, compute the features.
        feature_names = list(features.keys())
        for feature_name in feature_names:
            features[feature_name] = concat_values(*features[feature_name])
        return cls(nr_objects_per_type, ValueDict(features), max_nr_objects_per_type=max_nr_objects_per_type, object_name2index=[state.object_name2index for state in states])

    @property
    def batch_dims(self) -> int:
        return 1

    @property
    def quantized(self):
        return False

    @property
    def batch_size(self) -> int:
        x = next(iter(self.features.tensor_dict.values()))
        return x.shape[0]

    @property
    def nr_objects_per_type(self):
        return self._nr_objects_per_type

    @property
    def max_nr_objects_per_type(self):
        return self._max_nr_objects_per_type

    @property
    def features(self):
        return self._features

    def clone(self):
        object_name2index = self.object_name2index.copy() if self.object_name2index is not None else None
        return type(self)(self.nr_objects_per_type.copy(), self.features.clone(), max_nr_objects_per_type=self.max_nr_objects_per_type.copy(), object_name2index=object_name2index)

    @property
    def object_name2index(self) -> Mapping[str, Tuple[str, int]]:
        """Return a mapping from the object name to a tuple of (typename, the index under that typename).
            That is, `state.object_name2index[name] == (typename, index)` iff. `state.object_type2name[typename][index] == name`.
        """
        return self._object_name2index

    def get_typename(self, name):
        return [record[name][0] for record in self._object_name2index]

    def get_typed_index(self, name):
        return [record[name][1] for record in self._object_name2index]

    def get_nr_objects_by_type(self, typename) -> Sequence[int]:
        return self._nr_objects_per_type[typename]

    def __str__(self):
        fmt = f'''{type(self).__name__}{{
  nr_objects_per_type:
'''
        for typename, number in self.nr_objects_per_type.items():
            fmt += f'    - {typename}: {number.tolist()}\n'
        fmt += '''
  states:
'''
        fmt += indent_text(kvformat(self.features.tensor_dict), 2, tabsize=2).rstrip() + '\n'
        fmt += '}'
        return fmt

    __repr__ = jacinle.repr_from_str


def concat_batch_states(*args: BatchState):
    all_typenames = list(args[0].nr_objects_per_type)
    all_features = list(args[0].features.all_feature_names)

    nr_objects_per_type = {typename: list() for typename in args[0].nr_objects_per_type}
    for arg in args:
        for typename in all_typenames:
            nr_objects_per_type[typename].append(arg.nr_objects_per_type[typename])

    for typename, nr_list in nr_objects_per_type.items():
        nr_objects_per_type[typename] = torch.cat(nr_list, dim=0)
    max_nr_objects_per_type = {key: value.max().item() for key, value in nr_objects_per_type.items()}

    features = {feature_name: list() for feature_name in all_features}
    for arg in args:
        assert isinstance(arg.features, ValueDict), 'Only TensorDict is implemented for BatchState.from_states.'
        for key, value in arg.features.tensor_dict.items():
            features[key].append(value)
    for key, values in features:
        features[key] = concat_values(*values)

    return BatchState(
        nr_objects_per_type,
        ValueDict(features),
        max_nr_objects_per_type=max_nr_objects_per_type,
        object_name2index=sum([state.object_name2index for state in args], start=[])
    )


class _TensorDefPredicate(object):
    def __init__(self, predicate_def, arguments):
        self.predicate_def = predicate_def
        self.arguments = arguments


class _TensorDefPredicateApplier(object):
    def __init__(self, predicate_def):
        self.predicate_def = predicate_def

    def __call__(self, *args):
        return _TensorDefPredicate(self.predicate_def, args)


class TensorDictDefHelper(object):
    def __init__(self, domain, state: State):
        self.domain = domain
        self.state = state

    def get_pred(self, name):
        if name in self.domain.features:
            return _TensorDefPredicateApplier(self.domain.features[name])
        elif name.replace('_', '-') in self.domain.features:
            return _TensorDefPredicateApplier(self.domain.features[name.replace('_', '-')])
        else:
            raise NotImplementedError('Unknown predicate: {}.'.format(name))

    def __getattr__(self, name):
        return self.get_pred(name)

    def define_predicates(self, predicates: Sequence[_TensorDefPredicate]):
        for predicate_def in self.domain.predicates.values():
            if predicate_def.name in self.state.features.all_feature_names:
                continue
            if predicate_def.cacheable and predicate_def.output_type == BOOL:
                sizes = list()
                for arg_def in predicate_def.arguments:
                    sizes.append(len(self.state.object_type2name[arg_def.typename]))
                self.state.features[predicate_def.name] = Value(
                    BOOL, [var.name for var in predicate_def.arguments],
                    torch.zeros(sizes, dtype=torch.int64),
                    batch_dims=0, quantized=True
                )

        for pred in predicates:
            assert isinstance(pred, _TensorDefPredicate)
            assert pred.predicate_def.output_type == BOOL
            name = pred.predicate_def.name
            arguments = [self.state.get_typed_index(arg) for arg in pred.arguments]
            self.state.features[name].tensor[tuple(arguments)] = 1

    def define_feature(self, feature_name, tensor_or_mapping, quantized=False):
        feature_def = self.domain.features[feature_name]
        sizes = list()
        for arg_def in feature_def.arguments:
            sizes.append(len(self.state.object_type2name[arg_def.typename]))
        sizes = tuple(sizes)
        if torch.is_tensor(tensor_or_mapping):
            self.state.features[feature_name] = Value(
                feature_def.output_type, [var.name for var in feature_def.arguments],
                tensor_or_mapping, batch_dims=0, quantized=quantized
            )
        else:
            if not quantized:
                tensor = torch.zeros(sizes + feature_def.output_type.size_tuple())
            else:
                tensor = torch.zeros(sizes, dtype=torch.int64)

            for key, value in tensor_or_mapping.items():
                if isinstance(key, tuple):
                    args = [self.state.get_typed_index(arg) for arg in key]
                else:
                    assert isinstance(key, str)
                    args = [self.state.get_typed_index(key)]
                tensor[tuple(args)] = value
            self.state.features[feature_name] = Value(
                feature_def.output_type, [var.name for var in feature_def.arguments],
                tensor, batch_dims=0, quantized=quantized
            )


def _iter_tensor_indices_and_values(tensor):
    tensor_flatten = tensor.reshape(-1, )
    if len(tensor.shape) > 1:
        indices = [range(x) for x in tensor.shape]
        yield from zip(itertools.product(*indices), tensor_flatten)
    else:
        yield from zip(range(tensor.shape[0]), tensor_flatten)
