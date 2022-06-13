import torch
import torch.nn.functional as F

import jacinle
import jactorch

from jacinle.utils.printing import indent_text
from typing import Any, Optional, Union, Tuple, List, Sequence, Set, Mapping

__all__ = [
    'ObjectType',
    'ValueType', 'NamedValueType', 'NamedValueTypeSlot',
    'BasicValueType', 'BOOL', 'INT64', 'FLOAT32', 'VectorValueType',
    'is_intrinsically_quantized', 'RUNTIME_BINDING',
    'Variable', 'StringConstant',
    'QINDEX', 'QuantizedTensorValue', 'Value', 'wrap_value', 'unwrap_value', 'concat_values',
    'MixedValue'
]


class ObjectType(object):
    def __init__(self, typename: str, parent_name: Optional[str] = None):
        self.typename = typename
        self.parent_name = parent_name

    def __str__(self):
        return f'T::{self.typename}'

    def __eq__(self, o: object) -> bool:
        if o == RUNTIME_BINDING:
            return True
        if isinstance(o, ObjectType):
            raise self.typename == o.typename
        return False

    def __ne__(self, o: object) -> bool:
        return not (self == o)


class ValueType(object):
    def ndim(self) -> int:
        raise NotImplementedError()

    def size(self) -> int:
        raise NotImplementedError()

    def size_tuple(self):
        raise NotImplementedError()

    def assignment_type(self) -> 'ValueType':
        return self

    def __str__(self):
        raise NotImplementedError()

    __repr__ = jacinle.repr_from_str

    def __eq__(self, o: object) -> bool:
        raise NotImplementedError()

    def __ne__(self, o: object) -> bool:
        return not (self == o)


class NamedValueType(ValueType):
    def __init__(self, typename: str, base_type: ValueType):
        self.typename = typename
        self.base_type = base_type

    def ndim(self):
        return self.base_type.ndim()

    def size(self):
        return self.base_type.size()

    def size_tuple(self):
        return self.base_type.size_tuple()

    def assignment_type(self):
        return self

    def __str__(self):
        return f'T::{self.typename}'

    def __eq__(self, o: object) -> bool:
        if isinstance(o, NamedValueType):
            return self.typename == o.typename
        return False


class NamedValueTypeSlot(object):
    def __init__(self, type):
        self.type = type

    def __str__(self):
        return '!' + str(self.type)


class BasicValueType(ValueType):
    def __init__(self, typename):
        assert typename in ('int32', 'int64', 'float32', 'float64', 'bool'), 'Unknown basic value type: {}.'.format(typename)
        self.typename = typename

    def __str__(self):
        return self.typename

    def ndim(self):
        return 0

    def size(self):
        return 1

    def size_tuple(self):
        return tuple()

    def __eq__(self, o: object) -> bool:
        if isinstance(o, BasicValueType):
            return self.typename == o.typename
        return False


BOOL = BasicValueType('bool')
INT64 = BasicValueType('int64')
FLOAT32 = BasicValueType('float32')


class VectorValueType(ValueType):
    def __init__(self, dtype: BasicValueType, dim: int, choices: int, factors: Optional[int] = 1):
        assert isinstance(dtype, BasicValueType), 'Does not support high-order vectors.'
        self.dtype = dtype
        self.dim = dim
        self.choices = choices
        self.factors = factors

    def __str__(self):
        return f'vector[{self.dtype}, dim={self.dim}, choices={self.choices}, factors={self.factors}]'

    @property
    def quantized(self):
        return self.choices > 0

    def ndim(self):
        return 1

    def size(self):
        return self.dim * self.factors

    def size_tuple(self):
        return (self.size(), )

    def assignment_type(self) -> 'ValueType':
        if self.quantized:
            return VectorValueType(FLOAT32, self.choices, 0, self.factors)
        return self

    def __eq__(self, o: object) -> bool:
        if isinstance(o, VectorValueType):
            return self.dtype == o.dtype and self.dim == o.dim and self.choices == o.choices and self.factors == o.factors
        return False


def is_intrinsically_quantized(type: ValueType):
    return type == BOOL or type == INT64 or (isinstance(type, NamedValueType) and is_intrinsically_quantized(type.base_type))


class _RuntimeBindingType(object):
    pass


RUNTIME_BINDING = _RuntimeBindingType()


class _Placeholder(object):
    def __init__(self, name: str, type: Optional[Union[ObjectType, NamedValueType]] = None):
        self.name = name
        self.type = type

    @property
    def typename(self):
        return self.type.typename

    def __str__(self):
        if self.type is None:
            return self.name
        else:
            if isinstance(self.type, ObjectType):
                return f'{self.name} - {self.type.typename}'
            else:
                return f'{self.name} - {self.type}'

    __repr__ = jacinle.repr_from_str


class Variable(_Placeholder):
    pass


class StringConstant(_Placeholder):
    pass


QINDEX = slice(None)


class QuantizedTensorValue(object):
    def __init__(self, value, indices, mask):
        self.value = value
        self.indices = indices
        self.mask = mask


class Value(object):
    def __init__(
        self,
        dtype: ValueType,
        batch_variables: Union[Sequence[str], int],
        tensor: Union[torch.Tensor, 'QuantizedTensorValue'],
        batch_dims: Optional[int] = 0,
        quantized: Optional[bool] = False,
        *, _check_tensor: Optional[bool] = False, _mask_certified_flag: Optional[bool] = True
    ):
        """Instantiate a Value object for storing intermediate computation results.

        The tensor is assumed to have the following layout:

            tensor[B1, B2, B3, ..., V1, V2, V3, ..., D1, D2, D3, ...]

            - The first `batch_dims` dimensions are "batch".
            - The next `len(batch_variables)` dimensions are "variables".
            - The next `dtype.ndim()` dimensions are data dimensions (e.g., images, vectors).
            - A special case is that `dtype.ndim()` can be zero (scalar).

        Args:
            dtype (ValueType): The data type of the Value object.
            batch_variables (Sequence[str], int): A sequence of variables that are processed in "batch." This typically corresponds to "quantified variables."
                It can also be a single integer, indicating the number of batched variables.
            tensor (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The actual tensor
            batch_dims (int, optional): The additional batch dimensions at the beginning. Defaults to 0.
            quantized (bool, optional): Whether the values in self.tensor is quantized.
            _check_tensor (bool, optional): internal flag, whether to run the tensor shape/type sanity check.
            _mask_certified_flag (bool, optional): internal flag, indicating whether self.tensor_mask is guaranteed to be the correct mask.
                This flag will be marked false when we do expand_as.
        """
        self.dtype = dtype
        self.batch_dims = batch_dims
        if isinstance(batch_variables, int):
            batch_variables = tuple([f'?{i}' for i in range(batch_variables)])
        self.batch_variables = tuple(batch_variables)
        self.quantized = quantized

        if isinstance(tensor, QuantizedTensorValue):
            self.tensor = tensor.value
            self.tensor_indices = tensor.indices
            self.tensor_mask = tensor.mask
        else:
            self.tensor = tensor
            self.tensor_indices = None
            self.tensor_mask = None

        if not torch.is_tensor(self.tensor):
            self.tensor = torch.tensor(self.tensor)

        if _check_tensor:
            self._check_tensor('tensor')
            self._check_tensor('tensor_indices', is_index=True)
            self._check_tensor('tensor_mask', is_index=True)
        self._mask_certified_flag = _mask_certified_flag

        self._backward_function = None
        self._backward_args = tuple()
        self.tensor_grad = None

    def set_backward_function(self, function, *args, **kwargs):
        self._backward_function = function
        self._backward_args = args, kwargs

    def backward(self):
        assert self.total_batch_dims == 0 and self.quantized
        self.tensor_grad = torch.ones_like(self.tensor)
        self.backward_inner()

    def backward_inner(self):
        assert self._backward_function is not None
        assert self._backward_function is not NotImplemented
        self._backward_function(self, *self._backward_args[0], **self._backward_args[1])

    def backward_compiled(self, ctx):
        """backward_v2 only works for compiled simple (conjunctive) preconditions
        and effects. This feature is used in the HFFHeuristic in v2.ai_heuristic.
        """
        assert self._backward_function is not None
        assert self._backward_function is not NotImplemented
        self._backward_function(ctx, *self._backward_args[0], **self._backward_args[1])

    def _check_tensor(self, tensor_name: str, is_index: Optional[bool] = False):
        tensor = getattr(self, tensor_name)
        if tensor is None:
            return
        try:
            if isinstance(self.dtype, NamedValueType):
                dtype = self.dtype.base_type
            else:
                dtype = self.dtype

            if isinstance(dtype, BasicValueType):
                if self.total_batch_dims == 0:
                    if dtype in (BOOL, INT64):
                        if self.quantized:
                            assert isinstance(tensor, (bool, int, torch.Tensor))
                            if isinstance(tensor, (bool, int)):
                                setattr(self, tensor_name, torch.tensor(tensor, dtype=torch.int64))
                            else:
                                if dtype == BOOL and tensor.dtype == torch.bool:
                                    setattr(self, tensor_name, tensor.to(torch.int64))
                                else:
                                    assert tensor.dtype == torch.int64
                        else:
                            assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                    else:
                        raise TypeError('Unsupported dtype: {}.'.format(self.dtype))
                else:
                    assert torch.is_tensor(tensor)
                    assert tensor.ndim == len(self.batch_variables) + self.batch_dims
                    if dtype in (BOOL, INT64):
                        if self.quantized:
                            if dtype == BOOL and tensor.dtype == torch.bool:
                                setattr(self, tensor_name, tensor.to(torch.int64))
                            else:
                                assert tensor.dtype == torch.int64
                        else:
                            assert tensor.dtype == torch.float32
                    else:
                        raise TypeError('Unsupported dtype: {}.'.format(self.dtype))
            elif isinstance(dtype, VectorValueType):
                assert torch.is_tensor(tensor)
                if self.quantized or is_index:
                    assert tensor.ndim == len(self.batch_variables) + self.batch_dims
                else:
                    assert tensor.ndim == len(self.batch_variables) + dtype.ndim() + self.batch_dims
            else:
                raise NotImplementedError('Unsupported dtype: {}.'.format(self.dtype))
        except AssertionError as e:
            axes = ', '.join(self.batch_variables)
            raise ValueError(f'Tensor {tensor_name} shape/dtype mismatch for Value[{self.dtype}, axes=[{axes}], batch_dims={self.batch_dims}, tdtype={tensor.dtype}, tdshape={tuple(tensor.shape)}]') from e

    def clone(self, clone_tensor=True):
        if clone_tensor:
            tensor = QuantizedTensorValue(_clone_tensor(self.tensor), _clone_tensor(self.tensor_indices), _clone_tensor(self.tensor_mask))
        else:
            tensor = QuantizedTensorValue(self.tensor, self.tensor_indices, self.tensor_mask)
        return type(self)(
            self.dtype, self.batch_variables, tensor, self.batch_dims, self.quantized,
            _check_tensor=False, _mask_certified_flag=self._mask_certified_flag
        )

    def rename_batch_variables(self, new_variables: Sequence[str], clone=False):
        assert len(self.batch_variables) == len(new_variables)
        rv = self.clone() if clone else self
        rv.batch_variables = tuple(new_variables)
        return rv

    def make_quantized(self):
        if self.quantized:
            return self.clone()

        assert is_intrinsically_quantized(self.dtype) or self.tensor_indices is not None

        if self.tensor_indices is not None:
            rv = self.tensor_indices
        elif self.dtype == BOOL:
            rv = (self.tensor > 0.5).to(torch.int64)
        elif self.dtype == INT64:
            rv = self.tensor.to(torch.int64)
        elif isinstance(self.dtype, NamedValueType) and self.dtype.base_type == INT64:
            rv = self.tensor.to(torch.int64)
        else:
            raise TypeError('Unable to quantize value. Need either tensor_indices, or intrinsically quantized dtypes: {}.'.format(str(self)))

        rv = QuantizedTensorValue(rv, None, self.tensor_mask)
        return type(self)(self.dtype, self.batch_variables, rv, self.batch_dims, quantized=True, _mask_certified_flag=self._mask_certified_flag)

    @property
    def total_batch_dims(self):
        return self.batch_dims + len(self.batch_variables)

    @property
    def nr_variables(self):
        return len(self.batch_variables)

    def get_variable_size(self, variable_name_or_index: Union[str, int]) -> int:
        if isinstance(variable_name_or_index, str):
            variable_name_or_index = self.batch_variables.index(variable_name_or_index)
        return self.tensor.size(variable_name_or_index + self.batch_dims)

    def has_optimistic_value(self):
        if self.quantized:
            from .optimistic import is_optimistic_value
            return is_optimistic_value(self.tensor).any()
        return False

    def item(self):
        assert self.total_batch_dims == 0
        if self.quantized:
            return int(self.tensor.item())
        return self.tensor.item()

    def __bool__(self):
        if self.dtype == BOOL:
            return bool(self.item())
        else:
            raise TypeError('Cannot convert Value object {} into bool.'.format(self))

    def index(self, indices: Optional[Tuple[Union[int, torch.Tensor], ...]]) -> 'Value':
        assert indices is None or isinstance(indices, tuple)
        if indices is None:
            return self.clone()

        assert len(indices) == self.nr_variables
        batch_variables = list()
        for i, idx in enumerate(indices):
            if idx == QINDEX:
                batch_variables.append(self.batch_variables[i])
        if self.batch_dims == 0:
            return type(self)(self.dtype, batch_variables, QuantizedTensorValue(
                self.tensor[indices] if self.tensor is not None else None,
                self.tensor_indices[indices] if self.tensor_indices is not None else None,
                self.tensor_mask[indices] if self.tensor_mask is not None else None
            ), batch_dims=self.batch_dims, quantized=self.quantized, _mask_certified_flag=self._mask_certified_flag)
        elif self.batch_dims == 1:
            indices = (jactorch.batch, ) + indices
            return type(self)(self.dtype, batch_variables, QuantizedTensorValue(
                jactorch.bvindex(self.tensor)[indices][0] if self.tensor is not None else None,
                jactorch.bvindex(self.tensor_indices)[indices][0] if self.tensor_indices is not None else None,
                jactorch.bvindex(self.tensor_mask)[indices][0] if self.tensor_mask is not None else None
            ), batch_dims=self.batch_dims, quantized=self.quantized, _mask_certified_flag=self._mask_certified_flag)
        else:
            raise NotImplementedError('Unsupported batched dims: {}.'.format(self.batch_dims))

    def set_index(self, indices: Optional[Tuple[Union[int, torch.Tensor], ...]], value: Union[bool, int, float, torch.Tensor, QuantizedTensorValue, 'Value']):
        assert indices is None or isinstance(indices, tuple)
        assert isinstance(value, (bool, int, float, torch.Tensor, QuantizedTensorValue, Value))
        if isinstance(value, Value):
            value = QuantizedTensorValue(value.tensor, value.tensor_indices, value.tensor_mask)

        value_indices = None
        if isinstance(value, (bool, int, float)):
            value = torch.tensor(value, dtype=self.tensor.dtype, device=self.tensor.device)
        if isinstance(value, QuantizedTensorValue):
            value, value_indices = value.value, value.indices

        if self.batch_dims == 0:
            if indices is None or len(indices) == 0:
                self.tensor = value
                self.tensor_indices = value_indices
            else:
                # We have to this cloning. Consider the following case:
                # v[0] = v[1]
                indices = indices[0] if len(indices) == 1 else indices

                self.tensor = self.tensor.clone()
                self.tensor[indices] = value

                if self.tensor_indices is not None:
                    assert value_indices is not None
                    self.tensor_indices[indices] = value_indices
        else:
            raise NotImplementedError('Unsupported batched dims: {}.'.format(self.batch_dims))

    def __getitem__(self, item):
        return self.index(item)

    def __setitem__(self, key, value):
        self.set_index(key, value)

    def expand_as(self, other: 'Value') -> 'Value':
        return self.expand(other.batch_variables, other.tensor.size()[
            other.batch_dims : other.batch_dims + len(other.batch_variables)
        ])

    def expand(self, batch_variables: Sequence[str], batch_sizes: Sequence[int]) -> 'Value':
        # if tuple(batch_variables) == self.batch_variables:
        #     return self

        data = self.tensor
        data_indices = self.tensor_indices
        data_mask = self.tensor_mask

        current_batch_variables = list(self.batch_variables)
        for var in batch_variables:
            if var not in current_batch_variables:
                data = data.unsqueeze(self.batch_dims)
                if data_indices is not None:
                    data_indices = data_indices.unsqueeze(self.batch_dims)
                if data_mask is not None:
                    data_mask = data_mask.unsqueeze(self.batch_dims)
                current_batch_variables.insert(0, var)

        indices = list()
        sizes = list()

        # process the first "batch" dims.
        for i in range(self.batch_dims):
            indices.append(i)
            sizes.append(data.size(i))

        # process the next "variables" dims.
        for var, size in zip(batch_variables, batch_sizes):
            indices.append(current_batch_variables.index(var) + self.batch_dims)
            sizes.append(size)

        if data_indices is not None:
            # corner case for "scalar" storage.
            if len(indices) > 0:
                data_indices = data_indices.permute(indices)
            data_indices = data_indices.expand(sizes)

        if data_mask is not None:
            # corner case for "scalar" storage.
            if len(indices) > 0:
                data_mask = data_mask.permute(indices)
            data_mask = data_mask.expand(sizes)

        if not self.quantized:
            # process the last "data" dims.
            for i in range(self.dtype.ndim()):
                indices.append(self.batch_dims + len(batch_variables))
                sizes.append(data.size(self.batch_dims + len(batch_variables) + i))

        # corner case for "scalar" storage.
        if len(indices) > 0:
            data = data.permute(indices)
        data = data.expand(sizes)

        value = Value(
            self.dtype, batch_variables,
            QuantizedTensorValue(data, data_indices, data_mask),
            batch_dims=self.batch_dims, quantized=self.quantized,
            _mask_certified_flag=False
        )

        if self._backward_function is not None:
            value.set_backward_function(self._backward_expand, self)
        return value

    def _backward_expand(self, output_value: 'Value', input_value: 'Value'):
        output_variables = list(output_value.batch_variables)
        input_variables = list(input_value.batch_variables)

        grad = output_value.tensor_grad
        reduce_dims = list()
        for i, name in enumerate(output_variables.copy()):
            if name not in input_variables:
                reduce_dims.append(i)
                output_variables.remove(name)
        if len(reduce_dims) > 0:
            grad = torch.sum(grad, dim=tuple(reduce_dims))

        permute_dims = list()
        assert len(input_variables) == len(output_variables)
        for name in input_variables:
            permute_dims.append(output_variables.index(name))
        if len(permute_dims) > 0:
            grad = torch.permute(grad, permute_dims)

        input_value.tensor_grad = grad
        input_value.backward_inner()
        return input_value

    def __str__(self):
        axes = ', '.join(('B', ) * self.batch_dims + self.batch_variables)
        quantized_flag = ', quantized' if self.quantized else ''
        backward_flag = ', backward=' + str(self._backward_function) if self._backward_function is not None else ''
        tensor_content = ''

        if self.tensor_grad is not None:
            content = str(self.tensor) + '\ngrad:\n' + str(self.tensor_grad)
        else:
            content = str(self.tensor)
        if self.tensor.numel() < type(self).STR_MAX_TENSOR_SIZE:
            if self.tensor.numel() < 10 and self.tensor.dim() <= 1:
                tensor_content = '{' + content.replace('\n', ' ') + '}'
            else:
                tensor_content = '{\n' + indent_text(content) + '\n}'
        return f'Value[{self.dtype}, axes=[{axes}], tdtype={self.tensor.dtype}, tdshape={tuple(self.tensor.shape)}{quantized_flag}{backward_flag}]{tensor_content}'

    __repr__ = jacinle.repr_from_str

    STR_MAX_TENSOR_SIZE = 100

    @classmethod
    def set_print_options(cls, max_tensor_size=STR_MAX_TENSOR_SIZE):
        cls.STR_MAX_TENSOR_SIZE = max_tensor_size


def wrap_value(value, dtype: Optional[ValueType] = None):
    if isinstance(value, Value):
        return value
    return Value(dtype, [], value)


def unwrap_value(value):
    if isinstance(value, Value):
        return value.tensor
    return value


def concat_values(*args: Value):  # will produce a Value with batch_dims == 1, but the input can be either 0-batch or 1-batch.
    assert len(args) > 0
    # 1. Sanity check.
    for value in args[1:]:
        assert value.dtype == args[0].dtype
        assert value.batch_variables == args[0].batch_variables
        assert value.quantized == args[0].quantized
        if args[0].tensor_indices is None:  # tensor_indices is None should be consistent.
            assert value.tensor_indices is None
        else:
            assert value.tensor_indices is not None

    all_tensor = [v.tensor for v in args]
    all_tensor_indices = [v.tensor_indices for v in args]
    all_tensor_mask = [v.tensor_mask for v in args]

    target_shape = tuple([
        max([v.get_variable_size(i) for v in args])
        for i in range(args[0].nr_variables)
    ])
    for i in range(len(args)):
        tensor, tensor_indices, tensor_mask = all_tensor[i], all_tensor_indices[i], all_tensor_mask[i]
        if tensor_mask is None:
            tensor_mask = torch.ones(tensor.size()[:args[i].total_batch_dims])
        all_tensor[i] = _pad_tensor(tensor, target_shape, args[i].dtype, args[i].batch_dims)
        if tensor_indices is not None:
            all_tensor_indices[i] = _pad_tensor(tensor_indices, target_shape, args[i].dtype, args[i].batch_dims)
        all_tensor_mask[i] = _pad_tensor(tensor_mask, target_shape, args[i].dtype, args[i].batch_dims)

        if args[0].batch_dims == 0:
            all_tensor[i] = all_tensor[i].unsqueeze(0)
            all_tensor_indices[i] = all_tensor_indices[i].unsqueeze(0) if all_tensor_indices[i] is not None else None
            all_tensor_mask[i] = all_tensor_mask[i].unsqueeze(0)
        else:
            assert args[0].batch_dims == 1

    return Value(
        args[0].dtype, args[0].batch_variables,
        QuantizedTensorValue(
            torch.cat(all_tensor, dim=0),
            torch.cat(all_tensor_indices, dim=0) if all_tensor_indices[0] is not None else None,
            torch.cat(all_tensor_mask, dim=0)
        ),
        batch_dims=1, quantized=args[0].quantized
    )


def _clone_tensor(tensor):
    return tensor if tensor is None else tensor.clone()


def _pad_tensor(tensor: torch.Tensor, target_shape: Sequence[int], dtype: ValueType, batch_dims: int):
    paddings = list()
    for size, max_size in zip(tensor.size()[batch_dims:], target_shape):
        paddings.extend((max_size - size, 0))
    if tensor.dim() - batch_dims == len(target_shape):
        pass
    elif tensor.dim() - batch_dims == len(target_shape) + dtype.ndim():
        paddings.extend([0 for _ in range(dtype.ndim() * 2)])
    else:
        raise ValueError('Shape error during tensor padding.')
    paddings.reverse()  # no need to add batch_dims.
    return F.pad(tensor, paddings, "constant", 0)


class MixedValue(object):
    def __init__(self, values: Sequence[Value]):
        self.values = values
        self._first_value = next(iter(values))

    @property
    def dtype(self):
        return self._first_value.dtype

    @property
    def batch_dims(self):
        return self._first_value.batch_dims

    @property
    def batch_variables(self):
        return self._first_value.batch_variables

    def nr_variables(self):
        return self._first_value.nr_variables
