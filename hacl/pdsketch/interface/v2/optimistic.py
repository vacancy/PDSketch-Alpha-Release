import torch

import jacinle
from jacinle.utils.enum import JacEnum
from typing import Any, Optional, Union, List, Set, Mapping
from .value import ValueType, NamedValueType, BOOL, Value, Variable

__all__ = [
    'OPTIM_MAGIC_NUMBER', 'OPTIM_MAGIC_NUMBER_UPPER', 'RELAX_MAGIC_NUMBER',
    'is_optimistic_value', 'optimistic_value_id', 'is_relaxed_value', 'is_optimistic_value', 'DeterminedValue', 'OptimisticValue',
    'cvt_opt_value', 'OptimisticConstraint', 'EqualOptimisticConstraint', 'OptimisticValueContext', 'RelaxedExecutionContext', 'RelaxedBackwardContext'
]


OPTIM_MAGIC_NUMBER = -2147483647
OPTIM_MAGIC_NUMBER_UPPER = (-2147483648) / 2
RELAX_MAGIC_NUMBER = int(OPTIM_MAGIC_NUMBER_UPPER * 1.5)


def is_optimistic_value(v):
    return v < OPTIM_MAGIC_NUMBER_UPPER


def optimistic_value_id(v):
    return v - OPTIM_MAGIC_NUMBER


def is_relaxed_value(v):
    return RELAX_MAGIC_NUMBER < v < OPTIM_MAGIC_NUMBER_UPPER


def relaxed_value_id(v):
    return v - RELAX_MAGIC_NUMBER


class DeterminedValue(object):
    def __init__(self, dtype: ValueType, value: Union[int, Value], quantized):
        assert isinstance(dtype, NamedValueType) or dtype == BOOL
        self.dtype = dtype
        self.value = value
        self.quantized = quantized

    def __str__(self):
        quantized_flag = ', quantized' if self.quantized else ''
        return f'D[{self.dtype}{quantized_flag}]::{self.value}'

    __repr__ = jacinle.repr_from_str


class OptimisticValue(object):
    def __init__(self, dtype: ValueType, identifier: int):
        assert isinstance(dtype, NamedValueType) or dtype == BOOL
        self.dtype = dtype
        self.identifier = identifier

    def __str__(self):
        return f'O[{self.dtype}]::@{optimistic_value_id(self.identifier)}'

    __repr__ = jacinle.repr_from_str


def cvt_opt_value(value: Union[DeterminedValue, OptimisticValue, Value, int], dtype: Optional[ValueType] = None):
    if isinstance(value, (DeterminedValue, OptimisticValue)):
        return value
    elif isinstance(value, Value):
        return DeterminedValue(value.dtype, value.item(), value.quantized)
    elif isinstance(value, int):
        assert dtype is not None
        if is_optimistic_value(value):
            return OptimisticValue(dtype, value)
        else:
            return DeterminedValue(dtype, value, True)
    else:
        raise TypeError('Unknown value type: {}.'.format(value))


class OptimisticConstraint(object):
    EQUAL = '__EQ__'

    def __init__(self, func_def, args, rv):
        self.func_def = func_def
        self.args = tuple(map(cvt_opt_value, args))
        self.rv = cvt_opt_value(rv)

    def __str__(self):
        if self.is_equal_constraint and isinstance(self.rv, DeterminedValue):
            if self.rv.value:
                return '__EQ__(' + ', '.join([str(x) for x in self.args]) + ')'
            else:
                return '__NEQ__(' + ', '.join([str(x) for x in self.args]) + ')'
        else:
            if isinstance(self.func_def, (str, JacEnum)):
                name = str(self.func_def)
            else:
                name = self.func_def.name
            return name + '(' + ', '.join(
                [str(x) for x in self.args]
            ) + ') == ' + str(self.rv)

    __repr__ = jacinle.repr_from_str

    @property
    def is_equal_constraint(self):
        return isinstance(self.func_def, str) and self.func_def == OptimisticConstraint.EQUAL

    @classmethod
    def from_function_def(cls, func_def, args, rv):
        def _cvt(x, dtype):
            if isinstance(x, bool):
                assert dtype == BOOL
                return DeterminedValue(BOOL, int(x), True)
            elif isinstance(x, int):
                if is_optimistic_value(x):
                    return OptimisticValue(dtype, x)
                else:
                    return DeterminedValue(dtype, x, True)
            else:
                assert isinstance(x, torch.Tensor)
                return DeterminedValue(dtype, x, False)
        args = [_cvt(x, var.type if isinstance(var, Variable) else var) for x, var in zip(args, func_def.arguments)]
        rv = _cvt(rv, func_def.output_type)
        return cls(func_def, args, rv)


class EqualOptimisticConstraint(OptimisticConstraint):
    def __init__(self, left, right, rv=None):
        super().__init__(OptimisticConstraint.EQUAL, [left, right], rv if rv is not None else DeterminedValue(BOOL, True, True))

    @classmethod
    def from_bool(cls, left, right, rv=None):
        def _cvt(x):
            if x is None:
                return x
            if isinstance(x, bool):
                return DeterminedValue(BOOL, int(x), True)
            else:
                assert isinstance(x, int) and is_optimistic_value(x)
                return OptimisticValue(BOOL, x)
        return cls(_cvt(left), _cvt(right), _cvt(rv))


class OptimisticValueContext(object):
    def __init__(
        self,
        counter: Optional[int] = 0,
        index2actionable: Optional[Mapping[int, bool]] = None,
        index2type: Optional[Mapping[int, 'ValueType']] = None,
        index2domain: Optional[Mapping[int, Set[Any]]] = None,
        constraints: Optional[List[OptimisticConstraint]] = None,
    ):
        self.optim_var_counter = counter
        self.index2actionable = index2actionable if index2actionable is not None else dict()
        self.index2type = index2type if index2type is not None else dict()
        self.index2domain = index2domain if index2domain is not None else dict()
        self.constraints = constraints if constraints is not None else list()

    def clone(self):
        return OptimisticValueContext(self.optim_var_counter, self.index2actionable.copy(), self.index2type.copy(), self.index2domain.copy(), self.constraints.copy())

    def new_actionable_var(self, dtype: ValueType, domain: Optional[Set[Any]] = None) -> int:
        identifier = self.new_var(dtype, domain)
        self.index2actionable[identifier] = True
        return identifier

    def new_var(self, dtype: ValueType, domain: Optional[Set[Any]] = None) -> int:
        identifier = OPTIM_MAGIC_NUMBER + self.optim_var_counter
        self.index2type[identifier] = dtype
        self.optim_var_counter += 1
        if domain is not None:
            self.index2domain[identifier] = domain
        return identifier

    def get_type(self, identifier: int) -> str:
        return self.index2type[identifier]

    def get_domain(self, identifier: int) -> Set[Any]:
        return self.index2domain[identifier]

    def add_domain_value(self, identifier: int, value: Any):
        if identifier not in self.index2domain:
            self.index2domain[identifier] = set()
        self.index2domain[identifier].add(value)

    def add_constraint(self, c: OptimisticConstraint):
        self.constraints.append(c)

    def __str__(self):
        fmt = 'OptimisticValueContext{\n'
        fmt += '  Actionable Variables:\n    ' + '\n    '.join([f'@{x - OPTIM_MAGIC_NUMBER} - {self.index2type[x]}' for x in self.index2actionable]) + '\n'
        fmt += '  Constraints:\n'
        for c in self.constraints:
            fmt += f'    {c}\n'
        fmt += '}'
        return fmt

    __repr__ = jacinle.repr_from_str


class RelaxedExecutionContext(object):
    def __init__(self, op_identifier: int):
        self.op_identifier = op_identifier
        self.backward_values: List[Value] = list()

    def add_backward_value(self, value: Value):
        self.backward_values.append(value)


class RelaxedBackwardContext(object):
    def __init__(self, rv_set: Set[int]):
        self.rv_set = rv_set
