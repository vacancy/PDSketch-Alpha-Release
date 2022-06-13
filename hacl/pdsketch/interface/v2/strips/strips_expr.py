"""
Strips expressions.
"""

import jacinle
from typing import Optional, Union, Sequence, Tuple, FrozenSet, Dict
from hacl.pdsketch.interface.v2.value import Variable

__all__ = [
    'StripsPredicateName', 'StripsProposition', 'StripsSASProposition', 'StripsState', 'StripsSASState',
    'StripsExpression', 'StripsValueOutputExpression', 'StripsVariableAssignmentExpression',
    'StripsBoolOutputExpression', 'StripsBoolConstant', 'StripsBoolPredicate', 'StripsSASPredicate',
    'StripsBoolAOFormula', 'StripsBoolFEFormula', 'StripsBoolNot', 'StripsSASExpression',
    'StripsAssignment', 'StripsConditionalAssignment', 'StripsDeicticAssignment'
]

StripsPredicateName = str
StripsProposition = str
StripsSASProposition = Tuple[str, int]
StripsState = frozenset[StripsProposition]
StripsSASState = dict[Tuple[StripsPredicateName, str], int]  # mapping from (predicate name, arguments_str) to value.


class StripsExpression(object):
    __repr__ = jacinle.repr_from_str


class StripsValueOutputExpression(StripsExpression):
    pass


class StripsVariableAssignmentExpression(StripsExpression):
    pass


class StripsBoolOutputExpression(StripsValueOutputExpression):
    pass


class StripsBoolConstant(StripsBoolOutputExpression):
    def __init__(self, value: bool):
        self.value = value

    def __str__(self):
        return 'true' if self.value else 'false'


class StripsBoolPredicate(StripsBoolOutputExpression):
    def __init__(self, name: StripsPredicateName, negated: bool, arguments: Sequence[Variable]):
        self.name = name
        self.arguments = arguments
        self.negated = negated

    def __str__(self):
        argument_str = ' '.join(str(x) for x in self.arguments)
        fmt = f'({self.name} {argument_str})'
        if self.negated:
            return f'(not {fmt})'
        return fmt


class StripsSASPredicate(StripsBoolPredicate):
    def __init__(self, sas_name: StripsPredicateName, sas_index: Optional[int], negated: bool, arguments: Sequence[Variable]):
        if sas_index is None:
            super().__init__(sas_name, negated, arguments)
        else:
            super().__init__(sas_name + '@' + str(sas_index), negated, arguments)
        self.sas_name = sas_name
        self.sas_index = sas_index


class StripsBoolAOFormula(StripsBoolOutputExpression):
    def __init__(self, arguments: Sequence[StripsBoolOutputExpression], is_disjunction: bool):
        self.arguments = arguments
        self.is_disjunction = is_disjunction

    @property
    def is_conjunction(self) -> bool:
        return not self.is_disjunction

    def __str__(self):
        arguments_str = [str(arg) for arg in self.arguments]
        if sum(len(s) for s in arguments_str) > 120:
            arguments_str = [jacinle.indent_text(s) for s in arguments_str]
            fmt = '\n'.join(arguments_str)
            return f'(or\n{fmt}\n)' if self.is_disjunction else f'(and\n{fmt}\n)'
        return '({} {})'.format('or' if self.is_disjunction else 'and', ' '.join(arguments_str))


class StripsBoolFEFormula(StripsBoolOutputExpression):
    def __init__(self, variable: Variable, expr: StripsBoolOutputExpression, is_disjunction: bool):
        self.variable = variable
        self.expr = expr
        self.is_disjunction = is_disjunction

    @property
    def is_conjunction(self) -> bool:
        return not self.is_disjunction

    @property
    def is_forall(self) -> bool:
        return not self.is_disjunction

    @property
    def is_exists(self) -> bool:
        return self.is_disjunction

    def __str__(self):
        return '({} ({}) {})'.format('exists' if self.is_disjunction else 'forall', str(self.variable), str(self.expr))


class StripsBoolNot(StripsBoolOutputExpression):
    def __init__(self, expr: StripsBoolOutputExpression):
        self.expr = expr

    def __str__(self):
        return '(not {})'.format(str(self.expr))


class StripsSASExpression(StripsValueOutputExpression):  #  For all external functions.
    def __init__(self, mappings: Dict[int, StripsBoolOutputExpression]):
        self.mappings: Dict[int, StripsBoolOutputExpression] = mappings

    def __str__(self):
        return '(SAS\n{}\n)'.format('\n'.join('  ' + str(i) + ' <- ' + str(self.mappings[i]) for i in self.mappings))


class StripsAssignment(StripsVariableAssignmentExpression):
    def __init__(self, feature: Union[StripsBoolPredicate, StripsSASPredicate], value: Union[StripsBoolOutputExpression, StripsSASExpression]):
        self.feature = feature
        self.value = value

    def __str__(self):
        return '({} <- {})'.format(str(self.feature), str(self.value))


class StripsDeicticAssignment(StripsVariableAssignmentExpression):
    def __init__(self, variable: Variable, expr: StripsVariableAssignmentExpression):
        self.variable = variable
        self.expr = expr

    def __str__(self):
        return '(foreach ({}) {})'.format(self.variable, str(self.expr))


class StripsConditionalAssignment(StripsVariableAssignmentExpression):
    def __init__(self, assign_op: StripsAssignment, condition: StripsBoolOutputExpression):
        self.assign_op = assign_op
        self.condition = condition

    @property
    def feature(self):
        return self.assign_op.feature

    @property
    def value(self):
        return self.assign_op.value

    def __str__(self):
        return '({} if {})'.format(str(self.assign_op), str(self.condition))
