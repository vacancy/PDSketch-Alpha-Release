from ctypes.wintypes import BOOLEAN
import jacinle
import random
import itertools
import collections
import torch
from typing import Any, Optional, Union, List, Dict
from enum import IntEnum

from hacl.pdsketch.interface.v2.value import BOOL, ValueType, NamedValueType, Variable, Value, is_intrinsically_quantized, wrap_value
from hacl.pdsketch.interface.v2.optimistic import OptimisticConstraint, OptimisticValueContext, is_optimistic_value, optimistic_value_id, DeterminedValue, OptimisticValue
from hacl.pdsketch.interface.v2.expr import FunctionDef, QuantifierType, BoolOpType, BoolOp, FunctionApplication, FeatureEqualOp, VariableExpression
from hacl.pdsketch.interface.v2.domain import GeneratorDef, Domain

__all__ = ['Assignment', 'AssignmentType', 'CSPNotSolvable', 'simple_csp_solve']


class AssignmentType(IntEnum):
    EQUAL = 0
    VALUE = 1
    IGNORE = 2


class Assignment(collections.namedtuple('_Assignment', ['t', 'd'])):
    t: AssignmentType
    d: Any

    @property
    def type(self):
        """Alias of `assignment.t`."""
        return self.t

    @property
    def data(self):
        """Alias of `assignment.d`."""
        return self.d


class CSPNotSolvable(Exception):
    pass


class CSPNoGenerator(Exception):
    pass


def _determined(*args):
    for x in args:
        if isinstance(x, OptimisticValue):
            return False
    return True


def _check_eq(domain: Domain, dtype: ValueType, v1: torch.Tensor, v2: torch.Tensor):
    if is_intrinsically_quantized(dtype):
        return (v1 == v2).item()
    assert isinstance(dtype, NamedValueType)
    eq_function = domain.get_external_function('type::' + dtype.typename + '::equal')
    return eq_function(wrap_value(v1, dtype), wrap_value(v2, dtype)).item()


def _check(domain: Domain, c: OptimisticConstraint):
    if c.func_def is BoolOpType.NOT:
        return c.args[0].value == (not c.rv.value)
    elif c.func_def in (QuantifierType.FORALL, BoolOpType.AND):
        return all([x.value for x in c.args]) == c.rv.value
    elif c.func_def in (QuantifierType.EXISTS, BoolOpType.OR):
        return any([x.value for x in c.args]) == c.rv.value
    elif c.is_equal_constraint:
        if c.args[0].dtype == BOOL:
            return (c.args[0].value == c.args[1].value) == c.rv.value
        else:
            return _check_eq(domain, c.args[0].dtype, c.args[0].value, c.args[1].value) == c.rv.value
    else:
        assert isinstance(c.func_def, FunctionDef)
        func = domain.get_external_function(c.func_def.name)
        rv = wrap_value(func(*[wrap_value(x.value, x.dtype) for x in c.args]), c.func_def.output_type)
        if rv.dtype == BOOL:
            return (rv.item() > 0.5) == c.rv.value
        else:
            return _check_eq(domain, c.func_def.output_type, rv, c.rv.value)


def _match_generator(c: OptimisticConstraint, g: GeneratorDef):
    def gen_input_output(func_arguments, rv_variable=None):
        inputs, outputs = [None for _ in range(len(g.input_vars))], [None for _ in range(len(g.output_vars))]
        for argc, argg in zip(c.args, func_arguments):
            if isinstance(argc, OptimisticValue):
                if argg.name.startswith('?g'):
                    outputs[int(argg.name[2:])] = argc
                else:
                    return None, None
            else:
                if argg.name.startswith('?c'):
                    inputs[int(argg.name[2:])] = argc.value
                else:
                    return None, None
        if rv_variable is not None:
            if rv_variable.name.startswith('?c'):
                inputs[int(rv_variable.name[2:])] = c.rv.value
            else:
                return None, None
        return inputs, outputs

    if isinstance(c.rv, DeterminedValue) and c.rv.dtype == BOOL:
        if c.rv.value:  # match (pred ?x ?y) == True
            if isinstance(g.flatten_certifies, FunctionApplication):
                if c.func_def.name == g.flatten_certifies.function_def.name:
                    return gen_input_output(g.flatten_certifies.arguments)
        else:
            if isinstance(g.flatten_certifies, BoolOp) and g.flatten_certifies.bool_op_type is BoolOpType.NOT:  # match (pred ?x ?y) == False
                if isinstance(g.flatten_certifies.arguments[0], FunctionApplication):
                    if c.func_def.name == g.flatten_certifies.arguments[0].function_def.name:
                        return gen_input_output(g.flatten_certifies.arguments[0].arguments)
                elif c.is_equal_constraint and isinstance(g.flatten_certifies.arguments[0], FeatureEqualOp):  # match (equal ?x ?y) == False
                    if c.args[0].dtype == g.flatten_certifies.arguments[0].feature.output_type:
                        return gen_input_output([g.flatten_certifies.arguments[0].feature, g.flatten_certifies.arguments[0].value])
    if isinstance(c.rv, DeterminedValue) and isinstance(c.func_def, FunctionDef):  # match (pred ?x ?y) == ?z
        if isinstance(g.flatten_certifies, FeatureEqualOp) and c.func_def.name == g.flatten_certifies.feature.function_def.name:
            if isinstance(g.flatten_certifies.value, VariableExpression):
                return gen_input_output(g.flatten_certifies.feature.arguments, g.flatten_certifies.value)
    return None, None


def _apply_assignments(domain, constraints: List[OptimisticConstraint], assignments: Dict[int, Assignment]):
    new_constraints = list()
    for c in constraints:
        if c is None:
            continue
        if isinstance(c.rv, OptimisticValue) and c.rv.identifier in assignments and assignments[c.rv.identifier].t is AssignmentType.IGNORE:
            continue

        new_args = list(c.args)
        for i, x in enumerate(c.args):
            if isinstance(x, OptimisticValue) and x.identifier in assignments:
                new_args[i] = DeterminedValue(x.dtype, assignments[x.identifier].d, False)
        new_rv = c.rv
        if isinstance(c.rv, OptimisticValue) and c.rv.identifier in assignments:
            new_rv = DeterminedValue(c.rv.dtype, assignments[c.rv.identifier].d, False)

        nc = OptimisticConstraint(c.func_def, new_args, new_rv)
        if _determined(nc.rv) and _determined(*nc.args):
            if _check(domain, nc):
                continue
            else:
                raise CSPNotSolvable()
        new_constraints.append(nc)
    return new_constraints


def _filter_deterministic_equal(domain, constraints: List[OptimisticConstraint], assignments: Dict[int, Assignment]):
    progress= False
    for i, c in enumerate(constraints):
        if c.is_equal_constraint and isinstance(c.rv, DeterminedValue):
            if c.rv.value:
                if isinstance(c.args[0], OptimisticValue):
                    if isinstance(c.args[1], OptimisticValue):
                        assignments[c.args[0].identifier] = Assignment(AssignmentType.EQUAL, c.args[1].identifier)
                    else:
                        assignments[c.args[0].identifier] = Assignment(AssignmentType.VALUE, c.args[1].value)
                    constraints[i] = None
                elif isinstance(c.args[1], OptimisticValue):
                    if isinstance(c.args[0], OptimisticValue):
                        assignments[c.args[1].identifier] = Assignment(AssignmentType.EQUAL, c.args[0].identifier)
                    else:
                        assignments[c.args[1].identifier] = Assignment(AssignmentType.VALUE, c.args[0].value)
                    constraints[i] = None
                else:
                    raise AssertionError('Sanity check failed.')
                progress = True
            else:
                if c.args[0].dtype == BOOL:
                    constraints[i] = OptimisticConstraint(BoolOpType.NOT, [c.args[0]],  c.args[1])
                    progress = True
    if progress:
        return progress, _apply_assignments(domain, constraints, assignments)
    return progress, constraints


def _filter_unused_rhs(domain, constraints: List[OptimisticConstraint], assignments: Dict[int, Assignment]):
    used = dict()
    for c in constraints:
        for x in c.args:
            if isinstance(x, OptimisticValue):
                used.setdefault(x.identifier, 0)
                used[x.identifier] += 1
        if isinstance(c.rv, OptimisticValue):
            used.setdefault(c.rv.identifier, 0)
    for k, v in used.items():
        if v == 0:
            assignments[k] = Assignment(AssignmentType.IGNORE, None)
    return _apply_assignments(domain, constraints, assignments)


def _filter_deterministic_clauses(domain, constraints, assignments):
    progress = False
    for c in constraints:
        if isinstance(c.func_def, (QuantifierType, BoolOpType)):
            if _determined(c.rv):
                if c.func_def in (QuantifierType.FORALL, BoolOpType.AND) and c.rv.value:
                    for x in c.args:
                        if isinstance(x, OptimisticValue):
                            assignments[x.identifier] = Assignment(AssignmentType.VALUE, True)
                            progress = True
                            # print('assign', optimistic_value_id(x), True)
                        elif not x:
                            raise CSPNotSolvable()
                elif c.func_def in (QuantifierType.EXISTS, BoolOpType.OR) and not c.rv.value:
                    for x in c.args:
                        if isinstance(x, OptimisticValue):
                            progress = True
                            assignments[x.identifier] = Assignment(AssignmentType.VALUE, False)
                            # print('assign', optimistic_value_id(x), False)
                        elif x:
                            raise CSPNotSolvable()
                elif c.func_def is BoolOpType.NOT:
                    progress = True
                    assignments[c.args[0].identifier] = Assignment(AssignmentType.VALUE, not c.rv.value)
    if progress:
        return progress, _apply_assignments(domain, constraints, assignments)
    return progress, constraints


def _find_bool_variable(csp: OptimisticValueContext, constraints, assignments):
    count = collections.defaultdict(int)
    for c in constraints:
        for x in itertools.chain(c.args, [c.rv]):
            if isinstance(x, OptimisticValue) and x.identifier not in assignments and x.dtype == BOOL:
                count[x.identifier] += 1
    if len(count) == 0:
        return None
    return max(count, key=count.get)


def _find_gen_variable(domain: Domain, constraints: List[OptimisticConstraint]):
    # Step 1: find all applicable generators.
    all_generators = list()
    for c in constraints:
        for g in domain.generators.values():
            i, o = _match_generator(c, g)
            # jacinle.log_function.print('matching', c, g, i, o)
            if i is not None:
                all_generators.append((c, g, i, o))

    # Step 2: find if there is any variable with only one generator.
    target_to_generator = collections.defaultdict(list)
    for c, g, i, o in all_generators:
        for target in o:
            target_to_generator[target.identifier].append((c, g, i, o))

    for target, generators in target_to_generator.items():
        if len(target_to_generator[target]) == 1:
            return target_to_generator[target]

    if len(all_generators) > 0:
        max_priority = max([r[1].priority for r in all_generators])
        all_generators = [r for r in all_generators if r[1].priority == max_priority]
        return all_generators

    return None


def _find_typegen_variable(domain: Domain, dtype: ValueType):
    assert isinstance(dtype, NamedValueType)
    for g in domain.generators.values():
        if len(g.function_def.arguments) == 0 and g.function_def.output_type[0] == dtype:
            if isinstance(g.certifies, BoolOp) and g.certifies.bool_op_type is BoolOpType.AND:
                return g


def simple_csp_solve(
    domain: Domain,
    csp: OptimisticValueContext,
    max_generator_trials: Optional[int] = 3,
    solvable_only: Optional[bool] = False
):
    csp = domain.value_quantizer.unquantize_optimistic_context(csp)
    constraints = csp.constraints.copy()

    # @jacinle.log_function
    def dfs(constraints, assignments):
        if len(constraints) == 0:
            return assignments

        progress = True
        while progress:
            progress, constraints = _filter_deterministic_equal(domain, constraints, assignments)
        constraints = _filter_unused_rhs(domain, constraints, assignments)
        progress = True
        while progress:
            progress, constraints = _filter_deterministic_clauses(domain, constraints, assignments)

        if len(constraints) == 0:
            return assignments

        next_bool_var = _find_bool_variable(csp, constraints, assignments)
        if next_bool_var is not None:
            assignments_true = assignments.copy()
            assignments_true[next_bool_var] = Assignment(AssignmentType.VALUE, True)
            try:
                constraints_true = _apply_assignments(domain, constraints, assignments_true)
                return dfs(constraints_true, assignments_true)
            except CSPNotSolvable:
                pass

            assignments_false = assignments.copy()
            assignments_false[next_bool_var] = Assignment(AssignmentType.VALUE, False)
            try:
                constraints_false = _apply_assignments(domain, constraints, assignments_false)
                return dfs(constraints_false, assignments_false)
            except CSPNotSolvable:
                pass

            raise CSPNotSolvable()
        else:
            # jacinle.log_function.print('remaining constraints', *constraints, sep='\n  ')

            next_gen_vars = _find_gen_variable(domain, constraints)
            if next_gen_vars is None:
                for name, dtype in csp.index2type.items():
                    if name not in assignments:
                        g = _find_typegen_variable(domain, dtype)
                        if g is not None:
                            next_gen_vars = [(None, g, [], [OptimisticValue(dtype, name)])]
                            break
                if next_gen_vars is None:
                    # jacinle.log_function.print('Can not find a generator. Constraints:\n  ' + '\n  '.join([str(x) for x in constraints]))
                    raise CSPNoGenerator('Can not find a generator. Constraints:\n  ' + '\n  '.join([str(x) for x in constraints]))

            if len(next_gen_vars) > 1:
                # jacinle.log_function.print('generator orders', *[str(vv[1]).split('\n')[0] for vv in next_gen_vars], sep='\n  ')
                pass

            for vv in next_gen_vars:
                c, g, args, outputs_target = vv

                for j in range(max_generator_trials):
                    outputs = domain.get_external_function(g.function_def.name)(*args)
                    if outputs is None:
                        break

                    # jacinle.log_function.print('running generator', g, f'count = {j}')
                    new_assignments = assignments.copy()
                    for output, target in zip(outputs, outputs_target):
                        output = wrap_value(output, target.dtype)
                        new_assignments[target.identifier] = Assignment(AssignmentType.VALUE, output)
                    try:
                        new_constraints = _apply_assignments(domain, constraints, new_assignments)
                        return dfs(new_constraints, new_assignments)
                    except CSPNotSolvable:
                        pass
            raise CSPNotSolvable()

    try:
        assignments = dfs(constraints, {})
        if solvable_only:
            return True
        for name, dtype in csp.index2type.items():
            if name not in assignments:
                g = _find_typegen_variable(domain, dtype)
                if g is None:
                    raise NotImplementedError('Can not find a generator for unbounded variable {}, type {}.'.format(name, dtype))
                else:
                    output, = domain.get_external_function(g.function_def.name)()
                    assignments[name] = Assignment(AssignmentType.VALUE, wrap_value(output, dtype))
        return assignments
    except CSPNotSolvable:
        return None
    except CSPNoGenerator:
        raise
