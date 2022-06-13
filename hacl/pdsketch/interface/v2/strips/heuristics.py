import jacinle
from typing import Optional, Tuple, List, Set, Dict
from .strips_expr import StripsProposition, StripsState
from .grounded_expr import GStripsClassifierForwardDiffReturn, GStripsClassifier, GStripsSimpleAssignment, GStripsConditionalAssignment, gs_is_simple_conjunctive_classifier
from .grounding import GStripsTask, GStripsTranslatorBase

__all__ = ['StripsHeuristic', 'StripsBlindHeuristic', 'StripsRPGHeuristic', 'StripsHFFHeuristic']


class StripsHeuristic(object):
    def __init__(self, task: GStripsTask, translator: Optional[GStripsTranslatorBase] = None):
        self.task = task
        self.goal_func = task.goal.compile()
        self.translator = translator

    @classmethod
    def from_type(cls, type_identifier: str, task: GStripsTask, translator: Optional[GStripsTranslatorBase] = None, **kwargs) -> 'StripsHeuristic':
        if type_identifier == 'hff':
            return StripsHFFHeuristic(task, translator, **kwargs)
        elif type_identifier == 'blind':
            return StripsBlindHeuristic(task, translator)

    def compute(self, state: StripsState) -> int:
        raise NotImplementedError()


class StripsBlindHeuristic(StripsHeuristic):
    def __init__(self, task: GStripsTask, translator: Optional[GStripsTranslatorBase] = None):
        super().__init__(task, translator)

    def compute(self, state: StripsState) -> int:
        goal_rv = self.goal_func(state)
        return 0 if goal_rv else 1


class StripsRPGHeuristic(StripsHeuristic):
    def __init__(self, task: GStripsTask, translator: Optional[GStripsTranslatorBase] = None, forward_relevance_analysis: bool = True, backward_relevance_analysis: bool = True):
        super().__init__(task, translator)

        self.forward_relevance_analysis = forward_relevance_analysis
        self.backward_relevance_analysis = backward_relevance_analysis

        if task.is_relaxed:
            self.relaxed = task
        else:
            assert translator is not None
            self.relaxed = translator.recompile_relaxed_task(task, forward_relevance_analysis=forward_relevance_analysis, backward_relevance_analysis=backward_relevance_analysis)

    def compute_rpg_forward_diff(self, state: StripsState, relaxed_task: Optional[GStripsTask] = None) -> Tuple[
        List[Set[StripsState]],
        Dict[StripsProposition, int],
        List[List[Tuple[int, int, Set[StripsProposition]]]],
        List[List[Tuple[int, int, Set[StripsProposition]]]],
        GStripsClassifierForwardDiffReturn
    ]:
        if relaxed_task is None:
            relaxed_task = self.relaxed

        with GStripsClassifier.enable_forward_diff_ctx():
            F_sets = [set(state)]
            A_sets = []
            D_sets = []

            used_operators = set()
            used_derived_predicates = set()
            # print('rpginit', F_sets[-1])

            goal_rv = self.goal_func(F_sets[-1])
            while not goal_rv.rv:
                # for op in relaxed_task.operators:
                #     print(' ', op.raw_operator, op.precondition, op.applicable(F_sets[-1]))

                new_ops: List[Tuple[int, int, Set[StripsProposition]]] = list()  # op_index, eff_index, op_precondition
                # print('Starting new step')
                for i, op in enumerate(relaxed_task.operators):
                    op_pred_rv = None
                    for j, e in enumerate(op.effects):
                        if (i, j) not in used_operators:
                            if op_pred_rv is None:
                                op_pred_rv = op.applicable(F_sets[-1])
                                # print('Evaluating op', op.raw_operator, op_pred_rv)
                            if op_pred_rv.rv:
                                if isinstance(e, GStripsSimpleAssignment):
                                    new_ops.append((i, 0, op_pred_rv.propositions))
                                    used_operators.add((i, j))
                                elif isinstance(e, GStripsConditionalAssignment):
                                    eff_pred_rv = e.applicable(F_sets[-1])
                                    if eff_pred_rv.rv:
                                        new_ops.append((i, j, op_pred_rv.propositions | eff_pred_rv.propositions))
                                        # print('  Use operator', i, j)
                                        used_operators.add((i, j))
                            else:
                                break

                new_F = F_sets[-1].copy()
                for op_index, effect_index, _ in new_ops:
                    op = relaxed_task.operators[op_index]
                    eff = op.effects[effect_index]
                    new_F.update(eff.add_effects)

                new_dps: List[Tuple[int, int, Set[StripsProposition]]] = list()  # dp_index, eff_index, dp_precondition
                for i, dp in enumerate(relaxed_task.derived_predicates):
                    for j, e in enumerate(dp.effects):
                        if (i, j) not in used_derived_predicates:
                            dp_pred_rv = e.applicable(new_F)
                            # print((i, j), dp_pred_rv)
                            if dp_pred_rv.rv:
                                used_derived_predicates.add((i, j))
                                new_dps.append((i, j, dp_pred_rv.propositions))

                for dp_index, effect_index, _ in new_dps:
                    dp = relaxed_task.derived_predicates[dp_index]
                    eff = dp.effects[effect_index]
                    new_F.update(eff.add_effects)

                # print('depth', len(F_sets), new_F)
                if len(new_F) == len(F_sets[-1]):
                    break

                A_sets.append(new_ops)
                D_sets.append(new_dps)
                F_sets.append(new_F)
                goal_rv = self.goal_func(F_sets[-1])

            F_levels = {}
            for i in range(0, len(F_sets)):
                for f in F_sets[i]:
                    if f not in F_levels:
                        F_levels[f] = i

            return F_sets, F_levels, A_sets, D_sets, goal_rv

    def compute(self, state: StripsState) -> int:
        raise NotImplementedError()


class StripsHFFHeuristic(StripsRPGHeuristic):
    def compute(self, state: StripsState):
        relaxed = self.translator.recompile_task_new_state(self.relaxed, state, forward_relevance_analysis=False, backward_relevance_analysis=False)
        state = relaxed.state

        F_sets, F_levels, A_sets, D_sets, goal_rv = self.compute_rpg_forward_diff(state, relaxed_task=relaxed)

        if not goal_rv.rv:
            return int(1e9)

        selected: Set[Tuple[int, int]] = set()
        selected_facts = [set() for _ in F_sets]

        goal_propositions = goal_rv.propositions
        for pred in goal_propositions:
            if pred not in F_levels:
                return int(1e9)
            selected_facts[F_levels[pred]].add(pred)

        for t in reversed(list(range(len(A_sets)))):
            for dp_index, effect_index, propositions in D_sets[t]:
                dp = self.relaxed.derived_predicates[dp_index]
                eff = dp.effects[effect_index]
                if eff.add_effects.intersection(selected_facts[t+1]):
                    # print('Selecting Derived Predicate', dp_index, effect_index, propositions, eff.add_effects)
                    selected_facts[t + 1] -= eff.add_effects
                    for pred in propositions:
                        # print('  Selecting predicate', pred)
                        selected_facts[F_levels[pred]].add(pred)
            for op_index, effect_index, propositions in A_sets[t]:
                op = self.relaxed.operators[op_index]
                eff = op.effects[effect_index]
                if eff.add_effects.intersection(selected_facts[t+1]):
                    # TODO:: FIX...
                    # print('Selecting Action', op_index, effect_index, op.raw_operator, propositions, eff.add_effects)
                    selected.add((op_index, effect_index))  # only add the operator or both? Maybe at each level, just add the operator? or add a flag?
                    selected_facts[t + 1] -= eff.add_effects
                    for pred in propositions:
                        # print('  Selecting predicate', pred)
                        selected_facts[F_levels[pred]].add(pred)
        return len(selected)
