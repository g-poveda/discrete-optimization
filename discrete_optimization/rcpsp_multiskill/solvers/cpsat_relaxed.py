#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import itertools
import math
from typing import Any, Hashable

import cpmpy as cp
import numpy as np
from ortools.sat.python.cp_model import (
    CpModel,
    CpSolverSolutionCallback,
    Domain,
    LinearExpr,
)

from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpySolver,
    MetaCpmpyConstraint,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
    PreemptiveMultiskillRcpspSolution,
    compute_discretize_calendar_skills,
    create_fake_tasks_multiskills,
    discretize_calendar_,
)


def compute_skills_task_matrix(problem: MultiskillRcpspProblem):
    nb_skills = len(problem.skills_set)
    nb_tasks = problem.nb_tasks
    skills = problem.skills_list
    skills_to_index = {skills[i]: i for i in range(nb_skills)}
    bin_matrix = np.zeros(shape=(nb_tasks, nb_skills), dtype=bool)
    for i in range(nb_tasks):
        t = problem.tasks_list[i]
        modes = problem.mode_details[t][list(problem.mode_details[t].keys())[0]]
        for s in skills_to_index:
            if modes.get(s, 0) > 0:
                bin_matrix[i, skills_to_index[s]] = modes.get(s, 0)
    return bin_matrix


def compute_skills_matrix(problem: MultiskillRcpspProblem):
    nb_skills = len(problem.skills_list)
    nb_workers = problem.nb_employees
    skills = problem.skills_list
    skills_to_index = {skills[i]: i for i in range(nb_skills)}
    workers = problem.employees_list
    bin_matrix = np.zeros(shape=(nb_workers, nb_skills), dtype=bool)
    for worker_index in range(nb_workers):
        worker = workers[worker_index]
        nz_skills = problem.employees[worker].get_non_zero_skills()
        for nz_s in nz_skills:
            bin_matrix[worker_index, skills_to_index[nz_s]] = 1
    return bin_matrix


class CpSatRelaxedMsRcpspSolver(OrtoolsCpSatSolver):
    problem: MultiskillRcpspProblem
    hyperparameters = [
        CategoricalHyperparameter(
            name="add_pair_skills_resource", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="add_all_combination_skills", choices=[True, False], default=False
        ),
    ]

    def __init__(self, problem: MultiskillRcpspProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}
        self.skills_matrix = compute_skills_matrix(problem=self.problem)
        self.skills_to_index = {
            problem.skills_list[i]: i for i in range(len(self.problem.skills_list))
        }

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> MultiskillRcpspSolution:
        schedule = {}
        modes_dict = {}
        for task in self.variables["base_variable"]["starts"]:
            schedule[task] = {
                "start_time": cpsolvercb.Value(
                    self.variables["base_variable"]["starts"][task]
                ),
                "end_time": cpsolvercb.Value(
                    self.variables["base_variable"]["ends"][task]
                ),
            }
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                modes_dict[task] = modes[0]
        return MultiskillRcpspSolution(
            problem=self.problem, modes=modes_dict, schedule=schedule, employee_usage={}
        )

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        self.cp_model = CpModel()
        assert not self.problem.is_multimode
        one_skill_per_task = args.get("one_skill_per_task", False)
        self.create_base_variable()
        self.constraint_redundant_cumulative_skills()
        self.constraint_redundant_cumulative_worker(
            one_skill_per_worker_per_task=one_skill_per_task
        )
        self.constraint_precedence()
        ratios = self.compute_skills_constraint_ratios()
        nb_skills = len(self.problem.skills_list)
        if args["add_all_combination_skills"]:
            self.add_virtual_skills_pools()
        if args["add_pair_skills_resource"]:
            self.add_virtual_resource_pools(
                ratios=ratios, threshold=1, max_nb_res_pool=10000
            )
        makespan = self.cp_model.NewIntVar(
            lb=0, ub=self.problem.horizon, name="makespan"
        )
        self.cp_model.AddMaxEquality(
            makespan,
            [
                self.variables["base_variable"]["ends"][t]
                for t in self.variables["base_variable"]["ends"]
            ],
        )
        self.cp_model.Minimize(makespan)

    def compute_skills_constraint_ratios(self):
        ratios = {}
        nb_skills = len(self.problem.skills_list)
        for i in range(nb_skills):
            nb_i = np.sum(self.skills_matrix[:, i])
            for j in range(i + 1, nb_skills):
                nb_j = np.sum(self.skills_matrix[:, j])
                union = np.sum(self.skills_matrix[:, i] + self.skills_matrix[:, j] >= 1)
                ratio = (nb_i + nb_j) / union
                ratios[(i, j)] = ratio
        from itertools import product

        return ratios

    def add_virtual_skills_pools(self):
        skill_list = self.problem.skills_list
        all_subsets = []
        # Generate subsets for every possible length r, from 1 to the full list size
        for r in range(2, len(skill_list) + 1):
            subsets_of_length_r = itertools.combinations(skill_list, r)
            all_subsets.extend(subsets_of_length_r)
        for set_skills in all_subsets:
            self.add_virtual_skill_resource(set_skills=set_skills)
        return all_subsets

    def add_virtual_resource_pools(
        self, ratios: dict[tuple, float], threshold: float, max_nb_res_pool: int
    ):
        sorted_ratios = sorted(ratios, key=lambda x: ratios[x], reverse=True)
        for i in range(min(len(sorted_ratios), max_nb_res_pool)):
            print(ratios[sorted_ratios[i]])
            if ratios[sorted_ratios[i]] >= threshold:
                set_skills = {
                    self.problem.skills_list[index_s] for index_s in sorted_ratios[i]
                }
                self.add_virtual_skill_resource(set_skills)

    def add_virtual_skill_resource(self, set_skills: set[Hashable]):
        index_skills = [self.skills_to_index[s] for s in set_skills]
        workers_indexes = [
            worker_index
            for worker_index in range(self.problem.nb_employees)
            if sum(
                [
                    self.skills_matrix[worker_index, index_skill]
                    for index_skill in index_skills
                ]
            )
            >= 1
        ]
        nb_workers = len(workers_indexes)

        some_employee = next(emp for emp in self.problem.employees)
        len_calendar = len(self.problem.employees[some_employee].calendar_employee)
        merged_calendar = np.zeros(len_calendar)
        for worker_index in workers_indexes:
            merged_calendar += np.array(
                self.problem.employees[
                    self.problem.employees_list[worker_index]
                ].calendar_employee
            )
        discr_calendar = discretize_calendar_(merged_calendar)
        intervals_consume = []
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            mode_details = self.problem.mode_details[task][modes[0]]
            sum_needed = sum([mode_details.get(skill, 0) for skill in set_skills])
            if sum_needed > 0:
                intervals_consume.append(
                    (self.variables["base_variable"]["intervals"][task], sum_needed)
                )

        calendar_tasks = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name="calendar_res"
                ),
                f.get("value", 0),
            )
            for f in discr_calendar
            if f.get("value", 0) > 0
        ]
        self.cp_model.AddCumulative(
            [x[0] for x in intervals_consume] + [x[0] for x in calendar_tasks],
            [x[1] for x in intervals_consume] + [x[1] for x in calendar_tasks],
            capacity=nb_workers,
        )

    def create_base_variable(self):
        start_var = {}
        end_var = {}
        duration_var = {}
        interval_var = {}
        for task in self.problem.tasks_list:
            possible_duration = [
                self.problem.mode_details[task][m]["duration"]
                for m in self.problem.mode_details[task]
            ]
            start_var[task] = self.cp_model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"start_{task}"
            )
            end_var[task] = self.cp_model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"end_{task}"
            )
            duration_var[task] = self.cp_model.NewIntVarFromDomain(
                domain=Domain.FromValues(possible_duration), name=f"duration_{task}"
            )
            interval_var[task] = self.cp_model.NewIntervalVar(
                start=start_var[task],
                size=duration_var[task],
                end=end_var[task],
                name=f"interval_{task}",
            )
        self.variables["base_variable"] = {
            "starts": start_var,
            "ends": end_var,
            "durations": duration_var,
            "intervals": interval_var,
        }

    def constraint_redundant_cumulative_skills(self):
        discr_calendar, dict_calendar_skills = compute_discretize_calendar_skills(
            problem=self.problem
        )
        for skill in self.problem.skills_set:
            intervals_consume = []
            for task in self.problem.tasks_list:
                modes = list(self.problem.mode_details[task].keys())
                if len(modes) == 1:
                    if self.problem.mode_details[task][modes[0]].get(skill, 0) > 0:
                        intervals_consume.append(
                            (
                                self.variables["base_variable"]["intervals"][task],
                                self.problem.mode_details[task][modes[0]][skill],
                            )
                        )
                else:
                    for mode in modes:
                        if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                            intervals_consume.append(
                                (
                                    self.variables["mode_variable"]["opt_intervals"][
                                        task
                                    ][mode],
                                    self.problem.mode_details[task][mode][skill],
                                )
                            )
            calendar_tasks = [
                (
                    self.cp_model.NewFixedSizeIntervalVar(
                        start=f["start"], size=f["duration"], name="calendar_res"
                    ),
                    f.get("value", 0),
                )
                for f in discr_calendar[skill]
                if f.get("value", 0) > 0
            ]
            self.cp_model.AddCumulative(
                [x[0] for x in intervals_consume] + [x[0] for x in calendar_tasks],
                [x[1] for x in intervals_consume] + [x[1] for x in calendar_tasks],
                capacity=int(np.max(dict_calendar_skills[skill])),
            )

    def constraint_redundant_cumulative_worker(
        self, one_skill_per_worker_per_task: bool = True
    ):
        some_employee = next(emp for emp in self.problem.employees)
        len_calendar = len(self.problem.employees[some_employee].calendar_employee)
        merged_calendar = np.zeros(len_calendar)
        for emp in self.problem.employees:
            merged_calendar += np.array(self.problem.employees[emp].calendar_employee)
        discr_calendar = discretize_calendar_(merged_calendar)
        intervals_consume = []
        max_skill_over_worker = {s: 0 for s in self.problem.skills_set}
        for emp in self.problem.employees:
            for s in self.problem.skills_set:
                max_skill_over_worker[s] = max(
                    max_skill_over_worker[s],
                    self.problem.employees[emp].get_skill_level(s),
                )
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                skills_needed = {
                    s: self.problem.mode_details[task][modes[0]].get(s, 0)
                    for s in self.problem.skills_set
                    if self.problem.mode_details[task][modes[0]].get(s, 0) > 0
                }
                if len(skills_needed) > 0:
                    lb_nb_worker_needed = max(
                        [
                            int(math.ceil(skills_needed[s] / max_skill_over_worker[s]))
                            for s in skills_needed
                        ]
                    )
                    if one_skill_per_worker_per_task:
                        lb_nb_worker_needed = sum(skills_needed.values())
                    intervals_consume.append(
                        (
                            self.variables["base_variable"]["intervals"][task],
                            lb_nb_worker_needed,
                        )
                    )
            else:
                for mode in modes:
                    skills_needed = {
                        s: self.problem.mode_details[task][modes[0]].get(s, 0)
                        for s in self.problem.skills_set
                        if self.problem.mode_details[task][modes[0]].get(s, 0) > 0
                    }
                    if len(skills_needed) > 0:
                        lb_nb_worker_needed = max(
                            [
                                int(
                                    math.ceil(
                                        skills_needed[s] / max_skill_over_worker[s]
                                    )
                                )
                                for s in skills_needed
                            ]
                        )
                        if one_skill_per_worker_per_task:
                            lb_nb_worker_needed = sum(skills_needed.values())
                        intervals_consume.append(
                            (
                                self.variables["mode_variable"]["opt_intervals"][task][
                                    mode
                                ],
                                lb_nb_worker_needed,
                            )
                        )
        calendar_tasks = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name="calendar_res"
                ),
                f.get("value", 0),
            )
            for f in discr_calendar
            if f.get("value", 0) > 0
        ]
        self.cp_model.AddCumulative(
            [x[0] for x in intervals_consume] + [x[0] for x in calendar_tasks],
            [x[1] for x in intervals_consume] + [x[1] for x in calendar_tasks],
            capacity=self.problem.nb_employees,
        )

    def constraint_precedence(self):
        for task in self.problem.successors:
            for succ in self.problem.successors[task]:
                self.cp_model.Add(
                    self.variables["base_variable"]["starts"][succ]
                    >= self.variables["base_variable"]["ends"][task]
                )


class CpsatAllocationSubproblem(OrtoolsCpSatSolver):

    problem: MultiskillRcpspProblem

    def __init__(self, problem: MultiskillRcpspProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}
        self.skills_matrix = compute_skills_matrix(problem=self.problem)
        self.skills_to_index = {
            problem.skills_list[i]: i for i in range(len(self.problem.skills_list))
        }
        self.fake_tasks, self.fake_tasks_unit = create_fake_tasks_multiskills(
            self.problem
        )
        self.base_solution: MultiskillRcpspSolution = None

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> MultiskillRcpspSolution:
        modes_dict = {}
        schedule = self.base_solution.schedule
        employee_usage = {}
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                modes_dict[task] = modes[0]
            else:
                for mode in self.variables["mode_variable"]["is_present"][task]:
                    if cpsolvercb.Value(
                        self.variables["mode_variable"]["is_present"][task][mode]
                    ):
                        modes_dict[task] = mode
                        break

        for task in self.problem.tasks_list:
            skills_needed = set(
                [
                    s
                    for s in self.problem.skills_set
                    if self.problem.mode_details[task][modes_dict[task]].get(s, 0) > 0
                ]
            )
            employee_usage[task] = {}
            if task in self.variables["worker_variable"]["worker_used"]:
                for worker in self.variables["worker_variable"]["worker_used"][task]:
                    if cpsolvercb.Value(
                        self.variables["worker_variable"]["worker_used"][task][worker]
                    ):
                        sk_nz = self.problem.employees[worker].get_non_zero_skills()
                        if "skills_used" in self.variables["worker_variable"]:
                            contrib = set()
                            for s in self.variables["worker_variable"]["skills_used"][
                                task
                            ][worker]:
                                if cpsolvercb.Value(
                                    self.variables["worker_variable"]["skills_used"][
                                        task
                                    ][worker][s]
                                ):
                                    contrib.add(s)
                        else:
                            contrib = set(sk_nz).intersection(skills_needed)
                        if len(contrib) > 0:
                            employee_usage[task][worker] = contrib
        sol = MultiskillRcpspSolution(
            problem=self.problem,
            schedule=schedule,
            modes=modes_dict,
            employee_usage=employee_usage,
        )
        return sol

    def init_model(self, base_solution: MultiskillRcpspSolution, **args: Any) -> None:
        optional_tasks = args.get("optional_tasks", False)
        one_skill_per_task = args.get("one_skill_per_task", True)
        starts = [base_solution.get_start_time(t) for t in self.problem.tasks_list]
        ends = [base_solution.get_end_time(t) for t in self.problem.tasks_list]
        self.base_solution = base_solution
        self.cp_model = CpModel()
        tasks_done_var = {}
        if optional_tasks:
            for task in self.problem.tasks_list:
                tasks_done_var[task] = self.cp_model.NewBoolVar(f"{task}_done")
        else:
            tasks_done_var = {t: 1 for t in self.problem.tasks_list}
        skills_used_var = {}
        worker_used_var = {}
        opt_interval_var = {}
        for index_task in range(self.problem.nb_tasks):
            task = self.problem.tasks_list[index_task]
            skills_of_task = set()
            mode = list(self.problem.mode_details[task].keys())[0]
            for skill in self.problem.skills_set:
                if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                    skills_of_task.add(skill)
            if len(skills_of_task) == 0:
                # no need of employees
                continue
            skills_used_var[task] = {}
            worker_used_var[task] = {}
            opt_interval_var[task] = {}
            for worker in self.problem.employees:
                skills_used_var[task][worker] = {}
                worker_used_var[task][worker] = self.cp_model.NewBoolVar(
                    name=f"used_{task}_{worker}"
                )
                opt_interval_var[task][worker] = self.cp_model.NewOptionalIntervalVar(
                    start=starts[index_task],
                    size=ends[index_task] - starts[index_task],
                    end=ends[index_task],
                    is_present=worker_used_var[task][worker],
                    name=f"opt_{task}_{worker}",
                )
                skills_of_worker = self.problem.employees[worker].get_non_zero_skills()
                for s in skills_of_task:
                    if s not in skills_of_worker:
                        continue
                    else:
                        skills_used_var[task][worker][s] = self.cp_model.NewBoolVar(
                            name=f"skill_{task}_{worker}_{s}"
                        )

                for s in skills_used_var[task][worker]:
                    self.cp_model.Add(
                        skills_used_var[task][worker][s]
                        <= worker_used_var[task][worker]
                    )
                self.cp_model.AddBoolOr(
                    [
                        skills_used_var[task][worker][s]
                        for s in skills_used_var[task][worker]
                    ]
                ).OnlyEnforceIf(worker_used_var[task][worker])
                if one_skill_per_task:
                    if len(skills_used_var[task][worker]) >= 1:
                        self.cp_model.AddAtMostOne(
                            [
                                skills_used_var[task][worker][s]
                                for s in skills_used_var[task][worker]
                            ]
                        )

        for worker in self.problem.employees:
            intervals_consume = []
            for task in opt_interval_var:
                if worker in opt_interval_var[task]:
                    intervals_consume.append(
                        (
                            opt_interval_var[task][worker],
                            1,
                        )
                    )
            calendar_tasks = [
                (
                    self.cp_model.NewFixedSizeIntervalVar(
                        start=f["start"], size=f["duration"], name="calendar_res"
                    ),
                    f.get(worker, 0),
                )
                for f in self.fake_tasks_unit
                if f.get(worker, 0) > 0
            ]
            self.cp_model.AddCumulative(
                [x[0] for x in intervals_consume] + [x[0] for x in calendar_tasks],
                [x[1] for x in intervals_consume] + [x[1] for x in calendar_tasks],
                capacity=1,
            )
        self.variables["worker_variable"] = {
            "worker_used": worker_used_var,
            "opt_intervals": opt_interval_var,
            "skills_used": skills_used_var,
        }
        self.variables["tasks_done"] = tasks_done_var
        self.create_skills_constraints_v2(optional_tasks=optional_tasks)
        if optional_tasks:
            self.cp_model.Maximize(
                sum(
                    [
                        self.variables["tasks_done"][t]
                        for t in self.variables["tasks_done"]
                    ]
                )
            )

    def create_skills_constraints_v2(self, **args):
        """
        using skills_used variable
        """
        exact_skill = args.get("exact_skill", False)
        optional_tasks = args.get("optional_tasks", False)
        for task in self.variables["worker_variable"]["worker_used"]:
            mode = list(self.problem.mode_details[task].keys())[0]
            skills_req = {
                s: self.problem.mode_details[task][mode].get(s, 0)
                for s in self.problem.skills_set
            }
            skills_req = {s: skills_req[s] for s in skills_req if skills_req[s] > 0}
            for s in skills_req:
                terms = []
                weights = []
                for worker in self.variables["worker_variable"]["worker_used"][task]:
                    if (
                        s
                        in self.variables["worker_variable"]["skills_used"][task][
                            worker
                        ]
                    ):
                        terms.append(
                            self.variables["worker_variable"]["skills_used"][task][
                                worker
                            ][s]
                        )
                        weights.append(
                            self.problem.employees[worker].get_skill_level(s)
                        )
                if exact_skill:
                    if optional_tasks:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights) == skills_req[s]
                        ).OnlyEnforceIf(self.variables["tasks_done"][task])
                    else:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights) == skills_req[s]
                        )
                else:
                    if optional_tasks:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights) >= skills_req[s]
                        ).OnlyEnforceIf(self.variables["tasks_done"][task])
                    else:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights) >= skills_req[s]
                        )


class CpmpyAllocationSubproblem(CpmpySolver):
    problem: MultiskillRcpspProblem

    def __init__(self, problem: MultiskillRcpspProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}
        self.skills_matrix = compute_skills_matrix(problem=self.problem)
        self.skills_to_index = {
            problem.skills_list[i]: i for i in range(len(self.problem.skills_list))
        }
        self.fake_tasks, self.fake_tasks_unit = create_fake_tasks_multiskills(
            self.problem
        )
        self.base_solution: MultiskillRcpspSolution = None
        self.meta_constraints = []
        self.soft_constraints = []
        self.hard_constraints = []

    def retrieve_current_solution(self) -> Solution:
        modes_dict = {}
        schedule = self.base_solution.schedule
        employee_usage = {}
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                modes_dict[task] = modes[0]
            else:
                for mode in self.variables["mode_variable"]["is_present"][task]:
                    if self.variables["mode_variable"]["is_present"][task][
                        mode
                    ].value():
                        modes_dict[task] = mode
                        break

        for task in self.problem.tasks_list:
            skills_needed = set(
                [
                    s
                    for s in self.problem.skills_set
                    if self.problem.mode_details[task][modes_dict[task]].get(s, 0) > 0
                ]
            )
            employee_usage[task] = {}
            if task in self.variables["worker_variable"]["worker_used"]:
                for worker in self.variables["worker_variable"]["worker_used"][task]:
                    if self.variables["worker_variable"]["worker_used"][task][
                        worker
                    ].value():
                        sk_nz = self.problem.employees[worker].get_non_zero_skills()
                        if "skills_used" in self.variables["worker_variable"]:
                            contrib = set()
                            for s in self.variables["worker_variable"]["skills_used"][
                                task
                            ][worker]:
                                if self.variables["worker_variable"]["skills_used"][
                                    task
                                ][worker][s].value():
                                    contrib.add(s)
                        else:
                            contrib = set(sk_nz).intersection(skills_needed)
                        if len(contrib) > 0:
                            employee_usage[task][worker] = contrib
        sol = MultiskillRcpspSolution(
            problem=self.problem,
            schedule=schedule,
            modes=modes_dict,
            employee_usage=employee_usage,
        )
        return sol

    def get_hard_meta_constraints(self) -> list[MetaCpmpyConstraint]:
        return [mc for mc in self.meta_constraints if mc.metadata["setting"] == "hard"]

    def get_soft_meta_constraints(self) -> list[MetaCpmpyConstraint]:
        return [mc for mc in self.meta_constraints if mc.metadata["setting"] == "soft"]

    def init_model(self, base_solution: MultiskillRcpspSolution, **args: Any) -> None:
        optional_tasks = args.get("optional_tasks", False)
        one_skill_per_task = args.get("one_skill_per_task", True)

        starts = [base_solution.get_start_time(t) for t in self.problem.tasks_list]
        ends = [base_solution.get_end_time(t) for t in self.problem.tasks_list]
        self.base_solution = base_solution
        self.model = cp.Model()
        tasks_done_var = {}
        if optional_tasks:
            for task in self.problem.tasks_list:
                tasks_done_var[task] = cp.boolvar(shape=1, name=f"{task}_done")
        else:
            tasks_done_var = {t: 1 for t in self.problem.tasks_list}
        skills_used_var = {}
        worker_used_var = {}
        opt_interval_var = {}
        for index_task in range(self.problem.nb_tasks):
            task = self.problem.tasks_list[index_task]
            skills_of_task = set()
            mode = list(self.problem.mode_details[task].keys())[0]
            for skill in self.problem.skills_set:
                if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                    skills_of_task.add(skill)
            if len(skills_of_task) == 0:
                # no need of employees
                continue
            skills_used_var[task] = {}
            worker_used_var[task] = {}
            opt_interval_var[task] = {}
            for worker in self.problem.employees:
                calendar = self.problem.employees[worker].calendar_employee
                if not all(
                    calendar[x] == 1
                    for x in range(starts[index_task], ends[index_task])
                ):
                    continue

                skills_used_var[task][worker] = {}
                worker_used_var[task][worker] = cp.boolvar(
                    1, name=f"used_{task}_{worker}"
                )
                skills_of_worker = self.problem.employees[worker].get_non_zero_skills()
                for s in skills_of_task:
                    if s not in skills_of_worker:
                        continue
                    else:
                        skills_used_var[task][worker][s] = cp.boolvar(
                            shape=1, name=f"skill_{task}_{worker}_{s}"
                        )
                constraints_skills_to_worker_used = []
                for s in skills_used_var[task][worker]:
                    constraints_skills_to_worker_used.append(
                        skills_used_var[task][worker][s]
                        <= worker_used_var[task][worker]
                    )

                skills_to_worker_used = cp.all(constraints_skills_to_worker_used)
                meta = MetaCpmpyConstraint(
                    name=f"link_{worker}_to_{task}",
                    constraints=[skills_to_worker_used],
                    metadata={"worker": worker, "task": task, "setting": "hard"},
                )
                self.meta_constraints.append(meta)
                self.model += skills_to_worker_used

                worker_implied_skills = worker_used_var[task][worker].implies(
                    cp.any(
                        [
                            skills_used_var[task][worker][s]
                            for s in skills_used_var[task][worker]
                        ]
                    )
                )
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"link_{worker}_to_{task}_2",
                        constraints=[worker_implied_skills],
                        metadata={"worker": worker, "task": task, "setting": "hard"},
                    )
                )
                self.model += worker_implied_skills
                if one_skill_per_task:
                    if len(skills_used_var[task][worker]) >= 1:
                        one_skill_per_worker = (
                            cp.sum(
                                [
                                    skills_used_var[task][worker][s]
                                    for s in skills_used_var[task][worker]
                                ]
                            )
                            <= 1
                        )
                        self.meta_constraints.append(
                            MetaCpmpyConstraint(
                                name=f"one_skill_{worker}_{task}",
                                constraints=[one_skill_per_worker],
                                metadata={
                                    "worker": worker,
                                    "task": task,
                                    "type": "one_skill_used",
                                    "setting": "hard",
                                },
                            )
                        )
                        self.model += one_skill_per_worker
        overlaps_constraint = []
        for worker in self.problem.employees:
            tasks = [t for t in worker_used_var if worker in worker_used_var[t]]
            for t in tasks:
                index_t = self.problem.index_task[t]
                start = starts[index_t]
                end = ends[index_t]
                overlapping_tasks = [
                    self.problem.tasks_list[i]
                    for i in range(self.problem.nb_tasks)
                    if starts[i] <= start < ends[i]
                    if worker in worker_used_var[self.problem.tasks_list[i]]
                ]

                overlap = (
                    cp.sum([worker_used_var[t][worker] for t in overlapping_tasks]) <= 1
                )
                overlaps_constraint.append(
                    MetaCpmpyConstraint(
                        name=f"no_overlap_{worker}_{t}",
                        constraints=[overlap],
                        metadata={
                            "task": t,
                            "overlapping": overlapping_tasks,
                            "worker": worker,
                            "setting": "hard",
                        },
                    )
                )
                # self.model += overlap
        for t in self.problem.tasks_list:
            ov = [
                overlaps
                for overlaps in overlaps_constraint
                if overlaps.metadata["task"] == t
            ]
            if len(ov) == 0:
                continue
            cnj = cp.all([ovv.constraints[0] for ovv in ov])
            if str(cnj) not in [str(c) for c in self.model.constraints]:
                self.model += cnj
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"no_overlap_{t}",
                        constraints=[cnj],
                        metadata={
                            "task": t,
                            "type": "overlap",
                            "overlapping": ov[0].metadata["overlapping"],
                            "setting": "soft",
                        },
                    )
                )

        self.variables["worker_variable"] = {
            "worker_used": worker_used_var,
            "opt_intervals": opt_interval_var,
            "skills_used": skills_used_var,
        }
        self.variables["tasks_done"] = tasks_done_var
        self.create_skills_constraints_v2(optional_tasks=optional_tasks)
        if optional_tasks:
            self.model.maximize(
                cp.sum(
                    [
                        self.variables["tasks_done"][t]
                        for t in self.variables["tasks_done"]
                    ]
                )
            )

    def create_skills_constraints_v2(self, **args):
        """
        using skills_used variable
        """
        exact_skill = args.get("exact_skill", False)
        slack_skill = args.get("slack_skill", False)
        optional_tasks = args.get("optional_tasks", False)
        if slack_skill:
            slack_skill_dict = {}
        for task in self.variables["worker_variable"]["worker_used"]:
            mode = list(self.problem.mode_details[task].keys())[0]
            skills_req = {
                s: self.problem.mode_details[task][mode].get(s, 0)
                for s in self.problem.skills_set
            }
            skills_req = {s: skills_req[s] for s in skills_req if skills_req[s] > 0}
            if slack_skill:
                slack_skill_dict[task] = {}
            for s in skills_req:
                if slack_skill:
                    slack_skill_dict[task][s] = cp.intvar(
                        lb=0, ub=5, name=f"slack_{task}_{s}", shape=1
                    )
                terms = []
                weights = []
                for worker in self.variables["worker_variable"]["worker_used"][task]:
                    if (
                        s
                        in self.variables["worker_variable"]["skills_used"][task][
                            worker
                        ]
                    ):
                        terms.append(
                            self.variables["worker_variable"]["skills_used"][task][
                                worker
                            ][s]
                        )
                        weights.append(
                            self.problem.employees[worker].get_skill_level(s)
                        )
                if exact_skill:
                    if not slack_skill:
                        if optional_tasks:
                            c = self.variables["tasks_done"][task].implies(
                                cp.sum([w * t for w, t in zip(weights, terms)])
                                == skills_req[s]
                            )
                        else:
                            c = (
                                cp.sum([w * t for w, t in zip(weights, terms)])
                                == skills_req[s]
                            )
                    else:
                        if optional_tasks:
                            c = self.variables["tasks_done"][task].implies(
                                cp.sum([w * t for w, t in zip(weights, terms)])
                                == skills_req[s] + slack_skill_dict[task][s]
                            )
                        else:
                            c = (
                                cp.sum([w * t for w, t in zip(weights, terms)])
                                == skills_req[s] + slack_skill_dict[task][s]
                            )
                else:
                    if not slack_skill:
                        if optional_tasks:
                            c = self.variables["tasks_done"][task].implies(
                                cp.sum([w * t for w, t in zip(weights, terms)])
                                >= skills_req[s]
                            )
                        else:
                            c = (
                                cp.sum([w * t for w, t in zip(weights, terms)])
                                >= skills_req[s]
                            )
                    else:
                        if optional_tasks:
                            c = self.variables["tasks_done"][task].implies(
                                cp.sum([w * t for w, t in zip(weights, terms)])
                                >= skills_req[s] + slack_skill_dict[task][s]
                            )
                        else:
                            c = (
                                cp.sum([w * t for w, t in zip(weights, terms)])
                                >= skills_req[s] + slack_skill_dict[task][s]
                            )
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"skill_{s}_{task}_fulfilled",
                        constraints=[c],
                        metadata={
                            "skill": s,
                            "task": task,
                            "type": "skills",
                            "setting": "hard",
                        },
                    )
                )
                self.model += c

        if slack_skill:
            self.variables["slack_skill_var"] = slack_skill_dict


class CpmpyAllocationSubproblemExplainer(CpmpyAllocationSubproblem):
    """Here the overlap constraint are done 2 by 2,
    this can be usefull to detect chain of task that causes infeasibilities"""

    def init_model(self, base_solution: MultiskillRcpspSolution, **args: Any) -> None:
        optional_tasks = args.get("optional_tasks", False)
        one_skill_per_task = args.get("one_skill_per_task", True)

        starts = [base_solution.get_start_time(t) for t in self.problem.tasks_list]
        ends = [base_solution.get_end_time(t) for t in self.problem.tasks_list]
        self.base_solution = base_solution
        self.model = cp.Model()
        tasks_done_var = {}
        if optional_tasks:
            for task in self.problem.tasks_list:
                tasks_done_var[task] = cp.boolvar(shape=1, name=f"{task}_done")
        else:
            tasks_done_var = {t: 1 for t in self.problem.tasks_list}
        skills_used_var = {}
        worker_used_var = {}
        opt_interval_var = {}
        for index_task in range(self.problem.nb_tasks):
            task = self.problem.tasks_list[index_task]
            skills_of_task = set()
            mode = list(self.problem.mode_details[task].keys())[0]
            for skill in self.problem.skills_set:
                if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                    skills_of_task.add(skill)
            if len(skills_of_task) == 0:
                # no need of employees
                continue
            skills_used_var[task] = {}
            worker_used_var[task] = {}
            opt_interval_var[task] = {}
            for worker in self.problem.employees:
                calendar = self.problem.employees[worker].calendar_employee
                if not all(
                    calendar[x] == 1
                    for x in range(starts[index_task], ends[index_task])
                ):
                    continue

                skills_used_var[task][worker] = {}
                worker_used_var[task][worker] = cp.boolvar(
                    1, name=f"used_{task}_{worker}"
                )
                skills_of_worker = self.problem.employees[worker].get_non_zero_skills()
                for s in skills_of_task:
                    if s not in skills_of_worker:
                        continue
                    else:
                        skills_used_var[task][worker][s] = cp.boolvar(
                            shape=1, name=f"skill_{task}_{worker}_{s}"
                        )
                constraints_skills_to_worker_used = []
                for s in skills_used_var[task][worker]:
                    constraints_skills_to_worker_used.append(
                        skills_used_var[task][worker][s]
                        <= worker_used_var[task][worker]
                    )

                skills_to_worker_used = cp.all(constraints_skills_to_worker_used)
                meta = MetaCpmpyConstraint(
                    name=f"link_{worker}_to_{task}",
                    constraints=[skills_to_worker_used],
                    metadata={"worker": worker, "task": task, "setting": "hard"},
                )
                self.meta_constraints.append(meta)
                self.model += skills_to_worker_used

                worker_implied_skills = worker_used_var[task][worker].implies(
                    cp.any(
                        [
                            skills_used_var[task][worker][s]
                            for s in skills_used_var[task][worker]
                        ]
                    )
                )
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"link_{worker}_to_{task}_2",
                        constraints=[worker_implied_skills],
                        metadata={"worker": worker, "task": task, "setting": "hard"},
                    )
                )
                self.model += worker_implied_skills
                if one_skill_per_task:
                    if len(skills_used_var[task][worker]) >= 1:
                        one_skill_per_worker = (
                            cp.sum(
                                [
                                    skills_used_var[task][worker][s]
                                    for s in skills_used_var[task][worker]
                                ]
                            )
                            <= 1
                        )
                        self.meta_constraints.append(
                            MetaCpmpyConstraint(
                                name=f"one_skill_{worker}_{task}",
                                constraints=[one_skill_per_worker],
                                metadata={
                                    "worker": worker,
                                    "task": task,
                                    "type": "one_skill_used",
                                    "setting": "hard",
                                },
                            )
                        )
                        self.model += one_skill_per_worker

        for t in self.problem.tasks_list:
            if t not in worker_used_var:
                continue
            index_t = self.problem.index_task[t]
            start = starts[index_t]
            end = ends[index_t]
            overlapping_tasks = [
                self.problem.tasks_list[i]
                for i in range(self.problem.nb_tasks)
                if starts[i] <= start < ends[i]
                if i != index_t
            ]
            for ot in overlapping_tasks:
                if ot not in worker_used_var:
                    continue
                worker_commons = [
                    w
                    for w in self.problem.employees
                    if w in worker_used_var[t] and w in worker_used_var[ot]
                ]
                overlaps = cp.all(
                    [
                        worker_used_var[t][w] + worker_used_var[ot][w] <= 1
                        for w in worker_commons
                    ]
                )
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"no_overlap_{t}_{ot}",
                        constraints=[overlaps],
                        metadata={
                            "task": t,
                            "type": "overlap",
                            "overlapping": (ot, t),
                            "setting": "hard",
                        },
                    )
                )
                self.model += overlaps

        self.variables["worker_variable"] = {
            "worker_used": worker_used_var,
            "opt_intervals": opt_interval_var,
            "skills_used": skills_used_var,
        }
        self.variables["tasks_done"] = tasks_done_var
        self.create_skills_constraints_v2(optional_tasks=optional_tasks)
        if optional_tasks:
            self.model.maximize(
                cp.sum(
                    [
                        self.variables["tasks_done"][t]
                        for t in self.variables["tasks_done"]
                    ]
                )
            )


class CpmpyAllocationPreemptiveOperatorSubproblem(CpmpySolver):
    problem: MultiskillRcpspProblem

    def __init__(self, problem: MultiskillRcpspProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}
        self.skills_matrix = compute_skills_matrix(problem=self.problem)
        self.skills_to_index = {
            problem.skills_list[i]: i for i in range(len(self.problem.skills_list))
        }
        self.fake_tasks, self.fake_tasks_unit = create_fake_tasks_multiskills(
            self.problem
        )
        self.base_solution: MultiskillRcpspSolution = None
        self.slots_event_per_task: dict[int, list[tuple[int, int]]] = None
        self.meta_constraints = []
        self.soft_constraints = []
        self.hard_constraints = []
        self.optional_tasks: bool = False

    def retrieve_current_solution(self) -> PreemptiveMultiskillRcpspSolution:
        if self.optional_tasks:
            print(
                sum(
                    self.variables["tasks_done"][t].value()
                    for t in self.variables["tasks_done"]
                ),
                " tasks done over ",
                self.problem.nb_tasks,
            )
        schedule = {
            self.problem.tasks_list[i]: {
                "starts": [x[0] for x in self.slots_event_per_task[i]],
                "ends": [x[1] for x in self.slots_event_per_task[i]],
            }
            for i in range(self.problem.nb_tasks)
        }
        employee_usage = {}
        modes_dict = {}
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                modes_dict[task] = modes[0]
            else:
                for mode in self.variables["mode_variable"]["is_present"][task]:
                    if self.variables["mode_variable"]["is_present"][task][
                        mode
                    ].value():
                        modes_dict[task] = mode
                        break

        for task in self.problem.tasks_list:
            index_task = self.problem.index_task[task]
            skills_needed = set(
                [
                    s
                    for s in self.problem.skills_set
                    if self.problem.mode_details[task][modes_dict[task]].get(s, 0) > 0
                ]
            )
            employee_usage[task] = []
            for index_subtask in range(len(self.slots_event_per_task[index_task])):
                contrib_worker = {}
                if (task, index_subtask) in self.variables["worker_variable"][
                    "worker_used"
                ]:
                    for worker in self.variables["worker_variable"]["worker_used"][
                        (task, index_subtask)
                    ]:
                        if self.variables["worker_variable"]["worker_used"][
                            (task, index_subtask)
                        ][worker].value():
                            sk_nz = self.problem.employees[worker].get_non_zero_skills()
                            if "skills_used" in self.variables["worker_variable"]:
                                contrib = set()
                                for s in self.variables["worker_variable"][
                                    "skills_used"
                                ][(task, index_subtask)][worker]:
                                    if self.variables["worker_variable"]["skills_used"][
                                        (task, index_subtask)
                                    ][worker][s].value():
                                        contrib.add(s)
                            else:
                                contrib = set(sk_nz).intersection(skills_needed)
                            if len(contrib) > 0:
                                contrib_worker[worker] = contrib
                employee_usage[task].append(contrib_worker)
        return PreemptiveMultiskillRcpspSolution(
            problem=self.problem,
            modes=modes_dict,
            schedule=schedule,
            employee_usage=employee_usage,
        )

    def discrete_events_schedule(self, base_solution: MultiskillRcpspSolution):
        starts = [base_solution.get_start_time(t) for t in self.problem.tasks_list]
        set_starts = list(set(starts))
        # sorted_starts = sorted(starts)
        ends = [base_solution.get_end_time(t) for t in self.problem.tasks_list]
        slots_event_per_task = {i: [] for i in range(self.problem.nb_tasks)}
        for i in range(self.problem.nb_tasks):
            start = starts[i]
            end = ends[i]
            starts_event = sorted([st for st in set_starts if start < st < end])
            if len(starts_event) > 0:
                slots_event_per_task[i] = [(start, starts_event[0])]
                for j in range(1, len(starts_event)):
                    slots_event_per_task[i].append(
                        (starts_event[j - 1], starts_event[j])
                    )
                slots_event_per_task[i].append((starts_event[-1], end))
            else:
                slots_event_per_task[i] = [(start, end)]
        return slots_event_per_task

    def init_model(self, base_solution: MultiskillRcpspSolution, **kwargs: Any) -> None:
        optional_tasks = kwargs.get("optional_tasks", False)
        one_skill_per_task = kwargs.get("one_skill_per_task", True)
        self.optional_tasks = optional_tasks
        self.base_solution = base_solution
        self.model = cp.Model()
        slots_event_per_task: dict[
            int, list[tuple[int, int]]
        ] = self.discrete_events_schedule(base_solution=base_solution)
        self.slots_event_per_task = slots_event_per_task
        starts = [base_solution.get_start_time(t) for t in self.problem.tasks_list]
        ends = [base_solution.get_end_time(t) for t in self.problem.tasks_list]
        self.base_solution = base_solution
        self.model = cp.Model()
        tasks_done_var = {}
        skills_used_var = {}
        worker_used_var = {}
        opt_interval_var = {}
        if optional_tasks:
            for task in self.problem.tasks_list:
                tasks_done_var[task] = cp.boolvar(shape=1, name=f"{task}_done")
        else:
            tasks_done_var = {t: 1 for t in self.problem.tasks_list}
        for index_task in range(self.problem.nb_tasks):
            task = self.problem.tasks_list[index_task]
            skills_of_task = set()
            mode = list(self.problem.mode_details[task].keys())[0]
            for skill in self.problem.skills_set:
                if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                    skills_of_task.add(skill)
            if len(skills_of_task) == 0:
                # no need of employees
                continue
            slots = slots_event_per_task[index_task]
            for index in range(len(slots)):
                st, end = slots[index]
                skills_used_var[(task, index)] = {}
                worker_used_var[(task, index)] = {}
                opt_interval_var[(task, index)] = {}
                for worker in self.problem.employees:
                    calendar = self.problem.employees[worker].calendar_employee
                    if not all(calendar[x] == 1 for x in range(st, end)):
                        continue
                    skills_used_var[(task, index)][worker] = {}
                    worker_used_var[(task, index)][worker] = cp.boolvar(
                        1, name=f"used_{task,index}_{worker}"
                    )
                    skills_of_worker = self.problem.employees[
                        worker
                    ].get_non_zero_skills()
                    for s in skills_of_task:
                        if s not in skills_of_worker:
                            continue
                        else:
                            skills_used_var[(task, index)][worker][s] = cp.boolvar(
                                shape=1, name=f"skill_{task,index}_{worker}_{s}"
                            )
                    constraints_skills_to_worker_used = []
                    for s in skills_used_var[(task, index)][worker]:
                        constraints_skills_to_worker_used.append(
                            skills_used_var[(task, index)][worker][s]
                            <= worker_used_var[(task, index)][worker]
                        )

                    skills_to_worker_used = cp.all(constraints_skills_to_worker_used)
                    meta = MetaCpmpyConstraint(
                        name=f"link_{(task, index)}_to_{task,index}",
                        constraints=[skills_to_worker_used],
                        metadata={
                            "worker": worker,
                            "task": task,
                            "part": index,
                            "slot": (st, end),
                            "type": "link_task",
                            "setting": "hard",
                        },
                    )
                    self.meta_constraints.append(meta)
                    self.model += skills_to_worker_used

                    worker_implied_skills = worker_used_var[(task, index)][
                        worker
                    ].implies(
                        cp.any(
                            [
                                skills_used_var[(task, index)][worker][s]
                                for s in skills_used_var[(task, index)][worker]
                            ]
                        )
                    )
                    self.meta_constraints.append(
                        MetaCpmpyConstraint(
                            name=f"link_{worker}_to_{task,index}_2",
                            constraints=[worker_implied_skills],
                            metadata={
                                "worker": worker,
                                "task": task,
                                "part": index,
                                "slot": (st, end),
                                "type": "link",
                                "setting": "hard",
                            },
                        )
                    )
                    self.model += worker_implied_skills
                    if one_skill_per_task:
                        if len(skills_used_var[(task, index)][worker]) >= 1:
                            one_skill_per_worker = (
                                cp.sum(
                                    [
                                        skills_used_var[(task, index)][worker][s]
                                        for s in skills_used_var[(task, index)][worker]
                                    ]
                                )
                                <= 1
                            )
                            self.meta_constraints.append(
                                MetaCpmpyConstraint(
                                    name=f"one_skill_{worker}_{(task, index)}",
                                    constraints=[one_skill_per_worker],
                                    metadata={
                                        "worker": worker,
                                        "task": task,
                                        "part": index,
                                        "slot": (st, end),
                                        "type": "one_skill_used",
                                        "setting": "hard",
                                    },
                                )
                            )
                            self.model += one_skill_per_worker
        overlaps_constraint = []
        for worker in self.problem.employees:
            tasks = [
                t[0] for t in worker_used_var if worker in worker_used_var[(t[0], 0)]
            ]
            for t in tasks:
                index_t = self.problem.index_task[t]
                slots = slots_event_per_task[index_t]
                for index_subtask in range(len(slots)):
                    st, end = slots_event_per_task[index_t][index_subtask]
                    overlapping_tasks = [
                        (self.problem.tasks_list[index_task], index_sub)
                        for index_task in slots_event_per_task
                        for index_sub in range(len(slots_event_per_task[index_task]))
                        if st <= slots_event_per_task[index_task][index_sub][0] < end
                        and (
                            (self.problem.tasks_list[index_task], index_sub)
                            in worker_used_var
                            and worker
                            in worker_used_var[
                                (self.problem.tasks_list[index_task], index_sub)
                            ]
                        )
                    ]
                    overlap = (
                        cp.sum([worker_used_var[t][worker] for t in overlapping_tasks])
                        <= 1
                    )
                    overlaps_constraint.append(
                        MetaCpmpyConstraint(
                            name=f"no_overlap_{worker}_{t}",
                            constraints=[overlap],
                            metadata={
                                "task": t,
                                "overlapping": overlapping_tasks,
                                "part": index_subtask,
                                "slot": (st, end),
                                "worker": worker,
                                "setting": "hard",
                            },
                        )
                    )
                # self.model += overlap
        for t in self.problem.tasks_list:
            ov = [
                overlaps
                for overlaps in overlaps_constraint
                if overlaps.metadata["task"] == t
            ]
            if len(ov) == 0:
                continue
            cnj = cp.all([ovv.constraints[0] for ovv in ov])
            # print(ov)
            if str(cnj) not in [str(c) for c in self.model.constraints]:
                self.model += cnj
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"no_overlap_{t}",
                        constraints=[cnj],
                        metadata={
                            "task": t,
                            "type": "overlap",
                            "overlapping": ov[0].metadata["overlapping"],
                            "setting": "soft",
                        },
                    )
                )
        self.variables["worker_variable"] = {
            "worker_used": worker_used_var,
            "opt_intervals": opt_interval_var,
            "skills_used": skills_used_var,
        }
        self.variables["tasks_done"] = tasks_done_var
        self.create_skills_constraints_v2(optional_tasks=optional_tasks)
        if optional_tasks:
            self.model.maximize(
                cp.sum(
                    [
                        self.variables["tasks_done"][t]
                        for t in self.variables["tasks_done"]
                    ]
                )
            )

    def create_skills_constraints_v2(self, **args):
        """
        using skills_used variable
        """
        exact_skill = args.get("exact_skill", False)
        optional_tasks = args.get("optional_tasks", False)
        for task, sub in self.variables["worker_variable"]["worker_used"]:
            mode = list(self.problem.mode_details[task].keys())[0]
            skills_req = {
                s: self.problem.mode_details[task][mode].get(s, 0)
                for s in self.problem.skills_set
            }
            skills_req = {s: skills_req[s] for s in skills_req if skills_req[s] > 0}
            for s in skills_req:
                terms = []
                weights = []
                for worker in self.variables["worker_variable"]["worker_used"][
                    (task, sub)
                ]:
                    if (
                        s
                        in self.variables["worker_variable"]["skills_used"][
                            (task, sub)
                        ][worker]
                    ):
                        terms.append(
                            self.variables["worker_variable"]["skills_used"][
                                (task, sub)
                            ][worker][s]
                        )
                        weights.append(
                            self.problem.employees[worker].get_skill_level(s)
                        )
                if exact_skill:
                    if optional_tasks:
                        c = self.variables["tasks_done"][task].implies(
                            cp.sum([w * t for w, t in zip(weights, terms)])
                            == skills_req[s]
                        )
                    else:
                        c = (
                            cp.sum([w * t for w, t in zip(weights, terms)])
                            == skills_req[s]
                        )
                else:
                    if optional_tasks:
                        c = self.variables["tasks_done"][task].implies(
                            cp.sum([w * t for w, t in zip(weights, terms)])
                            >= skills_req[s]
                        )
                    else:
                        c = (
                            cp.sum([w * t for w, t in zip(weights, terms)])
                            >= skills_req[s]
                        )
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"skill_{s}_{task, sub}_fulfilled",
                        constraints=[c],
                        metadata={
                            "skill": s,
                            "task": task,
                            "part": sub,
                            "type": "skills",
                            "setting": "hard",
                        },
                    )
                )
                self.model += c
