#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Cp-sat model for the preemptive rcpsp problem.
import logging
from typing import Any, Iterable

import numpy as np
from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    Domain,
    IntervalVar,
    LinearExpr,
    LinearExprT,
)

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat.calendar_preemptive import (
    CalendarPreemptiveCpSatSolver,
    CumulativeResource,
    ModelingPreemptive,
    OtherCalendarResource,
    Task,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.utils import create_fake_tasks
from discrete_optimization.rcpsp_preemptive.problem import (
    PreemptiveRcpspProblem,
    PreemptiveRcpspSolution,
)

logger = logging.getLogger(__name__)


class CpSatPreemptiveRcpspSolver(OrtoolsCpSatSolver, WarmstartMixin):
    problem: PreemptiveRcpspProblem

    def __init__(
        self,
        problem: PreemptiveRcpspProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.create_preempt_variables(
            max_nb_preemption=kwargs.get("max_nb_preemption", None)
        )
        self.constraint_convention_variables()
        self.create_modes_variables()
        self.create_resource_consumption_variables()
        self.create_duration_variables()
        self.constraint_variable_to_duration()
        self.constraint_precedence()
        self.constraint_resource()
        self.variables["objectives"] = {
            "makespan": self.variables["ends"][self.problem.sink_task][0],
            "nb_preemption": sum(
                [
                    self.variables["presences"][t][i]
                    for t in self.variables["presences"]
                    for i in range(len(self.variables["presences"][t]))
                ]
            ),
        }
        self.cp_model.minimize(self.variables["ends"][self.problem.sink_task][0])

    def implements_lexico_api(self) -> bool:
        return True

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        if obj == "makespan":
            sol: PreemptiveRcpspSolution = res[-1][0]
            return sol.get_max_end_time()
        if obj == "nb_preemption":
            sol: PreemptiveRcpspSolution = res[-1][0]
            return sum(
                [sol.get_number_of_part(task) for task in self.problem.tasks_list]
            )

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.minimize(self.variables["objectives"][obj])

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.variables["objectives"].keys())

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        self.cp_model.add(self.variables["objectives"][obj] <= value)

    def create_modes_variables(self):
        modes_dict = {}
        for t in self.problem.tasks_list:
            modes_dict[t] = {}
            modes = list(self.problem.mode_details[t].keys())
            nb_modes = len(self.problem.mode_details[t])
            if nb_modes == 1:
                modes_dict[t][modes[0]] = 1
            else:
                for m in modes:
                    modes_dict[t][m] = self.cp_model.NewBoolVar(name=f"mode_{t}_{m}")
                self.cp_model.add_exactly_one([modes_dict[t][m] for m in modes])
        self.variables["modes"] = modes_dict

    def create_resource_consumption_variables(self):
        modes_var = self.variables["modes"]
        resource_consumption_dict = {}
        for t in self.problem.tasks_list:
            resource_consumption_dict[t] = {}
            modes = list(self.problem.mode_details[t].keys())
            nb_modes = len(self.problem.mode_details[t])
            if nb_modes == 1:
                for r in self.problem.resources_list:
                    cons = self.problem.mode_details[t][modes[0]].get(r, 0)
                    if cons > 0:
                        resource_consumption_dict[t][r] = cons
            else:
                potential_resources = set(
                    [
                        r
                        for r in self.problem.resources_list
                        if any(
                            self.problem.mode_details[t][m].get(r, 0) > 0 for m in modes
                        )
                    ]
                )
                for r in potential_resources:
                    values = [self.problem.mode_details[t][m].get(r, 0) for m in modes]
                    resource_consumption_dict[t][r] = self.cp_model.NewIntVar(
                        lb=min(values),
                        ub=max(values),
                        name=f"resource_consumption_{t}_{r}",
                    )
                for m in modes_var[t]:
                    for r in potential_resources:
                        cons = self.problem.mode_details[t][m].get(r, 0)
                        self.cp_model.add(
                            resource_consumption_dict[t][r] == cons
                        ).only_enforce_if(modes_var[t][m])
        self.variables["resource_consumption"] = resource_consumption_dict

    def create_duration_variables(self):
        modes_var = self.variables["modes"]
        duration_dict = {}
        for t in self.problem.tasks_list:
            modes = list(self.problem.mode_details[t].keys())
            nb_modes = len(self.problem.mode_details[t])
            if nb_modes == 1:
                duration_dict[t] = self.problem.mode_details[t][modes[0]]["duration"]
            else:
                potential_durations = list(
                    set([self.problem.mode_details[t][m]["duration"] for m in modes])
                )
                duration_dict[t] = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromValues(potential_durations), name=f"duration_{t}"
                )
                for m in modes_var[t]:
                    dur = self.problem.mode_details[t][m]["duration"]
                    self.cp_model.add(duration_dict[t] == dur).only_enforce_if(
                        modes_var[t][m]
                    )
        self.variables["duration"] = duration_dict

    def create_preempt_variables(self, max_nb_preemption: int | None = None) -> None:
        starts = {}
        durations = {}
        ends = {}
        presences = {}
        intervals = {}
        for t in self.problem.tasks_list:
            possible_durations = [
                self.problem.mode_details[t][m]["duration"]
                for m in self.problem.mode_details[t]
            ]
            max_duration = max(possible_durations)
            if max_nb_preemption is None:
                nb_preemption = max_duration + 1  # Naive
            else:
                nb_preemption = min(max_nb_preemption, max_duration + 1)
            if not self.problem.preemptive_indicator[t]:
                nb_preemption = 1
            starts[t] = [
                self.cp_model.NewIntVar(
                    lb=0, ub=self.problem.horizon, name=f"start_{t}_{i}"
                )
                for i in range(nb_preemption)
            ]
            ends[t] = [
                self.cp_model.NewIntVar(
                    lb=0, ub=self.problem.horizon, name=f"end_{t}_{i}"
                )
                for i in range(nb_preemption)
            ]
            # min_duration_preempt = 1
            # if max_duration == 0:
            min_duration_preempt = 0
            durations[t] = [
                self.cp_model.NewIntVar(
                    lb=min_duration_preempt, ub=max_duration, name=f"duration_{t}_{i}"
                )
                for i in range(nb_preemption)
            ]
            presences[t] = [
                self.cp_model.NewBoolVar(name=f"presence_{t}_{i}")
                for i in range(nb_preemption)
            ]
            intervals[t] = [
                self.cp_model.NewOptionalIntervalVar(
                    start=starts[t][i],
                    end=ends[t][i],
                    size=durations[t][i],
                    is_present=presences[t][i],
                    name=f"interval_{t}_{i}",
                )
                for i in range(nb_preemption)
            ]
        self.variables["starts"] = starts
        self.variables["durations"] = durations
        self.variables["ends"] = ends
        self.variables["presences"] = presences
        self.variables["intervals"] = intervals

    def constraint_convention_variables(self):
        for t in self.variables["presences"]:
            nb_preemption = len(self.variables["presences"][t])
            self.cp_model.add(self.variables["presences"][t][0] == 1)
            modes = list(self.problem.mode_details[t].keys())
            potential_durations = list(
                set([self.problem.mode_details[t][m]["duration"] for m in modes])
            )
            if min(potential_durations) > 0:
                self.cp_model.add(self.variables["durations"][t][0] >= 1)
            for i in range(nb_preemption - 1):
                # Ordered intervals and present until some point, then all absent.
                self.cp_model.add(
                    self.variables["presences"][t][i]
                    >= self.variables["presences"][t][i + 1]
                )
                self.cp_model.add(
                    self.variables["ends"][t][i] <= self.variables["starts"][t][i + 1]
                )
                self.cp_model.add(
                    self.variables["ends"][t][i] <= self.variables["ends"][t][i + 1]
                )
                (
                    self.cp_model.add(
                        self.variables["ends"][t][i]
                        < self.variables["starts"][t][i + 1]
                    ).only_enforce_if(self.variables["presences"][t][i + 1])
                )

            for i in range(1, nb_preemption):
                self.cp_model.add(
                    self.variables["durations"][t][i] >= 1
                ).only_enforce_if(self.variables["presences"][t][i])
                self.cp_model.add(
                    self.variables["durations"][t][i] == 0
                ).only_enforce_if(self.variables["presences"][t][i].Not())
                self.cp_model.add(
                    self.variables["starts"][t][i] == self.variables["ends"][t][i - 1]
                ).only_enforce_if(self.variables["presences"][t][i].Not())
                self.cp_model.add(
                    self.variables["ends"][t][i] == self.variables["ends"][t][i - 1]
                ).only_enforce_if(self.variables["presences"][t][i].Not())

    def constraint_variable_to_duration(self):
        for t in self.variables["presences"]:
            nb_preemption = len(self.variables["presences"][t])
            self.cp_model.add(
                sum(self.variables["durations"][t][i] for i in range(nb_preemption))
                == self.variables["duration"][t]
            )

    def constraint_precedence(self):
        for t in self.problem.successors:
            for succ in self.problem.successors[t]:
                self.cp_model.add(
                    self.variables["starts"][succ][0] >= self.variables["ends"][t][-1]
                )

    def constraint_resource(self):
        fake_tasks = create_fake_tasks(self.problem)
        for r in self.problem.resources:
            if r not in self.problem.non_renewable_resources:
                self.constraint_resource_cumulative(resource=r, fake_tasks=fake_tasks)
            else:
                self.constraint_resource_non_renewable(resource=r)

    def constraint_resource_cumulative(
        self, resource: str, fake_tasks: list[dict[str, int]]
    ):
        potential_tasks = [
            t
            for t in self.variables["resource_consumption"]
            if resource in self.variables["resource_consumption"][t]
        ]
        intervals = [
            (
                self.variables["intervals"][t][i],
                self.variables["resource_consumption"][t][resource],
            )
            for t in potential_tasks
            for i in range(len(self.variables["intervals"][t]))
        ]
        fake_tasks_of_interest = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name=f"res_"
                ),
                f.get(resource, 0),
            )
            for f in fake_tasks
            if f.get(resource, 0) > 0
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.add_cumulative(
            [x[0] for x in intervals + fake_tasks_of_interest],
            [x[1] for x in intervals + fake_tasks_of_interest],
            capa,
        )

    def constraint_resource_non_renewable(self, resource: str):
        potential_tasks = [
            t
            for t in self.variables["resource_consumption"]
            if resource in self.variables["resource_consumption"][t]
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.add(
            sum(
                [
                    self.variables["resource_consumption"][t][resource]
                    for t in potential_tasks
                ]
            )
            <= capa
        )

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> PreemptiveRcpspSolution:
        modes_dict = {}
        schedule = {}
        for t in self.variables["starts"]:
            sched = []
            for i in range(len(self.variables["starts"][t])):
                present = cpsolvercb.value(self.variables["presences"][t][i])
                if present:
                    sched.append(
                        (
                            cpsolvercb.value(self.variables["starts"][t][i]),
                            cpsolvercb.value(self.variables["ends"][t][i]),
                        )
                    )
                else:
                    break

            # For duration-0 tasks, record the time point even though no parts are present
            if len(sched) == 0:
                time_point = cpsolvercb.value(self.variables["starts"][t][0])
                schedule[t] = {
                    "starts": [time_point],
                    "ends": [time_point],
                }
            else:
                schedule[t] = {
                    "starts": [x[0] for x in sched],
                    "ends": [x[1] for x in sched],
                }

            modes = list(self.variables["modes"][t].keys())
            if len(modes) == 1:
                modes_dict[t] = modes[0]
            else:
                for m in self.variables["modes"][t]:
                    if cpsolvercb.value(self.variables["modes"][t][m]):
                        modes_dict[t] = m
        modes = [
            modes_dict[t]
            for t in self.problem.tasks_list
            if t not in {self.problem.source_task, self.problem.sink_task}
        ]
        return PreemptiveRcpspSolution(
            problem=self.problem, rcpsp_schedule=schedule, rcpsp_modes=modes
        )

    def set_warm_start(self, solution: PreemptiveRcpspSolution) -> None:
        """Make the solver warm start from the given solution."""
        self.cp_model.clear_hints()
        for task in self.variables["starts"]:
            starts = solution.rcpsp_schedule[task]["starts"]
            ends = solution.rcpsp_schedule[task]["ends"]
            # Set hints for each preemption part
            for i, (st, end) in enumerate(zip(starts, ends)):
                if i < len(self.variables["starts"][task]):
                    self.cp_model.AddHint(self.variables["starts"][task][i], st)
                    self.cp_model.AddHint(self.variables["ends"][task][i], end)
                    self.cp_model.AddHint(self.variables["presences"][task][i], 1)
            # Set remaining parts as not present
            for i in range(len(starts), len(self.variables["starts"][task])):
                self.cp_model.AddHint(self.variables["presences"][task][i], 0)
        # Set mode hints
        for task, mode in zip(self.problem.tasks_list_non_dummy, solution.rcpsp_modes):
            if len(self.variables["modes"][task]) > 1:
                for m in self.variables["modes"][task]:
                    self.cp_model.AddHint(
                        self.variables["modes"][task][m], 1 if m == mode else 0
                    )


class CpSatPreemptiveRcpspSolverUnitTime(OrtoolsCpSatSolver, WarmstartMixin):
    problem: PreemptiveRcpspProblem

    def __init__(
        self,
        problem: PreemptiveRcpspProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.create_modes_variables()
        self.create_preempt_variables()
        self.constraint_convention_variables()
        self.create_resource_consumption_variables()
        self.constraint_precedence()
        self.constraint_resource()
        self.variables["objectives"] = {
            "makespan": self.variables["ends"][self.problem.sink_task][0]
        }
        self.cp_model.minimize(self.variables["ends"][self.problem.sink_task][0])

    def implements_lexico_api(self) -> bool:
        return True

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        if obj == "makespan":
            sol: PreemptiveRcpspSolution = res[-1][0]
            return sol.get_max_end_time()
        return None

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.minimize(self.variables["objectives"][obj])

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.variables["objectives"].keys())

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        self.cp_model.add(self.variables["objectives"][obj] <= value)

    def create_modes_variables(self):
        modes_dict = {}
        for t in self.problem.tasks_list:
            modes_dict[t] = {}
            modes = list(self.problem.mode_details[t].keys())
            nb_modes = len(self.problem.mode_details[t])
            if nb_modes == 1:
                modes_dict[t][modes[0]] = 1
            else:
                for m in modes:
                    modes_dict[t][m] = self.cp_model.NewBoolVar(name=f"mode_{t}_{m}")
                self.cp_model.add_exactly_one([modes_dict[t][m] for m in modes])
        self.variables["modes"] = modes_dict

    def create_resource_consumption_variables(self):
        modes_var = self.variables["modes"]
        resource_consumption_dict = {}
        for t in self.problem.tasks_list:
            resource_consumption_dict[t] = {}
            modes = list(self.problem.mode_details[t].keys())
            nb_modes = len(self.problem.mode_details[t])
            if nb_modes == 1:
                for r in self.problem.resources_list:
                    cons = self.problem.mode_details[t][modes[0]].get(r, 0)
                    if cons > 0:
                        resource_consumption_dict[t][r] = cons
            else:
                potential_resources = set(
                    [
                        r
                        for r in self.problem.resources_list
                        if any(
                            self.problem.mode_details[t][m].get(r, 0) > 0 for m in modes
                        )
                    ]
                )
                for r in potential_resources:
                    values = [self.problem.mode_details[t][m].get(r, 0) for m in modes]
                    resource_consumption_dict[t][r] = self.cp_model.NewIntVar(
                        lb=min(values),
                        ub=max(values),
                        name=f"resource_consumption_{t}_{r}",
                    )
                for m in modes_var[t]:
                    for r in potential_resources:
                        cons = self.problem.mode_details[t][m].get(r, 0)
                        self.cp_model.add(
                            resource_consumption_dict[t][r] == cons
                        ).only_enforce_if(modes_var[t][m])
        self.variables["resource_consumption"] = resource_consumption_dict

    def create_preempt_variables(self) -> None:
        starts = {}
        durations = {}
        ends = {}
        presences = {}
        intervals = {}
        opt_intervals = {}
        for t in self.problem.tasks_list:
            possible_durations = [
                self.problem.get_task_mode_duration(t, m)
                for m in self.problem.get_task_modes(t)
            ]
            max_duration = max(possible_durations)
            if not self.problem.preemptive_indicator[t]:
                starts[t] = [
                    self.cp_model.NewIntVar(
                        lb=0, ub=self.problem.horizon, name=f"start_{t}_part_{0}"
                    )
                ]
                ends[t] = [
                    self.cp_model.NewIntVar(
                        lb=0, ub=self.problem.horizon, name=f"end_{t}_part_{0}"
                    )
                ]
                durations[t] = [
                    self.cp_model.NewIntVar(
                        lb=min(possible_durations),
                        ub=max_duration,
                        name=f"duration_{t}_part_{0}",
                    )
                ]
                presences[t] = [1]
                intervals[t] = [
                    self.cp_model.NewOptionalIntervalVar(
                        start=starts[t][0],
                        end=ends[t][0],
                        size=durations[t][0],
                        is_present=presences[t][0],
                        name=f"interval_{t}_part_{0}",
                    )
                ]
            else:
                starts[t] = [
                    self.cp_model.NewIntVar(
                        lb=0, ub=self.problem.horizon, name=f"start_{t}_part_{i}"
                    )
                    for i in range(max(1, max_duration))
                ]
                ends[t] = [
                    self.cp_model.NewIntVar(
                        lb=0, ub=self.problem.horizon, name=f"end_{t}_part_{i}"
                    )
                    for i in range(max(1, max_duration))
                ]
                durations[t] = [
                    self.cp_model.NewIntVar(lb=0, ub=1, name=f"duration_{t}_part_{i}")
                    for i in range(max(1, max_duration))
                ]
                presences[t] = [
                    self.cp_model.NewBoolVar(name=f"presence_{t}_{i}")
                    for i in range(max(1, max_duration))
                ]
                intervals[t] = [
                    self.cp_model.NewOptionalIntervalVar(
                        start=starts[t][i],
                        end=ends[t][i],
                        size=durations[t][i],
                        is_present=presences[t][i],
                        name=f"interval_{t}_part_{i}",
                    )
                    for i in range(max(1, max_duration))
                ]
            for m in self.problem.get_task_modes(t):
                duration = self.problem.get_task_mode_duration(t, m)
                if not self.problem.preemptive_indicator[t]:
                    opt_intervals[(t, m)] = [
                        self.cp_model.NewOptionalIntervalVar(
                            start=starts[t][0],
                            size=duration,
                            end=ends[t][0],
                            is_present=self.variables["modes"][t][m],
                            name=f"interval_{t, m}_{0}",
                        )
                    ]
                else:
                    opt_intervals[(t, m)] = [
                        self.cp_model.NewOptionalIntervalVar(
                            start=starts[t][i],
                            size=durations[t][i],
                            end=ends[t][i],
                            is_present=self.variables["modes"][t][m],
                            name=f"interval_{t, m}_{i}",
                        )
                        for i in range(duration)
                    ]
        self.variables["starts"] = starts
        self.variables["durations"] = durations
        self.variables["ends"] = ends
        self.variables["presences"] = presences
        self.variables["intervals"] = intervals
        self.variables["opt_intervals"] = opt_intervals

    def constraint_convention_variables(self):
        for t in self.variables["starts"]:
            nb_parts = len(self.variables["presences"][t])
            modes = list(self.problem.mode_details[t].keys())
            potential_durations = list(
                set([self.problem.mode_details[t][m]["duration"] for m in modes])
            )
            # Only force first presence to be 1 if task has non-zero duration
            if min(potential_durations) > 0:
                self.cp_model.add(self.variables["presences"][t][0] == 1)
                self.cp_model.add(self.variables["durations"][t][0] >= 1)
            else:
                # For duration-0 tasks, ensure start == end (since optional interval
                # with is_present=False doesn't enforce end = start + size)
                self.cp_model.add(
                    self.variables["ends"][t][0] == self.variables["starts"][t][0]
                )
            for i in range(nb_parts - 1):
                # Ordered intervals and present until some point, then all absent.
                self.cp_model.add(
                    self.variables["presences"][t][i]
                    >= self.variables["presences"][t][i + 1]
                )
                self.cp_model.add(
                    self.variables["ends"][t][i] <= self.variables["starts"][t][i + 1]
                )
                self.cp_model.add(
                    self.variables["ends"][t][i] <= self.variables["ends"][t][i + 1]
                )
                self.cp_model.add(
                    self.variables["starts"][t][i + 1] == self.variables["ends"][t][i]
                ).only_enforce_if(self.variables["presences"][t][i + 1].Not())
                self.cp_model.add(
                    self.variables["durations"][t][i + 1] == 0
                ).only_enforce_if(self.variables["presences"][t][i + 1].Not())
                self.cp_model.add(
                    self.variables["presences"][t][i + 1] == 0
                ).only_enforce_if(self.variables["presences"][t][i].Not())
            for m in self.problem.get_task_modes(t):
                mode_selected = self.variables["modes"][t][m]
                duration_mode = self.problem.get_task_mode_duration(t, m)
                for i in range(duration_mode):
                    if self.problem.preemptive_indicator[t]:
                        self.cp_model.add(
                            self.variables["durations"][t][i] == 1
                        ).only_enforce_if(mode_selected)
                        self.cp_model.add(
                            self.variables["presences"][t][i] == 1
                        ).only_enforce_if(mode_selected)
                for j in range(duration_mode, len(self.variables["starts"][t])):
                    self.cp_model.add(
                        self.variables["presences"][t][j] == 0
                    ).only_enforce_if(mode_selected)
                    self.cp_model.add(
                        self.variables["durations"][t][j] == 0
                    ).only_enforce_if(mode_selected)

    def constraint_precedence(self):
        for t in self.problem.successors:
            for succ in self.problem.successors[t]:
                self.cp_model.add(
                    self.variables["starts"][succ][0] >= self.variables["ends"][t][-1]
                )

    def constraint_resource(self):
        fake_tasks = create_fake_tasks(self.problem)
        for r in self.problem.resources:
            if r not in self.problem.non_renewable_resources:
                self.constraint_resource_cumulative(resource=r, fake_tasks=fake_tasks)
            else:
                self.constraint_resource_non_renewable(resource=r)

    def constraint_resource_cumulative(
        self, resource: str, fake_tasks: list[dict[str, int]]
    ):
        potential_tasks = [
            t
            for t in self.variables["resource_consumption"]
            if resource in self.variables["resource_consumption"][t]
        ]
        intervals = [
            (
                self.variables["intervals"][t][i],
                self.variables["resource_consumption"][t][resource],
            )
            for t in potential_tasks
            for i in range(len(self.variables["intervals"][t]))
        ]
        fake_tasks_of_interest = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name=f"res_"
                ),
                f.get(resource, 0),
            )
            for f in fake_tasks
            if f.get(resource, 0) > 0
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.add_cumulative(
            [x[0] for x in intervals + fake_tasks_of_interest],
            [x[1] for x in intervals + fake_tasks_of_interest],
            capa,
        )

    def constraint_resource_non_renewable(self, resource: str):
        potential_tasks = [
            t
            for t in self.variables["resource_consumption"]
            if resource in self.variables["resource_consumption"][t]
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.add(
            sum(
                [
                    self.variables["resource_consumption"][t][resource]
                    for t in potential_tasks
                ]
            )
            <= capa
        )

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> PreemptiveRcpspSolution:
        modes_dict = {}
        schedule = {}
        for t in self.variables["starts"]:
            sched = []
            for i in range(len(self.variables["starts"][t])):
                present = cpsolvercb.value(self.variables["presences"][t][i])
                if present:
                    sched.append(
                        (
                            cpsolvercb.value(self.variables["starts"][t][i]),
                            cpsolvercb.value(self.variables["ends"][t][i]),
                        )
                    )
                else:
                    break

            # For duration-0 tasks, record the time point even though no parts are present
            if len(sched) == 0:
                time_point = cpsolvercb.value(self.variables["starts"][t][0])
                schedule[t] = {
                    "starts": [time_point],
                    "ends": [time_point],
                }
            else:
                schedule[t] = {
                    "starts": [x[0] for x in sched],
                    "ends": [x[1] for x in sched],
                }

            modes = list(self.variables["modes"][t].keys())
            if len(modes) == 1:
                modes_dict[t] = modes[0]
            else:
                for m in self.variables["modes"][t]:
                    if cpsolvercb.value(self.variables["modes"][t][m]):
                        modes_dict[t] = m

        # Merge consecutive unit-time intervals
        schedule = merge_consecutive_unit_intervals(schedule)

        modes = [
            modes_dict[t]
            for t in self.problem.tasks_list
            if t not in {self.problem.source_task, self.problem.sink_task}
        ]
        return PreemptiveRcpspSolution(
            problem=self.problem, rcpsp_schedule=schedule, rcpsp_modes=modes
        )

    def set_warm_start(self, solution: PreemptiveRcpspSolution) -> None:
        """Make the solver warm start from the given solution."""
        self.cp_model.clear_hints()
        for task in self.variables["starts"]:
            starts = solution.rcpsp_schedule[task]["starts"]
            ends = solution.rcpsp_schedule[task]["ends"]
            # Decompose each preemptive interval into unit-time parts
            unit_idx = 0
            for st, end in zip(starts, ends):
                for t in range(st, end):
                    if unit_idx < len(self.variables["starts"][task]):
                        self.cp_model.AddHint(
                            self.variables["starts"][task][unit_idx], t
                        )
                        self.cp_model.AddHint(
                            self.variables["ends"][task][unit_idx], t + 1
                        )
                        self.cp_model.AddHint(
                            self.variables["durations"][task][unit_idx], 1
                        )
                        self.cp_model.AddHint(
                            self.variables["presences"][task][unit_idx], 1
                        )
                        unit_idx += 1
            # Set remaining unit-time parts as not present
            for i in range(unit_idx, len(self.variables["starts"][task])):
                self.cp_model.AddHint(self.variables["presences"][task][i], 0)
        # Set mode hints
        for task, mode in zip(self.problem.tasks_list_non_dummy, solution.rcpsp_modes):
            if len(self.variables["modes"][task]) > 1:
                for m in self.variables["modes"][task]:
                    self.cp_model.AddHint(
                        self.variables["modes"][task][m], 1 if m == mode else 0
                    )


class CpSatCalendarPreemptiveSolver(
    CalendarPreemptiveCpSatSolver[Task, CumulativeResource, OtherCalendarResource],
    WarmstartMixin,
):
    hyperparameters = [
        EnumHyperparameter(
            name="modeling_calendar_preemptive",
            enum=ModelingPreemptive,
            default=ModelingPreemptive.INDICATOR,
        )
    ]

    def get_optional_duration_of_task(self, task: Task, mode: int) -> LinearExpr:
        return self.variables["opt_durations"][task][mode]

    problem: PreemptiveRcpspProblem

    def __init__(
        self,
        problem: PreemptiveRcpspProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        # Use generic method from GenericSchedulingProblem
        calendar_data = self.problem.compute_task_durations_with_calendar_preemption()
        self.durations = calendar_data.durations

    def implements_lexico_api(self) -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.variables["objectives"].keys())

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        return [self.cp_model.add(self.variables["objectives"][obj] <= value)]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        sol = res[-1][0]
        kpis = self.problem.evaluate(sol)
        return kpis[obj]

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.minimize(self.variables["objectives"][obj])

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.create_main_variables(
            create_opt_duration=kwargs["modeling_calendar_preemptive"]
            == ModelingPreemptive.ELEMENT
        )
        self.create_duration_constraints(
            modeling=kwargs["modeling_calendar_preemptive"]
        )
        self.constraint_resource()
        self.constraint_precedence()
        self.variables["objectives"] = {}
        obj_list = []
        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if obj == "nb_preempted_tasks" and weight != 0:
                self.compute_nb_preempted_tasks()
                self.variables["objectives"][obj] = self._nb_preempted_tasks
                obj_list.append(weight * self.variables["objectives"][obj])
            if obj == "makespan" and weight != 0:
                self.variables["objectives"][obj] = self.variables["ends"][
                    self.problem.sink_task
                ]
                obj_list.append(weight * self.variables["objectives"][obj])
        if self.params_objective_function.sense_function == ModeOptim.MINIMIZATION:
            self.cp_model.minimize(sum(obj_list))
        if self.params_objective_function.sense_function == ModeOptim.MAXIMIZATION:
            self.cp_model.maximize(sum(obj_list))

    def get_task_mode_interval(self, task: Task, mode: int) -> IntervalVar:
        return self.variables["opt_intervals"][task][mode]

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        if start_or_end == StartOrEnd.START:
            return self.variables["starts"][task]
        if start_or_end == StartOrEnd.END:
            return self.variables["ends"][task]

    def get_task_duration_variable(self, task: Task):
        return self.variables["durations"][task]

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExprT:
        return self.variables["presences"][task][mode]

    def create_main_variables(self, create_opt_duration: bool = False):
        starts = {}
        ends = {}
        durations = {}
        intervals = {}
        opt_intervals = {}
        opt_durations = {}
        presences = {}
        for t in self.problem.tasks_list:
            starts[t] = self.cp_model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"start_{t}"
            )
            ends[t] = self.cp_model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"end_{t}"
            )
            positive_durations = self.problem.get_possible_durations_for_task(t)
            if len(positive_durations) > 1:
                durations[t] = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromValues(positive_durations), name=f"duration_{t}"
                )
            else:
                durations[t] = positive_durations[0]
            intervals[t] = self.cp_model.NewIntervalVar(
                start=starts[t], end=ends[t], size=durations[t], name=f"interval_{t}"
            )
            modes = list(self.problem.mode_details[t].keys())
            opt_intervals[t] = {}
            opt_durations[t] = {}
            presences[t] = {}
            if len(modes) == 1:
                opt_intervals[t][modes[0]] = intervals[t]
                opt_durations[t][modes[0]] = durations[t]
                presences[t][modes[0]] = 1
            else:
                for m in modes:
                    presences[t][m] = self.cp_model.NewBoolVar(name=f"presence_{t}_{m}")
                    if create_opt_duration:
                        poss_dur = self.problem.get_possible_durations_for_task_mode(
                            task=t, mode=m
                        )
                        if len(poss_dur) > 1:
                            opt_durations[t][m] = self.cp_model.NewIntVarFromDomain(
                                domain=Domain.FromValues(poss_dur),
                                name=f"duration_{t}_{m}",
                            )
                        else:
                            opt_durations[t][m] = poss_dur[0]
                    else:
                        opt_durations[t][m] = durations[t]
                    opt_intervals[t][m] = self.cp_model.NewOptionalIntervalVar(
                        start=starts[t],
                        end=ends[t],
                        size=opt_durations[t][m],
                        is_present=presences[t][m],
                        name=f"opt_interval_{t}_{m}",
                    )
                self.cp_model.add_exactly_one([presences[t][m] for m in presences[t]])
        # Need to bind the duration of non preemptive task to the expected value.
        for t in self.problem.tasks_list:
            if not self.problem.is_task_calendar_preempted(t):
                for m in self.problem.get_task_modes(t):
                    dur = self.problem.get_task_mode_duration(t, m)
                    self.cp_model.add(durations[t] == dur).only_enforce_if(
                        presences[t][m]
                    )
        self.variables["starts"] = starts
        self.variables["ends"] = ends
        self.variables["durations"] = durations
        self.variables["intervals"] = intervals
        self.variables["opt_intervals"] = opt_intervals
        self.variables["opt_durations"] = opt_durations
        self.variables["presences"] = presences

    def constraint_duration_of_tasks(self):
        """
        Tricky constraint : should take into account the partial preemption possibility,
        which makes duration variable based on calendars
        """
        durs = self.durations
        dictionary_indicators = {}
        for task_index, mode in durs:
            d = self.constraint_duration_of_task(
                task_index=task_index,
                mode=mode,
                duration_per_interval=durs[(task_index, mode)][1],
            )
            dictionary_indicators.update(d)
        self.variables["dictionary_indicators"] = dictionary_indicators
        for index in self.variables["presences"]:
            all_key = [
                x for x in self.variables["dictionary_indicators"] if x[0][0] == index
            ]
            self.cp_model.AddExactlyOne(
                [self.variables["dictionary_indicators"][x] for x in all_key]
            )

    def constraint_duration_of_task(
        self,
        task_index: int,
        mode: int,
        duration_per_interval: dict[int, list[tuple[int, int]]],
    ):
        dictionary_indicators = {}
        positive_durations = [d for d in duration_per_interval if d >= 0]
        if len(positive_durations) == 1:
            dur = int(positive_durations[0])
            interval = Domain.FromIntervals(duration_per_interval[dur])
            self.cp_model.AddLinearExpressionInDomain(
                self.variables["starts"][task_index], interval
            ).only_enforce_if(self.variables["presences"][task_index][mode])
            (
                self.cp_model.Add(
                    self.variables["durations"][task_index] == dur
                ).only_enforce_if(self.variables["presences"][task_index][mode])
            )
            dictionary_indicators[((task_index, mode), dur)] = self.variables[
                "presences"
            ][task_index][mode]
        else:
            for possible_duration in duration_per_interval:
                if possible_duration < 0:
                    continue
                interval = Domain.FromIntervals(
                    duration_per_interval[possible_duration]
                )
                dictionary_indicators[((task_index, mode), possible_duration)] = (
                    self.cp_model.NewBoolVar(
                        f"d_{(task_index, mode), possible_duration}"
                    )
                )
                self.cp_model.AddLinearExpressionInDomain(
                    self.variables["starts"][task_index], interval
                ).OnlyEnforceIf(
                    dictionary_indicators[((task_index, mode), possible_duration)]
                )
                self.cp_model.Add(
                    self.variables["durations"][task_index] == int(possible_duration)
                ).OnlyEnforceIf(
                    dictionary_indicators[((task_index, mode), possible_duration)]
                )
            # corrected version (to be confirmed)
            self.cp_model.Add(
                sum([dictionary_indicators[k] for k in dictionary_indicators])
                == self.variables["presences"][task_index][mode]
            )
        return dictionary_indicators

    def constraint_precedence(self):
        for t in self.problem.successors:
            for succ in self.problem.successors[t]:
                self.cp_model.add(
                    self.variables["starts"][succ] >= self.variables["ends"][t]
                )

    def constraint_resource(self):
        fake_tasks = create_fake_tasks(self.problem)
        for r in self.problem.resources:
            if r not in self.problem.non_renewable_resources:
                self.constraint_resource_cumulative(resource=r, fake_tasks=fake_tasks)
            else:
                self.constraint_resource_non_renewable(resource=r)

    def constraint_resource_cumulative(
        self, resource: str, fake_tasks: list[dict[str, int]]
    ):
        max_capacity = self.problem.get_max_resource_capacity(resource)
        potential_tasks = [
            (t, i, self.problem.mode_details[t][i].get(resource, 0))
            for t in self.variables["opt_intervals"]
            for i in self.variables["opt_intervals"][t]
            if self.problem.mode_details[t][i].get(resource, 0) > 0
        ]

        # First: add a base cumulative constraint with ALL tasks and NO fake tasks
        # This ensures resource limits are respected among real tasks
        task_pulse_all = [
            (self.variables["opt_intervals"][t][m], q) for t, m, q in potential_tasks
        ]
        if len(task_pulse_all) > 0:
            self.cp_model.add_cumulative(
                [x[0] for x in task_pulse_all],
                [x[1] for x in task_pulse_all],
                max_capacity,
            )

        # Second: add cumulative constraints with fake tasks for partial capacity reductions
        # (excluding full gaps where capacity reduction equals max_capacity)
        different_calendar_values = set(
            [f.get(resource, 0) for f in fake_tasks if f.get(resource, 0) > 0]
        )
        for diff_value in different_calendar_values:
            # Skip full capacity reductions - tasks can span these with adjusted duration
            if diff_value >= max_capacity:
                task_pulse = [
                    (self.variables["opt_intervals"][t][m], q)
                    for t, m, q in potential_tasks
                    if not self.problem.is_task_calendar_preempted(t)
                ]
            else:
                task_pulse = [
                    (self.variables["opt_intervals"][t][m], q)
                    for t, m, q in potential_tasks
                    if q + diff_value <= max_capacity
                    or not self.problem.is_task_calendar_preempted(t)
                ]
            calendar_pulse = [
                (
                    self.cp_model.new_fixed_size_interval_var(
                        start=f["start"], size=f["duration"], name=f"dummy_{resource}"
                    ),
                    f.get(resource, 0),
                )
                for f in fake_tasks
                if 0 < f.get(resource, 0) <= diff_value
            ]
            if len(task_pulse) == 0:
                continue
            self.cp_model.add_cumulative(
                [x[0] for x in task_pulse + calendar_pulse],
                [x[1] for x in task_pulse + calendar_pulse],
                max_capacity,
            )

    def constraint_resource_non_renewable(self, resource: str):
        potential_tasks = [
            (t, m, self.problem.mode_details[t][m].get(resource, 0))
            for t in self.variables["opt_intervals"]
            for m in self.variables["opt_intervals"][t]
            if self.problem.mode_details[t][m].get(resource, 0) > 0
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.add(
            sum([q * self.variables["presences"][t][m] for t, m, q in potential_tasks])
            <= capa
        )

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        for obj in self.variables["objectives"]:
            logger.info(
                f"Obj. {obj}: {cpsolvercb.value(self.variables['objectives'][obj])}"
            )
        schedule = {}
        modes_dict = {}
        for t in self.variables["starts"]:
            st = cpsolvercb.value(self.variables["starts"][t])
            end = cpsolvercb.value(self.variables["ends"][t])
            schedule[t] = {"starts": [st], "ends": [end]}
            for m in self.variables["presences"][t]:
                if cpsolvercb.value(self.variables["presences"][t][m]) > 0:
                    modes_dict[t] = m
        modes = [modes_dict[t] for t in self.problem.tasks_list_non_dummy]

        # Create calendar solution
        calendar_solution = PreemptiveRcpspSolution(
            problem=self.problem, rcpsp_schedule=schedule, rcpsp_modes=modes
        )

        # Transform to actual preemptive solution (split by calendar gaps)
        preemptive_solution = transform_calendar_preemptive_solution_to_preemptive(
            solution=calendar_solution,
            problem=self.problem,
        )

        return preemptive_solution

    def set_warm_start(self, solution: PreemptiveRcpspSolution) -> None:
        """Make the solver warm start from the given solution.

        The calendar solver models tasks as continuous intervals, so we use
        the min start and max end from the preemptive solution.
        """
        self.cp_model.clear_hints()
        for task in self.variables["starts"]:
            starts = solution.rcpsp_schedule[task]["starts"]
            ends = solution.rcpsp_schedule[task]["ends"]
            if len(starts) > 0:
                # Calendar solver uses single continuous interval
                calendar_start = min(starts)
                calendar_end = max(ends)
                self.cp_model.AddHint(self.variables["starts"][task], calendar_start)
                self.cp_model.AddHint(self.variables["ends"][task], calendar_end)
                self.cp_model.AddHint(
                    self.variables["durations"][task], calendar_end - calendar_start
                )
        # Set mode hints
        for task, mode in zip(self.problem.tasks_list_non_dummy, solution.rcpsp_modes):
            if len(self.variables["presences"][task]) > 1:
                for m in self.variables["presences"][task]:
                    self.cp_model.AddHint(
                        self.variables["presences"][task][m], 1 if m == mode else 0
                    )


def transform_calendar_preemptive_solution_to_preemptive(
    solution: PreemptiveRcpspSolution,
    problem: PreemptiveRcpspProblem,
    resource_calendar_dict: dict[tuple, np.ndarray] = None,
    task_mode_to_calendar: dict[tuple, np.ndarray] = None,
) -> PreemptiveRcpspSolution:
    if resource_calendar_dict is None:
        # Use generic method from GenericSchedulingProblem
        calendar_data = problem.compute_task_durations_with_calendar_preemption()
        resource_calendar_dict = calendar_data.resource_calendar_dict
        task_mode_to_calendar = calendar_data.task_mode_to_calendar
    sched = {}
    for t in solution.rcpsp_schedule:
        mode = 1
        if t in problem.tasks_list_non_dummy:
            mode = solution.rcpsp_modes[problem.index_task_non_dummy[t]]
        if (t, mode) not in task_mode_to_calendar:
            sched[t] = solution.rcpsp_schedule[t]
            continue
        calendar = resource_calendar_dict[task_mode_to_calendar[(t, mode)]]
        sts = solution.get_start_times_list(t)
        ends = solution.get_end_times_list(t)
        sched_list = []
        for st, end in zip(sts, ends):
            sched_list += [j for j in range(st, end) if calendar[j]]
        subparts = []
        cur_start = sched_list[0]
        for index in range(1, len(sched_list)):
            if sched_list[index] > sched_list[index - 1] + 1:
                subparts.append((cur_start, sched_list[index - 1] + 1))
                cur_start = sched_list[index]
        subparts.append((cur_start, sched_list[-1] + 1))
        sched[t] = {}
        sched[t]["starts"] = [x[0] for x in subparts]
        sched[t]["ends"] = [x[1] for x in subparts]
    return PreemptiveRcpspSolution(
        problem=problem, rcpsp_schedule=sched, rcpsp_modes=solution.rcpsp_modes
    )


def merge_consecutive_unit_intervals(
    schedule: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, list[int]]]:
    """Merge consecutive unit-time intervals into larger intervals.

    For example, transforms [0, 1], [1, 2], [2, 3] into [0, 3].
    This is useful for post-processing solutions from unit-time decomposition models.

    Args:
        schedule: dict mapping task -> {"starts": [...], "ends": [...]}

    Returns:
        dict with same structure but with consecutive intervals merged
    """
    merged_schedule = {}
    for task, sched in schedule.items():
        starts = sched["starts"]
        ends = sched["ends"]

        if len(starts) == 0:
            merged_schedule[task] = {"starts": [], "ends": []}
            continue

        merged_starts = [starts[0]]
        merged_ends = []

        for i in range(1, len(starts)):
            # Check if current interval is consecutive to previous
            if starts[i] == ends[i - 1]:
                # Consecutive, don't add new start
                pass
            else:
                # Gap found, close previous interval and start new one
                merged_ends.append(ends[i - 1])
                merged_starts.append(starts[i])

        # Close last interval
        merged_ends.append(ends[-1])

        merged_schedule[task] = {"starts": merged_starts, "ends": merged_ends}

    return merged_schedule
