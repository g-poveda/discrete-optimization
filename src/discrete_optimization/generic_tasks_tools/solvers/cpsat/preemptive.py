#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.calendar_preemptive import (
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.preemptive import (
    NonSkillCumulativeResource,
    OtherCalendarResource,
    Skill,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.generic_scheduling import (
    GenericSchedulingCpSatSolver,
)


class PreemptiveVariable(Enum):
    INDICATOR = 0
    ELEMENT = 1


class PreemptiveCpSatSolver(
    GenericSchedulingCpSatSolver[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ],
):
    use_cpm_for_task_bounds: bool = False
    tasks_bounds: dict[Task, tuple[int, int, int, int]]
    problem: GenericSchedulingProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]

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
            if not self.problem.is_preemptive(t):
                starts[t] = [
                    self.cp_model.NewIntVar(
                        lb=self.get_task_start_or_end_lower_bound(t, StartOrEnd.START),
                        ub=self.get_task_start_or_end_upper_bound(t, StartOrEnd.END),
                        name=f"start_{t}_part_{0}",
                    )
                ]
                ends[t] = [
                    self.cp_model.NewIntVar(
                        lb=self.get_task_start_or_end_lower_bound(t, StartOrEnd.END),
                        ub=self.get_task_start_or_end_upper_bound(t, StartOrEnd.END),
                        name=f"end_{t}_part_{0}",
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
