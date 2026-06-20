#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Module containing utils for problem having calendar based preemption:
This means the tasks can overlap over a period where the resource are not met for this task:
the duration of tasks can be pre-computed based on the start time.
The hypothesis is that the task will be executed whenever there is enough cumulative resource
available at a given time.
"""

from abc import abstractmethod
from typing import Generic

from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    OtherCalendarResource,
    Skill,
    SkillProblem,
    Task,
    UnaryResource,
)


class PreemptiveSchedulingProblem(
    SkillProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ],
):
    @abstractmethod
    def is_only_calendar_preemptive(self, task: Task) -> bool: ...
    @abstractmethod
    def is_anytime_preemptive(self, task: Task) -> bool: ...
    def is_calendar_preemptive(self, task: Task) -> bool:
        return self.is_anytime_preemptive(task) or self.is_only_calendar_preemptive(
            task
        )

    def is_preemptive(self, task: Task) -> bool:
        return self.is_anytime_preemptive(task) or self.is_calendar_preemptive(task)

    # @abstractmethod
    # def get_nb_max_preemption(self, task: Task, mode: int) -> int:
    #     ...
    # def get_default_nb_max_preemption(self, task: Task, mode: int) -> int:
    #     if self.is_preemptive(task):
    #         dur = self.get_task_mode_duration(task, mode)
    #         return dur
    #     else:
    #         return 1
    # @abstractmethod
    # def get_min_dur_subpart(self, task: Task, mode: int) -> int:
    #     ...
    # def get_default_min_dur_subpart(self, task: Task, mode: int) -> int:
    #     if self.is_preemptive(task):
    #         return 1
    #     else:
    #         return self.get_task_mode_duration(task, mode)
