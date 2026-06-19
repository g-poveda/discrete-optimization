#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, Optional, Union

import numpy as np
import wrapt

from discrete_optimization.generic_tasks_tools.allocation import (
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    Objective,
    Penalty,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
    NonRenewableResourceProblem,
    NonRenewableResourceSolution,
)
from discrete_optimization.generic_tasks_tools.precedence_scheduling import (
    PrecedenceSchedulingProblem,
    PrecedenceSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    Skill,
    SkillProblem,
    SkillSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cpm import Cpm
from discrete_optimization.generic_tasks_tools.timelag import (
    TimelagProblem,
    TimelagSolution,
    consolidate_min_time_lags,
)
from discrete_optimization.generic_tasks_tools.timewindow import (
    TimewindowProblem,
    TimewindowSolution,
)

CumulativeResource = Skill | NonSkillCumulativeResource
Resource = CumulativeResource | UnaryResource
AnyResource = NonRenewableResource | Resource

# Type aliases for calendar preemption
CalendarKey = Union[
    tuple[Hashable, int],  # Single resource: (resource_id, consumption)
    tuple[
        tuple[Hashable, int], ...
    ],  # Multiple resources: ((res1, cons1), (res2, cons2), ...)
]


@dataclass
class CalendarPreemptionData:
    """Data structure containing calendar-aware preemption information for scheduling problems.

    This class encapsulates the precomputed information needed to handle task preemption
    when resources have varying availability over time (e.g., workers unavailable on weekends).

    For each (task, mode) combination, it provides:
    - How the actual task duration varies based on start time
    - Which time intervals are valid for each possible duration
    - Binary calendars indicating when resources are available

    Attributes:
        durations: Maps (task, mode) to preemption duration information.
            For each (task, mode):
            - duration_array[t]: actual time units needed if task starts at time t
            - interval_dict[d]: list of [start, end) intervals where starting the task
                               results in duration d
        resource_calendar_dict: Maps resource consumption patterns to binary availability calendars.
            Each binary calendar is a numpy array where 1 = resources available, 0 = unavailable.
            The key is a CalendarKey identifying a unique resource consumption pattern.
        task_mode_to_calendar: Maps each (task, mode) to its resource consumption pattern key.
            Used to look up the corresponding binary calendar in resource_calendar_dict.

    Example:
       # >>> data = problem.compute_task_durations_with_calendar_preemption()
       # >>> # Get duration info for task 5, mode 1
       # >>> duration_array, interval_dict = data.durations[(task_5, 1)]
       # >>> # If starting at time 10, task will take duration_array[10] time units
       # >>> # Get the binary calendar for this task/mode
       # >>> calendar_key = data.task_mode_to_calendar[(task_5, 1)]
       # >>> binary_calendar = data.resource_calendar_dict[calendar_key]
       # >>> # binary_calendar[t] == 1 means resources available at time t

    """

    durations: dict[tuple[Task, int], tuple[np.ndarray, dict[int, list[list[int]]]]]
    resource_calendar_dict: dict[CalendarKey, np.ndarray]
    task_mode_to_calendar: dict[tuple[Task, int], CalendarKey]

    def get_duration_for_start_time(
        self, task: Task, mode: int, start_time: int
    ) -> int:
        """Get the actual duration if the task starts at the given time.

        Args:
            task: The task
            mode: The mode
            start_time: The proposed start time

        Returns:
            Actual duration in time units (may be larger than nominal duration due to preemption)

        """
        duration_array, _ = self.durations[(task, mode)]
        return int(duration_array[start_time])

    def get_binary_calendar(self, task: Task, mode: int) -> np.ndarray:
        """Get the binary availability calendar for a task/mode.

        Args:
            task: The task
            mode: The mode

        Returns:
            Binary numpy array where 1 = resources available, 0 = unavailable

        """
        calendar_key = self.task_mode_to_calendar[(task, mode)]
        return self.resource_calendar_dict[calendar_key]

    def to_index_based(
        self, problem: "GenericSchedulingProblem"
    ) -> tuple[
        dict[tuple[int, int], tuple[np.ndarray, dict[int, list[list[int]]]]],
        dict[tuple[int, int], np.ndarray],
        dict[tuple[int, int], CalendarKey],
    ]:
        """Convert task-based keys to index-based keys for solver compatibility.

        Some solvers (e.g., FlexProblem solvers) expect integer task indices rather
        than task objects. This method remaps all data to use indices.

        Args:
            problem: The scheduling problem providing the task-to-index mapping

        Returns:
            tuple of:
            - durations_by_index: dict[(task_index, mode)] -> (duration_array, interval_dict)
            - res_arrays: dict[(task_index, mode)] -> binary calendar array
            - task_mode_to_calendar_by_index: dict[(task_index, mode)] -> CalendarKey

        Example:
            >>> data = problem.compute_task_durations_with_calendar_preemption()
            >>> durations, res_arrays, task_mode_mapping = data.to_index_based(problem)

        """
        durations_by_index = {}
        res_arrays = {}
        task_mode_to_calendar_by_index = {}

        for task in problem.tasks_list:
            task_index = problem.get_index_from_task(task)
            for mode in problem.get_task_modes(task):
                if (task, mode) in self.durations:
                    durations_by_index[(task_index, mode)] = self.durations[
                        (task, mode)
                    ]
                if (task, mode) in self.task_mode_to_calendar:
                    calendar_key = self.task_mode_to_calendar[(task, mode)]
                    res_arrays[(task_index, mode)] = self.resource_calendar_dict[
                        calendar_key
                    ]
                    task_mode_to_calendar_by_index[(task_index, mode)] = calendar_key

        return durations_by_index, res_arrays, task_mode_to_calendar_by_index


# Calendar-aware preemption utilities


def compute_binary_calendar_for_resource_consumption(
    resource_availabilities: dict[CumulativeResource, np.ndarray],
    resource_consumption: dict[CumulativeResource, int],
) -> np.ndarray:
    """Compute binary calendar showing when resources are available for given consumption.

    Given resource availability calendars and a consumption requirement, this computes
    a binary array indicating at each time step whether all required resources are
    available in sufficient quantity.

    Args:
        resource_availabilities: dict mapping resource ID to availability array (values = capacity at each time)
        resource_consumption: dict mapping resource ID to required consumption amount

    Returns:
        Binary numpy array where 1 = all resources available with sufficient capacity, 0 = not available

    Example:
        If resource_availabilities = {'R1': [2, 2, 0, 2], 'R2': [1, 1, 1, 1]}
        and resource_consumption = {'R1': 1, 'R2': 1}
        returns [1, 1, 0, 1] (unavailable at t=2 because R1 has 0 capacity)

    """
    resource_ids = [
        r for r in resource_consumption.keys() if resource_consumption[r] > 0
    ]
    if len(resource_ids) == 0:
        # No resource required - always available
        horizon = max(
            (len(avail) for avail in resource_availabilities.values()), default=100
        )
        return np.ones(horizon, dtype=bool)

    # Start with the first resource
    first_resource = resource_ids[0]
    binary_calendar = (
        resource_availabilities[first_resource] >= resource_consumption[first_resource]
    )

    # AND with remaining resources
    for resource_id in resource_ids[1:]:
        binary_calendar &= (
            resource_availabilities[resource_id] >= resource_consumption[resource_id]
        )

    return binary_calendar


def compute_duration_with_calendar_preemption(
    orig_duration: int,
    resource_calendar: np.ndarray,
    cumulative_resource_calendar: np.ndarray,
) -> tuple[np.ndarray, dict[int, list[list[int]]]]:
    """Compute how task duration varies based on start time given resource calendar.

    This is the core algorithm for calendar-aware preemption. For each possible
    start time, it computes the actual duration (in time units) needed to complete
    orig_duration units of work, considering resource availability.

    When a resource has varying availability (e.g., unavailable on weekends), a task
    requiring N work units may take more than N time units to complete. This function
    computes the mapping between start time and actual completion time.

    Args:
        orig_duration: Original work duration (in work units, not time units)
        resource_calendar: Binary array indicating resource availability at each time (1=available, 0=unavailable)
        cumulative_resource_calendar: Cumulative sum of resource_calendar

    Returns:
        tuple of:
        - duration array: duration[i] = actual time units needed if starting at time i (-1 if cannot complete)
        - interval dict: maps each possible duration to list of [start, end] intervals
                        where that duration applies

    Example:
        If resource_calendar = [1, 1, 0, 1, 1] and orig_duration = 3:
        - Starting at t=0: work at t=0,1,3 → duration = 4
        - Starting at t=1: work at t=1,3,4 → duration = 4
        - Starting at t=3: work at t=3,4 → duration = 2 (only 2 units available)

    """
    duration = -np.ones((cumulative_resource_calendar.shape[0]))
    dict_of_interval_per_duration = {}
    current_interval = [0, 0]
    cur_duration = -1

    for i in range(cumulative_resource_calendar.shape[0]):
        if resource_calendar[i] == 0:
            # Resource unavailable at this time
            if duration[i] == duration[i - 1]:
                current_interval[1] = i
            else:
                prev_d = duration[i - 1]
                if prev_d not in dict_of_interval_per_duration:
                    dict_of_interval_per_duration[prev_d] = []
                dict_of_interval_per_duration[prev_d] += [
                    [current_interval[0], current_interval[1]]
                ]
                current_interval = [i, i]
            continue

        x = cumulative_resource_calendar[i]
        if x == 0:
            continue

        # Find when we complete orig_duration work units starting at time i
        index = next(
            (
                j
                for j in range(i, cumulative_resource_calendar.shape[0])
                if cumulative_resource_calendar[j] == x + orig_duration - 1
            ),
            None,
        )

        if index is not None:
            duration[i] = index - i + 1
            cur_duration = duration[i]
            if i >= 1:
                if duration[i] == duration[i - 1]:
                    current_interval[1] = i
                else:
                    prev_d = duration[i - 1]
                    if prev_d not in dict_of_interval_per_duration:
                        dict_of_interval_per_duration[prev_d] = []
                    dict_of_interval_per_duration[prev_d] += [
                        [current_interval[0], current_interval[1]]
                    ]
                    current_interval = [i, i]
        else:
            break

    if current_interval[0] != current_interval[1]:
        d = cur_duration
        if d not in dict_of_interval_per_duration:
            dict_of_interval_per_duration[d] = []
        dict_of_interval_per_duration[d] += [[current_interval[0], current_interval[1]]]

    if len(dict_of_interval_per_duration) == 0:
        dict_of_interval_per_duration[orig_duration] = current_interval

    return duration, dict_of_interval_per_duration


class GenericSchedulingProblem(
    SkillProblem[Task, UnaryResource, Skill, NonSkillCumulativeResource, UnaryResource],
    NonRenewableResourceProblem[Task, NonRenewableResource],
    PrecedenceSchedulingProblem[Task],
    TimelagProblem[Task],
    TimewindowProblem[Task],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
):
    """Scheduling problem with all optional features

    This class derives from other mixins to provide utilities that require that mix:
    - scheduling: tasks need to be scheduled
    - calendar: the renewable resources have their own calendar that will be used for constraining allocations and schedule
    - multimode: the tasks have several mode on which the duration depends
    - cumulative: the tasks consume cumulative resources according to the chosen mode
    - allocation: the tasks can have unary resources allocated to them
    - skill: some cumulative resource are skills that are brought to tasks by allocated unary resources
    - non-renewable: the tasks consume non-renewable resources according to the chosen mode
    - precedence: precedence constraints between tasks
    - cost: the choice of a mode or of an allocation has a given cost

    Even though this class is generic but encompasses also more specific cases:
    - singlemode: actually only one mode per task
    - no skills: if skills_list is empty
    - no allocation: unary_resources is empty
    - no cumulative ressources: if resources_list list only unary resources
    - no calendar: resource capacity can be given as a constant on [0, horizon)
    - no non-renewable ressources: if non_renewable_resources_list empty
    - no precedence constraints: precedence constraints empty
    - no cost: cost = 0

    We suppose that all renewable resources are
    - either cumulative ones
    - or unary resources

    This generic class is to be used to construct generic automatic solvers (e.g. ).

    """

    @property
    def calendar_resources_list(self) -> list[Resource]:
        return self.unary_resources_list + self.cumulative_resources_list

    def check_calendar_resources_list(self) -> None:
        """Check calendar resources list.

        Raises:
            AssertionError: if duplicates appear in the list

        Returns:

        """
        calendar_resources_list = (
            self.unary_resources_list + self.cumulative_resources_list
        )
        assert len(calendar_resources_list) == len(set(calendar_resources_list)), (
            "There are duplicates in calendar resources list, "
            "potentially because unary and cumulative resources intersect."
        )

    def update_resource_availabilities(self) -> None:
        super().update_resource_availabilities()
        self.check_calendar_resources_list()
        self.compute_task_durations_with_calendar_preemption.cache_clear()

    def is_unary_resource(self, resource: Resource) -> bool:
        """Check if given resource is a unary resource."""
        return resource in self.unary_resources_list

    @wrapt.lru_cache(maxsize=None)
    def get_task_start_or_end_tighter_lower_bound(
        self,
        task: Task,
        start_or_end: StartOrEnd,
        use_cpm: bool = False,
        horizon: Optional[int] = None,
    ) -> int:
        """Get a tighter lower bound on task start or end using possible durations.

        Args:
            use_cpm: whether to use CPM propagating bounds through precedence graph
            horizon: new horizon to take into account when computing tighter bounds,
                default to problem horizon.

        """
        if horizon is None:
            horizon = self.get_makespan_upper_bound()
        if use_cpm:
            tasks_bounds = self.compute_tighter_task_bounds(
                use_cpm=True, horizon=horizon
            )
            start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound = (
                tasks_bounds[task]
            )
            match start_or_end:
                case StartOrEnd.START:
                    return start_lower_bound
                case _:
                    return end_lower_bound
        else:
            if start_or_end == StartOrEnd.START:
                return max(  # best bound between:
                    # default bound
                    0,
                    # initial pb bound
                    self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.START
                    ),
                    # bound deduced from initial end lower bound and max duration
                    self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.END
                    )
                    - max(
                        self.get_task_mode_duration(task=task, mode=mode)
                        for mode in self.get_task_modes(task)
                    ),
                )
            else:
                return max(  # best bound between:
                    # default bound
                    0,
                    # initial pb bound
                    self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.END
                    ),
                    # bound deduced from initial start lower bound and min duration
                    max(
                        self.get_task_start_or_end_lower_bound(
                            task=task, start_or_end=StartOrEnd.START
                        ),
                        0,  # clip at 0
                    )
                    + min(
                        self.get_task_mode_duration(task=task, mode=mode)
                        for mode in self.get_task_modes(task)
                    ),
                )

    @wrapt.lru_cache(maxsize=None)
    def get_task_start_or_end_tighter_upper_bound(
        self,
        task: Task,
        start_or_end: StartOrEnd,
        use_cpm: bool = False,
        horizon: Optional[int] = None,
    ) -> int:
        """Get a tighter upper bound on task start or end using possible durations.

        Args:
            use_cpm: whether to use CPM propagating bounds through precedence graph
            horizon: new horizon to take into account when computing tighter bounds,
                default to problem horizon.

        """
        if horizon is None:
            horizon = self.get_makespan_upper_bound()
        if use_cpm:
            tasks_bounds = self.compute_tighter_task_bounds(
                use_cpm=True, horizon=horizon
            )
            start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound = (
                tasks_bounds[task]
            )
            match start_or_end:
                case StartOrEnd.START:
                    return start_upper_bound
                case _:
                    return end_upper_bound
        else:
            # no propagation, only using problem bounds + possible durations + new horizon
            if start_or_end == StartOrEnd.START:
                return min(  # best bound between:
                    # initial pb bound
                    self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.START
                    ),
                    # bound deduced from initial end upper bound (clipped at new horizon) and min duration
                    min(
                        self.get_task_start_or_end_upper_bound(
                            task=task, start_or_end=StartOrEnd.END
                        ),
                        horizon,  # new horizon
                    )
                    - min(  # min duration
                        self.get_task_mode_duration(task=task, mode=mode)
                        for mode in self.get_task_modes(task)
                    ),
                )
            else:
                return min(  # best bound between:
                    # default bound: new horizon
                    horizon,
                    # initial pb bound
                    self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.END
                    ),
                    # bound deduced from initial start upper bound and max duration
                    self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.START
                    )
                    + max(
                        self.get_task_mode_duration(task=task, mode=mode)
                        for mode in self.get_task_modes(task)
                    ),
                )

    def update_task_bounds(self) -> None:
        """Method to be called when problem time windows are updated.

        It clears necessary cache on computed tighter bounds.

        """
        self.get_task_start_or_end_tighter_upper_bound.cache_clear()
        self.get_task_start_or_end_tighter_lower_bound.cache_clear()
        self.compute_tighter_task_bounds.cache_clear()

    @wrapt.lru_cache(maxsize=None)
    def compute_tighter_task_bounds(
        self, use_cpm: bool = False, horizon: Optional[int] = None
    ) -> dict[Task, tuple[int, int, int, int]]:
        """Compute tighter task bounds from problem time windows and min-max tak durations.

        Args:
            use_cpm: whether to use CPM propagating bounds through precedence graph
            horizon: new horizon to take into account when computing tighter bounds,
                default to problem horizon.

        Returns:
            {task: (start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound)}

        """
        if horizon is None:
            horizon = self.get_makespan_upper_bound()
        if use_cpm:
            cpm = Cpm(problem=self, horizon=horizon)
            cpm.compute_task_bounds()
            return cpm.get_task_bounds()
        else:
            return {
                task: (
                    self.get_task_start_or_end_tighter_lower_bound(
                        task=task,
                        start_or_end=StartOrEnd.START,
                        horizon=horizon,
                        use_cpm=False,
                    ),
                    self.get_task_start_or_end_tighter_lower_bound(
                        task=task,
                        start_or_end=StartOrEnd.END,
                        horizon=horizon,
                        use_cpm=False,
                    ),
                    self.get_task_start_or_end_tighter_upper_bound(
                        task=task,
                        start_or_end=StartOrEnd.START,
                        horizon=horizon,
                        use_cpm=False,
                    ),
                    self.get_task_start_or_end_tighter_upper_bound(
                        task=task,
                        start_or_end=StartOrEnd.END,
                        horizon=horizon,
                        use_cpm=False,
                    ),
                )
                for task in self.tasks_list
            }

    def get_consolidated_time_lags(
        self,
        task1_start_or_end: StartOrEnd,
        task2_start_or_end: StartOrEnd,
        min_or_max: MinOrMax,
    ):
        """Get consolidated time lags.

        Same normalization as in `TimelagProblem` parent class.
        Also taking into account precedence constraints to enrich end to start min time lags.

        Args:
            task1_start_or_end:
            task2_start_or_end:
            min_or_max:

        Returns:

        """
        timelags = super().get_consolidated_time_lags(
            task1_start_or_end=task1_start_or_end,
            task2_start_or_end=task2_start_or_end,
            min_or_max=min_or_max,
        )
        if (task1_start_or_end, task2_start_or_end, min_or_max) == (
            StartOrEnd.END,
            StartOrEnd.START,
            MinOrMax.MIN,
        ):
            # end to start min time lags: we add precedence constraints and keep only max(resulting offsets)
            timelags = consolidate_min_time_lags(
                timelags
                + [
                    (task1, task2, 0)
                    for task1, next_tasks in self.get_precedence_constraints().items()
                    for task2 in next_tasks
                ]
            )
        return timelags

    def get_makespan_lower_bound(self) -> int:
        """Get a lower bound on global makespan.

        Computed tighter lower bounds on last tasks can be used to get a better makespan lower bound.

        """
        return max(
            self.get_task_start_or_end_tighter_lower_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            for task in self.get_last_tasks()
        )

    def get_makespan_tighter_lower_bound(
        self, use_cpm: bool = False, horizon: Optional[int] = None
    ) -> int:
        """Get a tighter lower bound on global makespan.

        Args:
            use_cpm: whether to use CPM bound propagation through precedence graph to improve tightness
            horizon: new horizon to take into account when computing tighter bounds,
                default to problem horizon.
                NB: The choice of horizon should not affect the result for the lower bound, but it could avoid
                CPM complete recomputation thanks to caching if it was already launched with same horizon.

        """
        if horizon is None:
            horizon = self.get_makespan_tighter_upper_bound()
        return max(
            self.get_task_start_or_end_tighter_lower_bound(
                task=task, start_or_end=StartOrEnd.END, use_cpm=use_cpm, horizon=horizon
            )
            for task in self.get_last_tasks()
        )

    def get_makespan_tighter_upper_bound(self) -> int:
        """Compute a tighter upper bound on makespan.

        The original makespan upper bound is used when computing tighter bounds for tasks starts and ends,
        via `self.compute_tighter_task_bounds()` or `self.get_task_start_or_end_tighter_upper_bound()`.
        From that tighter bounds, we can derive a new makespan upper bound.

        """
        return max(
            # do not use CPM (not necessary as last tasks horizon are just computed from horizon + time windows even in CPM)
            self.get_task_start_or_end_tighter_upper_bound(
                task=task, start_or_end=StartOrEnd.END, use_cpm=False
            )
            for task in self.get_last_tasks()
        )

    @wrapt.lru_cache(maxsize=None)
    def get_consolidated_precedence_constraints(self) -> dict[Task, set[Task]]:
        """Consolidate precedence constraints defined by problem.

        It takes into account time lags constraints.
        - end to start min constraint with non-negative offsets => precedence constraint
        - start synchronization => corresponding tasks should appear together in successors
        - end synchronization => corresponding tasks should share their successors

        """
        successors = defaultdict(set)

        # end to task min timelag with offset >=0  => precedence constraint
        # (original ones already included in consolidated time lags)
        for task1, task2, offset in self.get_consolidated_time_lags(
            task1_start_or_end=StartOrEnd.END,
            task2_start_or_end=StartOrEnd.START,
            min_or_max=MinOrMax.MIN,
        ):
            if offset >= 0:
                successors[task1].add(task2)

        # end together => same successors
        min_end_to_end_timelags_0_offset = [
            (t1, t2)
            for t1, t2, offset in self.get_consolidated_time_lags(
                task1_start_or_end=StartOrEnd.END,
                task2_start_or_end=StartOrEnd.END,
                min_or_max=MinOrMax.MIN,
            )
            if offset == 0
        ]
        max_end_to_end_timelags_0_offset = [
            (t1, t2) for t2, t1 in min_end_to_end_timelags_0_offset
        ]
        end_together = set(min_end_to_end_timelags_0_offset).intersection(
            max_end_to_end_timelags_0_offset
        )
        for task1, task2 in end_together:
            successors[task1].update(successors[task2])
            # the reverse will be done during the loop as (task2, task1) should also be in end_together

        # start together => same predecessors
        min_start_to_start_timelags_0_offset = [
            (t1, t2)
            for t1, t2, offset in self.get_consolidated_time_lags(
                task1_start_or_end=StartOrEnd.START,
                task2_start_or_end=StartOrEnd.START,
                min_or_max=MinOrMax.MIN,
            )
            if offset == 0
        ]
        max_start_to_start_timelags_0_offset = [
            (t1, t2) for t2, t1 in min_start_to_start_timelags_0_offset
        ]
        start_together = set(min_start_to_start_timelags_0_offset).intersection(
            max_start_to_start_timelags_0_offset
        )
        for task, next_tasks in successors.items():
            for task1, task2 in start_together:
                if task1 in next_tasks:
                    next_tasks.add(task2)

        return successors

    def update_time_lags(self) -> None:
        """Method to call when time lags have been updated.

        Clear cache from consolidated precedence constraints and time lags.

        Returns:

        """
        super().get_consolidated_time_lags.cache_clear()  # beware: parent class method also using cache !
        self.get_consolidated_precedence_constraints.cache_clear()

    def update_precedence_constraints(self) -> None:
        """Method to call when precedence constraints have been updated.

        Clear cache from consolidated precedence constraints and time lags.

        Returns:

        """
        self.update_time_lags()

    def compute_subobjective(
        self,
        variable: GenericSchedulingSolution,
        objective: Objective,
        resource_weights: Optional[dict[AnyResource, int]] = None,
    ) -> int:
        """Compute subobjective from given solution."""
        match objective:
            case Objective.MAKESPAN:
                return variable.get_max_end_time()
            case Objective.NB_TASKS_DONE:
                return variable.compute_nb_tasks_done()
            case Objective.NB_UNARY_RESOURCES_USED:
                return variable.compute_nb_unary_resources_used()
            case Objective.NB_RESOURCES_USED:
                return variable.compute_nb_calendar_resources_used(
                    weights=resource_weights
                ) + variable.compute_nb_non_renewable_resources_used(
                    weights=resource_weights
                )
            case Objective.RESOURCES_LEVELS:
                return variable.compute_aggregated_calendar_resources_levels(
                    weights=resource_weights
                ) + variable.compute_aggregated_non_renewable_resources_consumptions(
                    weights=resource_weights
                )
            case Objective.COST:
                return variable.compute_cost()
            case _:
                raise NotImplementedError()

    @wrapt.lru_cache(maxsize=None)
    def compute_task_durations_with_calendar_preemption(
        self, horizon: Optional[int] = None
    ) -> CalendarPreemptionData:
        """Compute preemptive durations for all tasks/modes considering resource calendars.

        For each (task, mode) combination, computes how the actual duration varies based on
        start time due to resource availability constraints from cumulative resources.

        This method automatically extracts the necessary data from the problem structure and
        caches the result for efficiency.

        Args:
            horizon: problem horizon (max time), defaults to get_makespan_upper_bound()

        Returns:
            CalendarPreemptionData containing:
            - durations: mapping from (task, mode) to (duration_array, interval_dict)
            - resource_calendar_dict: binary calendars for each resource consumption pattern
            - task_mode_to_calendar: mapping from (task, mode) to calendar key

        Example:
            >>> data = problem.compute_task_durations_with_calendar_preemption()
            >>> duration_array, interval_dict = data.durations[(task_1, 1)]
            >>> actual_duration = data.get_duration_for_start_time(task_1, 1, start_time=10)

        """
        if horizon is None:
            horizon = self.get_makespan_upper_bound()

        # Build resource_availabilities from problem calendars
        resource_availabilities = {
            resource: np.array(
                self.get_resource_calendar(resource=resource, horizon=horizon)
            )
            for resource in self.cumulative_resources_list
        }

        # Compute using the implementation from compute_task_durations_with_calendars
        # but inline it here to work directly with problem data
        resource_calendar_dict = {}
        durations = {}
        task_mode_to_calendar = {}

        for task in self.tasks_list:
            for mode in self.get_task_modes(task):
                duration = self.get_task_mode_duration(task=task, mode=mode)

                # Get resource consumption for this task/mode
                resource_consumption = {
                    resource: self.get_cumulative_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in self.cumulative_resources_list
                }

                # Filter to non-zero consumption
                resource_consumption = {
                    r: resource_consumption[r]
                    for r in resource_consumption
                    if resource_consumption[r] > 0
                }

                if len(resource_consumption) == 0:
                    # No resource required
                    durations[(task, mode)] = ([], {duration: [[0, horizon]]})
                else:
                    # Create a hashable key for this resource consumption pattern
                    if len(resource_consumption) == 1:
                        # Single resource
                        res_id = list(resource_consumption.keys())[0]
                        calendar_key = (res_id, resource_consumption[res_id])
                    else:
                        # Multiple resources - use tuple of (resource, consumption) pairs
                        calendar_key = tuple(sorted(resource_consumption.items()))

                    # Compute binary calendar if not cached
                    if calendar_key not in resource_calendar_dict:
                        binary_calendar = (
                            compute_binary_calendar_for_resource_consumption(
                                resource_availabilities=resource_availabilities,
                                resource_consumption=resource_consumption,
                            )
                        )
                        resource_calendar_dict[calendar_key] = binary_calendar

                    # Compute duration with preemption
                    binary_calendar = resource_calendar_dict[calendar_key]
                    durations[(task, mode)] = compute_duration_with_calendar_preemption(
                        orig_duration=duration,
                        resource_calendar=binary_calendar,
                        cumulative_resource_calendar=np.cumsum(binary_calendar),
                    )
                    task_mode_to_calendar[(task, mode)] = calendar_key

        return CalendarPreemptionData(
            durations=durations,
            resource_calendar_dict=resource_calendar_dict,
            task_mode_to_calendar=task_mode_to_calendar,
        )

    def compute_penalty(
        self, variable: GenericSchedulingSolution, penalty: Penalty
    ) -> int:
        """Compute penalty from given solution."""
        match penalty:
            case Penalty.TIME:
                penalty = 0
                # time windows
                for task in self.tasks_list:
                    start = variable.get_start_time(task)
                    end = variable.get_end_time(task)
                    start_lb = self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.START
                    )
                    end_lb = self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.END
                    )
                    start_ub = self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.START
                    )
                    end_ub = self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.END
                    )
                    penalty += max(0, start_lb - start)
                    penalty += max(0, end_lb - end)
                    penalty += max(0, start - start_ub)
                    penalty += max(0, end - end_ub)
                # time lags
                for task1_start_or_end in StartOrEnd:
                    for task2_start_or_end in StartOrEnd:
                        for min_or_max in MinOrMax:
                            for task1, task2, offset in self.get_original_time_lags(
                                task1_start_or_end=task1_start_or_end,
                                task2_start_or_end=task2_start_or_end,
                                min_or_max=min_or_max,
                            ):
                                t1 = variable.get_start_or_end_time(
                                    task=task1, start_or_end=task1_start_or_end
                                )
                                t2 = variable.get_start_or_end_time(
                                    task=task2, start_or_end=task2_start_or_end
                                )
                                if min_or_max == MinOrMax.MIN:
                                    penalty += max(0, t1 + offset - t2)
                                else:
                                    penalty += max(0, t2 - (t1 + offset))

            case _:
                raise NotImplementedError()

        return penalty

    def get_mode_cost(self, task: Task, mode: int) -> int:
        """Get cost of choosing given mode.

        Default to no cost. To be overridden in child classes with actual costs.

        Args:
            task:
            mode:

        Returns:

        """
        return 0

    def get_unary_resource_cost(
        self, task: Task, mode: int, unary_resource: UnaryResource
    ) -> int:
        """Get cost of allocating given unary resource.

        Default to no cost. To be overridden in child classes with actual costs.

        Args:
            task:
            mode:
            unary_resource:

        Returns:

        """
        return 0


class GenericSchedulingSolution(
    SkillSolution[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, UnaryResource
    ],
    NonRenewableResourceSolution[Task, NonRenewableResource],
    PrecedenceSchedulingSolution[Task],
    TimelagSolution[Task],
    TimewindowSolution[Task],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
):
    """Solution type associated to GenericSchedulingProblem."""

    problem: GenericSchedulingProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]

    def get_calendar_resource_consumption(self, resource: Resource, task: Task) -> int:
        """"""
        if self.problem.is_unary_resource(resource=resource):
            # unary resources: 0 (not allocated) or 1 (allocated)
            return int(self.is_allocated(task=task, unary_resource=resource))
        else:
            # cumulative resources
            return super().get_calendar_resource_consumption(
                resource=resource, task=task
            )

    def compute_cost(self) -> int:
        return sum(
            (
                self.problem.get_mode_cost(
                    task=task, mode=(mode := self.get_mode(task=task))
                )
                + sum(
                    self.problem.get_unary_resource_cost(
                        task=task, mode=mode, unary_resource=unary_resource
                    )
                    for unary_resource in self.get_task_allocation(task=task)
                )
            )
            for task in self.problem.tasks_list
        )
