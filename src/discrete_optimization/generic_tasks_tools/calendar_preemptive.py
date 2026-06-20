#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Module containing utils for problem having calendar based preemption:
This means the tasks can overlap over a period where the resource are not met for this task:
the duration of tasks can be pre-computed based on the start time.
The hypothesis is that the task will be executed whenever there is enough cumulative resource
available at a given time.
"""

import logging
from abc import abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Generic, Iterable, Union

import numpy as np
import wrapt

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
    CumulativeResourceProblem,
    CumulativeResourceSolution,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.preemptive import (
    OtherCalendarResource,
)

logger = logging.getLogger(__name__)
Resource = CumulativeResource | OtherCalendarResource


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
        self, problem: "CalendarPreemptiveProblem"
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


class CalendarPreemptiveProblem(
    CumulativeResourceProblem[Task, CumulativeResource, OtherCalendarResource],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    @abstractmethod
    def is_task_calendar_preempted(self, task: Task) -> bool:
        """
        Returns True if the task can calendar preempted
        """
        ...

    @wrapt.lru_cache(maxsize=None)
    def get_all_tasks_calendar_preempted(self):
        return [t for t in self.tasks_list if self.is_task_calendar_preempted(t)]

    @wrapt.lru_cache(maxsize=None)
    def compute_task_durations_with_calendar_preemption(
        self, horizon: int | None = None
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
            #>>> data = problem.compute_task_durations_with_calendar_preemption()
            #>>> duration_array, interval_dict = data.durations[(task_1, 1)]
            #>>> actual_duration = data.get_duration_for_start_time(task_1, 1, start_time=10)

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
            if not self.is_task_calendar_preempted(task):
                # We skip.
                continue
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

    def get_task_mode_duration_span_from_start_time(
        self, task: Task, mode: int, start_time: int
    ):
        if not self.is_task_calendar_preempted(task):
            return self.get_task_mode_duration(task=task, mode=mode)
        return self.compute_task_durations_with_calendar_preemption().get_duration_for_start_time(
            task=task, mode=mode, start_time=start_time
        )

    def get_possible_durations_for_task_mode(self, task: Task, mode: int) -> list[int]:
        if not self.is_task_calendar_preempted(task):
            return [self.get_task_mode_duration(task, mode)]
        return [
            int(d)
            for d in self.compute_task_durations_with_calendar_preemption().durations[
                (task, mode)
            ][1]
            if d >= 0
        ]

    def get_possible_durations_for_task(self, task: Task) -> list[int]:
        durs = []
        for m in self.get_task_modes(task):
            durs += self.get_possible_durations_for_task_mode(task=task, mode=m)
        return sorted(list(set(durs)))


class CalendarPreemptiveSolution(
    CumulativeResourceSolution[Task, CumulativeResource, OtherCalendarResource]
):
    problem: CalendarPreemptiveProblem[Task, CumulativeResource, OtherCalendarResource]

    def check_calendar_preemption_constraint(self) -> bool:
        for task in self.problem.get_all_tasks_calendar_preempted():
            dur = self.get_end_time(task=task) - self.get_start_time(task=task)
            start = self.get_start_time(task=task)
            mode = self.get_mode(task=task)
            expected_dur = self.problem.get_task_mode_duration_span_from_start_time(
                task=task, mode=mode, start_time=start
            )
            if expected_dur != dur:
                logger.debug(f"expected duration {expected_dur} != duration {dur}")
                return False
        return True

    def _compute_calendar_resource_consumption_np(
        self, resources: Iterable[Resource]
    ) -> np.ndarray:
        # Override the util function, so that the cumulative calendar resource constraint
        # is well checked ! we remove the consumption of the task on its idle time.
        makespan = self.get_max_end_time()
        resources_consumption = {
            resource: np.zeros(makespan, dtype=int) for resource in resources
        }
        for task in self.problem.tasks_list:
            start = self.get_start_time(task)
            end = self.get_end_time(task)
            mode = self.get_mode(task)
            if task in self.problem.get_all_tasks_calendar_preempted():
                for resource in resources:
                    resources_consumption[resource][start:end] += (
                        self.get_calendar_resource_consumption(
                            resource=resource, task=task
                        )
                    )
            else:
                mask = self.problem.compute_task_durations_with_calendar_preemption().get_binary_calendar(
                    task, mode
                )
                for resource in resources:
                    resource[resource][start:end] += np.multiply(
                        mask[start:end],
                        self.get_calendar_resource_consumption(
                            resource=resource, task=task
                        ),
                    )
        return resources_consumption


class NoCalendarPreemptiveProblem(
    CalendarPreemptiveProblem[Task, CumulativeResource, OtherCalendarResource]
):
    def is_task_calendar_preempted(self, task: Task) -> bool:
        return False


def compute_binary_calendar_for_resource_consumption(
    resource_availabilities: dict[Any, np.ndarray],
    resource_consumption: dict[Any, int],
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
