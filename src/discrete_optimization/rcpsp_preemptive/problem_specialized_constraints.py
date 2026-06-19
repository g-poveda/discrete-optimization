#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Special constraints utilities for preemptive RCPSP.

This module provides utility functions for handling special constraints in preemptive RCPSP problems.
The main `SpecialConstraintsPreemptiveRcpspProblem` class has been merged into `PreemptiveRcpspProblem`,
which now supports special constraints via the `special_constraints` parameter.

For backward compatibility, this module exports:
- `SpecialConstraintsPreemptiveRcpspProblem` (alias to `PreemptiveRcpspProblem`)
- `SpecialPreemptiveRcpspSolution` (alias to `PreemptiveRcpspSolution`)

And provides utility functions:
- `evaluate_constraints`: Compute penalty for violating special constraints
- `compute_constraints_details`: Get details of which constraints are violated
- `check_solution`: Check if a solution satisfies all special constraints
"""

import logging
from typing import Union

from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)
from discrete_optimization.rcpsp.utils import intersect
from discrete_optimization.rcpsp_preemptive.problem import (
    PreemptiveRcpspProblem,
    PreemptiveRcpspSolution,
)

logger = logging.getLogger(__name__)


# Backward compatibility aliases
SpecialPreemptiveRcpspSolution = PreemptiveRcpspSolution
SpecialConstraintsPreemptiveRcpspProblem = PreemptiveRcpspProblem


def evaluate_constraints(
    solution: Union[RcpspSolution, PreemptiveRcpspSolution],
    constraints: SpecialConstraintsDescription,
) -> float:
    """Evaluate the total penalty for violating special constraints."""
    list_constraints_not_respected = compute_constraints_details(solution, constraints)
    return sum([x[-1] for x in list_constraints_not_respected])


def compute_constraints_details(
    solution: Union[RcpspSolution, PreemptiveRcpspSolution],
    constraints: SpecialConstraintsDescription,
) -> list[tuple]:
    """Compute details of which special constraints are violated."""
    if (
        "rcpsp_schedule_feasible" in solution.__dict__.keys()
        and not solution.rcpsp_schedule_feasible
    ):
        return []

    list_constraints_not_respected = []

    for t1, t2 in constraints.start_together:
        time1 = solution.get_start_time(t1)
        time2 = solution.get_start_time(t2)
        if time1 != time2:
            list_constraints_not_respected.append(
                ("start_together", t1, t2, time1, time2, abs(time2 - time1))
            )

    for t1, t2 in constraints.start_at_end:
        time1 = solution.get_end_time(t1)
        time2 = solution.get_start_time(t2)
        if time1 != time2:
            list_constraints_not_respected.append(
                ("start_at_end", t1, t2, time1, time2, abs(time2 - time1))
            )

    for t1, t2, off in constraints.start_after_end_plus_offset:
        time1 = solution.get_end_time(t1) + off
        time2 = solution.get_start_time(t2)
        if not (time2 >= time1):
            list_constraints_not_respected.append(
                (
                    "start_after_end_plus_offset",
                    t1,
                    t2,
                    time1,
                    time2,
                    abs(time2 - time1),
                )
            )

    for t1, t2, off in constraints.start_to_start_min_time_lag:
        time1 = solution.get_start_time(t1) + off
        time2 = solution.get_start_time(t2)
        if not (time2 >= time1):
            list_constraints_not_respected.append(
                (
                    "start_to_start_min_time_lag",
                    t1,
                    t2,
                    time1,
                    time2,
                    abs(time2 - time1),
                )
            )

    for t1, t2, offset in constraints.start_to_start_max_time_lag:
        time1 = solution.get_start_time(t1) + offset
        time2 = solution.get_start_time(t2)
        if not (time2 <= time1):
            list_constraints_not_respected.append(
                (
                    "start_to_start_max_time_lag",
                    t1,
                    t2,
                    time1,
                    time2,
                    abs(time2 - time1),
                )
            )

    for t1, t2 in constraints.disjunctive_tasks:
        b = intersect(
            [solution.get_start_time(t1), solution.get_end_time(t1)],
            [solution.get_start_time(t2), solution.get_end_time(t2)],
        )
        if b is not None:
            list_constraints_not_respected.append(
                ("disjunctive", t1, t2, None, None, b[1] - b[0])
            )

    for t in constraints.start_times_window:
        if constraints.start_times_window[t][0] is not None:
            if solution.get_start_time(t) < constraints.start_times_window[t][0]:
                list_constraints_not_respected.append(
                    (
                        "start_window_0",
                        t,
                        t,
                        None,
                        None,
                        constraints.start_times_window[t][0]
                        - solution.get_start_time(t),
                    )
                )
        if constraints.start_times_window[t][1] is not None:
            if solution.get_start_time(t) > constraints.start_times_window[t][1]:
                list_constraints_not_respected.append(
                    (
                        "start_window_1",
                        t,
                        t,
                        None,
                        None,
                        solution.get_start_time(t)
                        - constraints.start_times_window[t][1],
                    )
                )

    for t in constraints.end_times_window:
        if constraints.end_times_window[t][0] is not None:
            if solution.get_end_time(t) < constraints.end_times_window[t][0]:
                list_constraints_not_respected.append(
                    (
                        "end_window_0",
                        t,
                        t,
                        None,
                        None,
                        constraints.end_times_window[t][0] - solution.get_end_time(t),
                    )
                )
        if constraints.end_times_window[t][1] is not None:
            if solution.get_end_time(t) > constraints.end_times_window[t][1]:
                list_constraints_not_respected.append(
                    (
                        "end_window_1",
                        t,
                        t,
                        None,
                        None,
                        solution.get_end_time(t) - constraints.end_times_window[t][1],
                    )
                )

    return list_constraints_not_respected


def check_solution(
    problem: PreemptiveRcpspProblem,
    solution: Union[PreemptiveRcpspSolution, RcpspSolution],
) -> bool:
    """Check if a solution satisfies all special constraints."""
    if not solution.rcpsp_schedule_feasible:
        return False

    for t1, t2 in problem.special_constraints.start_together:
        if solution.get_start_time(t1) != solution.get_start_time(t2):
            return False

    for t1, t2 in problem.special_constraints.start_at_end:
        if solution.get_start_time(t2) != solution.get_end_time(t1):
            return False

    for t1, t2, off in problem.special_constraints.start_after_end_plus_offset:
        if not (solution.get_start_time(t2) >= solution.get_end_time(t1) + off):
            return False

    for t1, t2 in problem.special_constraints.disjunctive_tasks:
        b = intersect(
            [solution.get_start_time(t1), solution.get_end_time(t1)],
            [solution.get_start_time(t2), solution.get_end_time(t2)],
        )
        if b is not None:
            return False

    for t in problem.special_constraints.start_times_window:
        if problem.special_constraints.start_times_window[t][0] is not None:
            if (
                solution.get_start_time(t)
                < problem.special_constraints.start_times_window[t][0]
            ):
                return False
        if problem.special_constraints.start_times_window[t][1] is not None:
            if (
                solution.get_start_time(t)
                > problem.special_constraints.start_times_window[t][1]
            ):
                return False

    for t in problem.special_constraints.end_times_window:
        if problem.special_constraints.end_times_window[t][0] is not None:
            if (
                solution.get_end_time(t)
                < problem.special_constraints.end_times_window[t][0]
            ):
                return False
        if problem.special_constraints.end_times_window[t][1] is not None:
            if (
                solution.get_end_time(t)
                > problem.special_constraints.end_times_window[t][1]
            ):
                return False

    for t1, t2, off in problem.special_constraints.start_to_start_min_time_lag:
        if not (solution.get_start_time(t2) >= solution.get_start_time(t1) + off):
            return False

    for t1, t2, offset in problem.special_constraints.start_to_start_max_time_lag:
        if not (solution.get_start_time(t2) <= solution.get_start_time(t1) + offset):
            return False

    return True
