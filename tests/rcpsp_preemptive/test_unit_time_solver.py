#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Test the CpSatPreemptiveRcpspSolverUnitTime solver."""

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp_preemptive.problem import (
    PreemptiveRcpspProblem,
    get_rcpsp_problemp_preemptive,
)
from discrete_optimization.rcpsp_preemptive.solvers.cpsat import (
    CpSatPreemptiveRcpspSolver,
    CpSatPreemptiveRcpspSolverUnitTime,
    merge_consecutive_unit_intervals,
)


def create_alamano_preemptive_model() -> PreemptiveRcpspProblem:
    """Create a simple test problem with resource calendar."""
    resource_r1 = []
    for i in range(40):
        resource_r1 += [3, 3, 3, 0, 0]
    return PreemptiveRcpspProblem(
        resources={"R1": resource_r1, "R2": [2] * 200, "R3": [2] * 200},
        non_renewable_resources=[],
        mode_details={
            "A0": {1: {"duration": 0}},
            "A1": {1: {"duration": 5, "R1": 3, "R2": 1}},
            "A2": {1: {"duration": 2, "R1": 1}},
            "A3": {1: {"duration": 3, "R2": 1, "R3": 1}},
            "A4": {1: {"duration": 4, "R1": 2}},
            "A5": {1: {"duration": 5, "R1": 2, "R2": 1, "R3": 2}},
            "A6": {1: {"duration": 4, "R1": 2, "R3": 1}},
            "A7": {1: {"duration": 7, "R2": 1}},
            "A8": {1: {"duration": 2, "R1": 2, "R2": 1}},
            "A9": {1: {"duration": 0}},
        },
        successors={
            "A0": ["A" + str(i) for i in range(1, 10)],
            "A1": ["A4", "A9"],
            "A2": ["A9"],
            "A3": ["A5", "A9"],
            "A4": ["A6", "A9"],
            "A5": ["A7", "A8", "A9"],
            "A6": ["A8", "A9"],
            "A7": ["A9"],
            "A8": ["A9"],
            "A9": [],
        },
        horizon=200,
        tasks_list=["A" + str(i) for i in range(10)],
        source_task="A0",
        sink_task="A9",
    )


def load_psplib_preemptive_model() -> PreemptiveRcpspProblem:
    """Load a small PSPLIB instance as preemptive problem."""
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)
    return get_rcpsp_problemp_preemptive(rcpsp_problem)


def test_merge_consecutive_intervals():
    """Test the merge_consecutive_unit_intervals function."""
    # Test case 1: consecutive intervals
    schedule = {
        "task1": {"starts": [0, 1, 2, 5, 6], "ends": [1, 2, 3, 6, 7]},
        "task2": {"starts": [10], "ends": [15]},
    }
    merged = merge_consecutive_unit_intervals(schedule)
    assert merged["task1"]["starts"] == [0, 5]
    assert merged["task1"]["ends"] == [3, 7]
    assert merged["task2"]["starts"] == [10]
    assert merged["task2"]["ends"] == [15]

    # Test case 2: no consecutive intervals
    schedule2 = {
        "task1": {"starts": [0, 3, 6], "ends": [1, 4, 7]},
    }
    merged2 = merge_consecutive_unit_intervals(schedule2)
    assert merged2["task1"]["starts"] == [0, 3, 6]
    assert merged2["task1"]["ends"] == [1, 4, 7]

    # Test case 3: empty schedule
    schedule3 = {
        "task1": {"starts": [], "ends": []},
    }
    merged3 = merge_consecutive_unit_intervals(schedule3)
    assert merged3["task1"]["starts"] == []
    assert merged3["task1"]["ends"] == []


def test_unit_time_solver_alamano():
    """Test the unit time solver on Alamano problem."""
    problem = create_alamano_preemptive_model()
    solver = CpSatPreemptiveRcpspSolverUnitTime(problem)
    solver.init_model()

    p = ParametersCp.default_cpsat()
    p.time_limit = 10
    res = solver.solve(parameters_cp=p)

    assert len(res) > 0
    solution = res.get_best_solution()

    # Verify solution is valid
    assert problem.satisfy(solution)
    fitness = problem.evaluate(solution)
    print(f"Unit time solver makespan: {fitness}")

    # Check that consecutive intervals are merged
    for task in solution.rcpsp_schedule:
        starts = solution.rcpsp_schedule[task]["starts"]
        ends = solution.rcpsp_schedule[task]["ends"]
        for i in range(len(starts) - 1):
            # No two consecutive intervals should have end[i] == start[i+1]
            assert ends[i] != starts[i + 1], (
                f"Task {task} has consecutive intervals that should be merged: "
                f"[{starts[i]}, {ends[i]}] and [{starts[i + 1]}, {ends[i + 1]}]"
            )


def test_unit_time_solver_psplib():
    """Test the unit time solver on PSPLIB instance."""
    problem = load_psplib_preemptive_model()
    solver = CpSatPreemptiveRcpspSolverUnitTime(problem)
    solver.init_model()

    p = ParametersCp.default_cpsat()
    p.time_limit = 10
    res = solver.solve(parameters_cp=p)

    assert len(res) > 0
    solution = res.get_best_solution()

    # Verify solution is valid
    assert problem.satisfy(solution)
    fitness = problem.evaluate(solution)
    print(f"Unit time solver makespan: {fitness}")


def test_compare_solvers():
    """Compare the unit time solver with the regular solver."""
    problem = load_psplib_preemptive_model()

    # Solve with regular solver
    solver1 = CpSatPreemptiveRcpspSolver(problem)
    solver1.init_model(max_nb_preemption=10)
    p1 = ParametersCp.default_cpsat()
    p1.time_limit = 20
    res1 = solver1.solve(parameters_cp=p1)
    sol1 = res1.get_best_solution()
    fitness1 = problem.evaluate(sol1)

    # Solve with unit time solver (give it more time since model is larger)
    solver2 = CpSatPreemptiveRcpspSolverUnitTime(problem)
    solver2.init_model()
    p2 = ParametersCp.default_cpsat()
    p2.time_limit = 40
    res2 = solver2.solve(parameters_cp=p2)
    sol2 = res2.get_best_solution()
    fitness2 = problem.evaluate(sol2)

    print(f"Regular solver makespan: {fitness1}")
    print(f"Unit time solver makespan: {fitness2}")

    # Both should find valid solutions
    assert problem.satisfy(sol1)
    assert problem.satisfy(sol2)

    # The unit time model is more complex, so we just verify both find feasible solutions
    # A detailed performance comparison would require longer run times and is better done
    # in a separate benchmark study
    print(
        f"Both solvers found valid solutions (detailed comparison requires longer runtime)"
    )


if __name__ == "__main__":
    test_merge_consecutive_intervals()
    print("✓ test_merge_consecutive_intervals passed")

    test_unit_time_solver_alamano()
    print("✓ test_unit_time_solver_alamano passed")

    test_unit_time_solver_psplib()
    print("✓ test_unit_time_solver_psplib passed")

    test_compare_solvers()
    print("✓ test_compare_solvers passed")

    print("\n✓ All tests passed!")
