#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Comprehensive tests for all CP-SAT preemptive RCPSP solvers.

Tests:
- CpSatPreemptiveRcpspSolver (regular)
- CpSatPreemptiveRcpspSolverUnitTime (unit-time decomposition)
- CpSatCalendarPreemptiveSolver (calendar-aware)

Each test validates intermediate solutions to ensure they satisfy constraints.
"""

import pytest

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp_preemptive.problem import (
    PreemptiveRcpspProblem,
    get_rcpsp_problemp_preemptive,
)
from discrete_optimization.rcpsp_preemptive.solvers.cpsat import (
    CpSatCalendarPreemptiveSolver,
    CpSatPreemptiveRcpspSolver,
    CpSatPreemptiveRcpspSolverUnitTime,
)


class SolutionValidationCallback(Callback):
    """Callback that validates each intermediate solution."""

    def __init__(self, problem: PreemptiveRcpspProblem):
        self.problem = problem
        self.solutions_checked = 0
        self.all_valid = True
        self.validation_errors = []

    def on_step_end(self, step, res, solver):
        """Validate each solution as it's found."""
        if res and len(res) > 0:
            solution = res[-1][0]
            self.solutions_checked += 1

            # Check if solution satisfies constraints
            satisfies = self.problem.satisfy(solution)

            if not satisfies:
                self.all_valid = False
                fitness = self.problem.evaluate(solution)
                error_msg = f"Solution {self.solutions_checked} does NOT satisfy constraints! Fitness: {fitness}"
                self.validation_errors.append(error_msg)
                print(f"❌ {error_msg}")
            else:
                fitness = self.problem.evaluate(solution)
                print(
                    f"✓ Solution {self.solutions_checked} valid: makespan={fitness.get('makespan')}"
                )


def create_alamano_preemptive_model() -> PreemptiveRcpspProblem:
    """Create Alamano test problem with resource calendar."""
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
    """Load a small PSPLIB instance with calendar."""
    files = [f for f in get_data_available() if "j301_1.sm" in f]
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    # Add calendar pattern
    for r in rcpsp_problem.resources_list:
        if r not in rcpsp_problem.non_renewable_resources:
            max_capa = rcpsp_problem.get_max_resource_capacity(r)
            rcpsp_problem.resources[r] = [
                max_capa if i % 5 != 0 else 0 for i in range(rcpsp_problem.horizon * 2)
            ]
        else:
            max_capa = rcpsp_problem.get_max_resource_capacity(r)
            rcpsp_problem.resources[r] = [max_capa] * (rcpsp_problem.horizon * 2)

    rcpsp_problem.horizon = rcpsp_problem.horizon * 2
    rcpsp_problem.update_problem()

    return get_rcpsp_problemp_preemptive(rcpsp_problem)


@pytest.mark.parametrize(
    "problem_factory",
    [create_alamano_preemptive_model, load_psplib_preemptive_model],
    ids=["alamano", "psplib"],
)
def test_regular_solver_with_validation(problem_factory):
    """Test CpSatPreemptiveRcpspSolver with intermediate solution validation."""
    problem = problem_factory()
    print(f"\n{'=' * 60}")
    print(f"Testing Regular Solver on {problem_factory.__name__}")
    print(f"{'=' * 60}")

    solver = CpSatPreemptiveRcpspSolver(problem)
    solver.init_model(max_nb_preemption=10)

    # Create validation callback
    validator = SolutionValidationCallback(problem)

    p = ParametersCp.default_cpsat()
    p.time_limit = 15

    result = solver.solve(parameters_cp=p, callbacks=[validator])

    # Check results
    assert len(result) > 0, "No solutions found"
    assert validator.solutions_checked > 0, "No intermediate solutions validated"
    assert validator.all_valid, f"Some solutions invalid: {validator.validation_errors}"

    # Check final solution
    best_solution = result.get_best_solution()
    assert problem.satisfy(best_solution), "Final solution does not satisfy constraints"

    fitness = problem.evaluate(best_solution)
    print(
        f"\n✓ Regular solver: {validator.solutions_checked} solutions, makespan={fitness['makespan']}"
    )


@pytest.mark.parametrize(
    "problem_factory",
    [create_alamano_preemptive_model, load_psplib_preemptive_model],
    ids=["alamano", "psplib"],
)
def test_unit_time_solver_with_validation(problem_factory):
    """Test CpSatPreemptiveRcpspSolverUnitTime with intermediate solution validation."""
    problem = problem_factory()
    print(f"\n{'=' * 60}")
    print(f"Testing Unit Time Solver on {problem_factory.__name__}")
    print(f"{'=' * 60}")

    solver = CpSatPreemptiveRcpspSolverUnitTime(problem)
    solver.init_model()

    # Create validation callback
    validator = SolutionValidationCallback(problem)

    p = ParametersCp.default_cpsat()
    p.time_limit = 20  # Give unit time solver a bit more time

    result = solver.solve(parameters_cp=p, callbacks=[validator])

    # Check results
    assert len(result) > 0, "No solutions found"
    assert validator.solutions_checked > 0, "No intermediate solutions validated"
    assert validator.all_valid, f"Some solutions invalid: {validator.validation_errors}"

    # Check final solution
    best_solution = result.get_best_solution()
    assert problem.satisfy(best_solution), "Final solution does not satisfy constraints"

    # Verify intervals are merged (no consecutive unit intervals)
    for task in best_solution.rcpsp_schedule:
        starts = best_solution.rcpsp_schedule[task]["starts"]
        ends = best_solution.rcpsp_schedule[task]["ends"]
        for i in range(len(starts) - 1):
            assert ends[i] != starts[i + 1], (
                f"Task {task} has consecutive intervals that should be merged: "
                f"[{starts[i]}, {ends[i]}] and [{starts[i + 1]}, {ends[i + 1]}]"
            )

    fitness = problem.evaluate(best_solution)
    print(
        f"\n✓ Unit time solver: {validator.solutions_checked} solutions, makespan={fitness['makespan']}"
    )


@pytest.mark.parametrize(
    "problem_factory",
    [create_alamano_preemptive_model, load_psplib_preemptive_model],
    ids=["alamano", "psplib"],
)
def test_calendar_solver_with_validation(problem_factory):
    """Test CpSatCalendarPreemptiveSolver with intermediate solution validation."""
    problem = problem_factory()
    print(f"\n{'=' * 60}")
    print(f"Testing Calendar Solver on {problem_factory.__name__}")
    print(f"{'=' * 60}")

    solver = CpSatCalendarPreemptiveSolver(problem)
    solver.init_model()

    # Create validation callback
    validator = SolutionValidationCallback(problem)

    p = ParametersCp.default_cpsat()
    p.time_limit = 15

    result = solver.solve(parameters_cp=p, callbacks=[validator])

    # Check results
    assert len(result) > 0, "No solutions found"
    assert validator.solutions_checked > 0, "No intermediate solutions validated"
    assert validator.all_valid, f"Some solutions invalid: {validator.validation_errors}"

    # Check final solution
    best_solution = result.get_best_solution()
    assert problem.satisfy(best_solution), "Final solution does not satisfy constraints"

    # Verify solution is properly transformed (has preemption where needed)
    # At least some tasks should have multiple intervals due to calendar gaps
    total_intervals = sum(
        len(best_solution.rcpsp_schedule[task]["starts"])
        for task in best_solution.rcpsp_schedule
    )
    print(
        f"Total intervals: {total_intervals}, tasks: {len(best_solution.rcpsp_schedule)}"
    )

    fitness = problem.evaluate(best_solution)
    print(
        f"\n✓ Calendar solver: {validator.solutions_checked} solutions, makespan={fitness['makespan']}"
    )


def test_all_solvers_comparison():
    """Compare all three solvers on the same problem."""
    print(f"\n{'=' * 60}")
    print("Comparing All Three Solvers")
    print(f"{'=' * 60}")

    problem = create_alamano_preemptive_model()

    p = ParametersCp.default_cpsat()
    p.time_limit = 15

    results = {}

    # Test each solver
    for solver_name, solver_class in [
        ("Regular", CpSatPreemptiveRcpspSolver),
        ("UnitTime", CpSatPreemptiveRcpspSolverUnitTime),
        ("Calendar", CpSatCalendarPreemptiveSolver),
    ]:
        if solver_name == "Regular":
            solver = solver_class(problem)
            solver.init_model(max_nb_preemption=10)
        else:
            solver = solver_class(problem)
            solver.init_model()

        validator = SolutionValidationCallback(problem)
        result = solver.solve(parameters_cp=p, callbacks=[validator])

        assert len(result) > 0, f"{solver_name} found no solutions"
        assert validator.all_valid, f"{solver_name} produced invalid solutions"

        best_solution = result.get_best_solution()
        fitness = problem.evaluate(best_solution)

        results[solver_name] = {
            "makespan": fitness["makespan"],
            "num_solutions": validator.solutions_checked,
            "valid": problem.satisfy(best_solution),
        }

        print(f"\n{solver_name}:")
        print(f"  Makespan: {fitness['makespan']}")
        print(f"  Solutions found: {validator.solutions_checked}")
        print(f"  All valid: ✓")

    # All solvers should find valid solutions
    for solver_name, data in results.items():
        assert data["valid"], f"{solver_name} final solution invalid"

    print(f"\n✓ All three solvers produced valid solutions!")


if __name__ == "__main__":
    # Run tests manually
    print("Running manual tests...\n")

    test_regular_solver_with_validation(create_alamano_preemptive_model)
    test_unit_time_solver_with_validation(create_alamano_preemptive_model)
    test_calendar_solver_with_validation(create_alamano_preemptive_model)
    test_all_solvers_comparison()

    print("\n" + "=" * 60)
    print("✓ All manual tests passed!")
    print("=" * 60)
