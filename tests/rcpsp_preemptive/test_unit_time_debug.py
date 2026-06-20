#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Debug script for unit time solver."""

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp_preemptive.solvers.cpsat import (
    CpSatPreemptiveRcpspSolverUnitTime,
)


def create_simple_problem():
    """Create a very simple problem for debugging."""
    from discrete_optimization.rcpsp_preemptive.problem import PreemptiveRcpspProblem

    return PreemptiveRcpspProblem(
        resources={"R1": [2] * 20},
        non_renewable_resources=[],
        mode_details={
            "A0": {1: {"duration": 0}},
            "A1": {1: {"duration": 3, "R1": 1}},
            "A2": {1: {"duration": 2, "R1": 1}},
            "A3": {1: {"duration": 0}},
        },
        successors={
            "A0": ["A1", "A2"],
            "A1": ["A3"],
            "A2": ["A3"],
            "A3": [],
        },
        horizon=20,
        tasks_list=["A0", "A1", "A2", "A3"],
        source_task="A0",
        sink_task="A3",
    )


if __name__ == "__main__":
    problem = create_simple_problem()
    print(f"Problem: {problem.n_jobs} tasks, horizon={problem.horizon}")
    print(f"Tasks: {problem.tasks_list}")

    solver = CpSatPreemptiveRcpspSolverUnitTime(problem)
    print("Initializing model...")
    solver.init_model()

    print("Model initialized successfully")
    print(f"Variables created: {list(solver.variables.keys())}")

    # Check variable counts
    for task in problem.tasks_list:
        if task in solver.variables["starts"]:
            n_parts = len(solver.variables["starts"][task])
            print(f"Task {task}: {n_parts} parts")

    p = ParametersCp.default_cpsat()
    p.time_limit = 30

    print("\nSolving...")
    res = solver.solve(
        parameters_cp=p, ortools_cpsat_solver_kwargs={"log_search_progress": True}
    )

    print(f"\nResult storage size: {len(res)}")

    if len(res) > 0:
        solution = res.get_best_solution()
        print(f"Solution found!")
        print(f"Fitness: {problem.evaluate(solution)}")
        print(f"Satisfy: {problem.satisfy(solution)}")
        print(f"\nSchedule:")
        for task in problem.tasks_list:
            sched = solution.rcpsp_schedule[task]
            print(f"  {task}: starts={sched['starts']}, ends={sched['ends']}")
    else:
        print("No solution found!")
        print("This might indicate a bug in the model constraints.")
