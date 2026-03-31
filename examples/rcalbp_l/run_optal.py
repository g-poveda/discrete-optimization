import logging

import didppy as dp
import optalcp as cp
from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcalbp_l.parser import get_data_available, parse_rcalbpl_json
from discrete_optimization.rcalbp_l.problem import plot_rcalbpl_dashboard
from discrete_optimization.rcalbp_l.solvers.optal import (
    OptalRCALBPLSolver,
)
from discrete_optimization.rcalbp_l.solvers.pareto_postprocess import (
    DpRCALBPLPostProSolver,
)

logging.basicConfig(level=logging.INFO)


def main():
    file = [f for f in get_data_available() if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)
    problem.nb_periods = 5
    problem.periods = range(problem.nb_periods)
    solver = OptalRCALBPLSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()

    p.nb_process = 12
    res = solver.solve(
        time_limit=100,
        parameters_cp=p,
        workers=[
            cp.WorkerParameters(
                searchType="FDS", noOverlapPropagationLevel=4, cumulPropagationLevel=3
            ),
            cp.WorkerParameters(
                searchType="FDSDual",
                noOverlapPropagationLevel=4,
                cumulPropagationLevel=3,
            ),
        ]
        * 2,
    )
    sol = res[-1][0]
    fig, slider = plot_rcalbpl_dashboard(problem, sol)
    print(problem.evaluate(sol), problem.satisfy(sol))


def main_easy():
    file = [f for f in get_data_available() if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)
    problem.durations = [
        [problem.durations[t][-1]] * len(problem.durations[t])
        for t in range(problem.nb_tasks)
    ]
    problem.nb_periods = 10
    problem.periods = range(problem.nb_periods)
    solver = OptalRCALBPLSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    res = solver.solve(
        time_limit=20,
        parameters_cp=p,
        workers=[
            cp.WorkerParameters(searchType="FDS"),
            cp.WorkerParameters(searchType="FDSDual"),
            cp.WorkerParameters(searchType="LNS"),
            cp.WorkerParameters(searchType="LNS"),
        ],
    )
    sol = res[-1][0]
    fig, slider = plot_rcalbpl_dashboard(problem, sol)
    print(problem.evaluate(sol), problem.satisfy(sol))


def main_sequential():
    """Test the OptalRCALBPLSolver with the sequential meta-solver."""
    file = [f for f in get_data_available() if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)

    from discrete_optimization.generic_tools.sequential_metasolver import (
        SequentialMetasolver,
        SubBrick,
    )
    from discrete_optimization.rcalbp_l.solvers.meta_solvers import (
        BackwardSequentialRCALBPLSolver,
    )

    p = ParametersCp.default_cpsat()
    p.nb_process = 12

    brick1 = SubBrick(
        BackwardSequentialRCALBPLSolver,
        kwargs=dict(
            future_chunk_size=1,
            phase2_chunk_size=5,
            time_limit_phase1=100,
            time_limit_phase2=30,
            use_sgs_warm_start=True,
            solver_class=OptalRCALBPLSolver,  # Use OptalCP solver
            parameters_cp=p,
        ),
    )

    brick2 = SubBrick(
        OptalRCALBPLSolver,
        dict(
            add_heuristic_constraint=False,
            parameters_cp=p,
            time_limit=200,
            workers=[
                cp.WorkerParameters(
                    searchType="FDS",
                    noOverlapPropagationLevel=4,
                    cumulPropagationLevel=3,
                ),
                cp.WorkerParameters(
                    searchType="FDSDual",
                    noOverlapPropagationLevel=4,
                    cumulPropagationLevel=3,
                ),
            ],
        ),
    )

    solver = SequentialMetasolver(
        list_subbricks=[
            brick1,
            brick2,  # Uncomment to add refinement step
        ],
        problem=problem,
    )

    res = solver.solve()
    if len(res) > 0:
        sol = res[-1][0]
        fig, slider = plot_rcalbpl_dashboard(problem, sol)
        postpro_solver = DpRCALBPLPostProSolver(problem=problem)
        front = postpro_solver.create_result_storage([])
        postpro_solver.init_model(from_solution=sol, max_nb_adjustments=1)
        for i in range(1, len(postpro_solver.decision_step) + 1):
            postpro_solver.init_model(from_solution=sol, max_nb_adjustments=i)
            res = postpro_solver.solve(solver=dp.CABS, time_limit=5, threads=10)
            front.extend(res[-1:])
            print(problem.evaluate(res[-1][0]))

        f1s, f2s = [], []
        for sol, fit in front:
            eval_ = problem.evaluate(sol)
            dur_rampup = eval_["ramp_up_duration"]
            nb_adjustments = eval_["nb_adjustments"]
            print(f"  Obj: {fit} | Sol: {sol}")
            if nb_adjustments >= 1:
                f1s.append(nb_adjustments)
                f2s.append(dur_rampup)
        # Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
        # Known optima for Example 9 are (1, 2) and (3, 0)
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
        plt.grid(True)
        plt.legend()
        print(problem.evaluate(sol), problem.satisfy(sol))
        plt.show()
    else:
        print("No solution found!")


def main_backward_full():
    """Test the full BackwardSequentialRCALBPLSolver with OptalCP (with CP solving in Phase 2)."""
    file = [f for f in get_data_available() if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)

    from discrete_optimization.rcalbp_l.solvers.meta_solvers import (
        BackwardSequentialRCALBPLSolver,
    )

    p = ParametersCp.default_cpsat()
    p.nb_process = 12

    solver = BackwardSequentialRCALBPLSolver(
        problem=problem,
        future_chunk_size=1,
        phase2_chunk_size=5,
        time_limit_phase1=100,
        time_limit_phase2=30,
        use_sgs_warm_start=True,
        solver_class=OptalRCALBPLSolver,  # Use OptalCP instead of CpSat
        parameters_cp=p,
    )

    res = solver.solve()
    if len(res) > 0:
        sol = res[-1][0]
        fig, slider = plot_rcalbpl_dashboard(problem, sol)
        print("Evaluation:", problem.evaluate(sol))
        print("Satisfies constraints:", problem.satisfy(sol))

        postpro_solver = DpRCALBPLPostProSolver(problem=problem)
        front = postpro_solver.create_result_storage([])
        postpro_solver.init_model(from_solution=sol, max_nb_adjustments=1)
        for i in range(1, len(postpro_solver.decision_step) + 1):
            postpro_solver.init_model(from_solution=sol, max_nb_adjustments=i)
            res = postpro_solver.solve(solver=dp.CABS, time_limit=5, threads=10)
            front.extend(res[-1:])
            print(problem.evaluate(res[-1][0]))

        f1s, f2s = [], []
        for sol, fit in front:
            eval_ = problem.evaluate(sol)
            dur_rampup = eval_["ramp_up_duration"]
            nb_adjustments = eval_["nb_adjustments"]
            print(f"  Obj: {fit} | Sol: {sol}")
            if nb_adjustments >= 1:
                f1s.append(nb_adjustments)
                f2s.append(dur_rampup)
        # Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
        # Known optima for Example 9 are (1, 2) and (3, 0)
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
        plt.grid(True)
        plt.legend()
        print(problem.evaluate(sol), problem.satisfy(sol))
        plt.show()
    else:
        print("No solution found!")


def main_balanced():
    """Test the BalancedBackwardSequentialRCALBPLSolver with OptalCP."""
    file = [f for f in get_data_available() if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)

    from discrete_optimization.rcalbp_l.solvers.meta_solvers import (
        BalancedBackwardSequentialRCALBPLSolver,
    )

    p = ParametersCp.default_cpsat()
    p.nb_process = 12

    solver = BalancedBackwardSequentialRCALBPLSolver(
        problem=problem,
        future_chunk_size=1,
        phase2_chunk_size=5,
        time_limit_phase1=100,
        time_limit_phase2=30,
        use_sgs_warm_start=True,
        solver_class=OptalRCALBPLSolver,  # Use OptalCP instead of CpSat
        parameters_cp=p,
    )

    res = solver.solve()

    if len(res) > 0:
        sol = res[-1][0]
        fig, slider = plot_rcalbpl_dashboard(problem, sol)
        print("Evaluation:", problem.evaluate(sol))
        print("Satisfies constraints:", problem.satisfy(sol))
        plt.show()
    else:
        print("No solution found!")


if __name__ == "__main__":
    # Choose which test to run:
    # main()  # Direct Optal solver (original)
    # main_easy()  # Simplified problem (original)
    main_sequential()  # Sequential meta-solver with SGS
    # main_backward_full()  # Full backward solver with CP in Phase 2
    # main_balanced()  # Balanced backward solver
