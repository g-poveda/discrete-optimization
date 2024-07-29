#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.vrp.plots.plot_vrp import plot_vrp_solution
from discrete_optimization.vrp.solver.vrp_cpsat_solver import CpSatVrpSolver
from discrete_optimization.vrp.vrp_model import VrpSolution
from discrete_optimization.vrp.vrp_parser import get_data_available, parse_file

logging.basicConfig(level=logging.INFO)


def run_cpsat_vrp():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    print(problem)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=False, cut_transition=False)
    p = ParametersCP.default_cpsat()
    p.nb_process = 10
    p.time_limit = 100
    res = solver.solve(parameters_cp=p)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    plot_vrp_solution(vrp_model=problem, solution=sol)
    plt.show()


def run_cpsat_vrp_on_tsp():
    from discrete_optimization.tsp.tsp_model import SolutionTSP
    from discrete_optimization.tsp.tsp_parser import (
        Point2D,
        TSPModel2D,
        get_data_available,
        parse_file,
    )
    from discrete_optimization.vrp.vrp_model import Customer2D, VrpProblem2D

    file = [f for f in get_data_available() if "tsp_200_1" in f][0]
    problem_tsp: TSPModel2D = parse_file(file_path=file, start_index=0, end_index=10)
    problem = VrpProblem2D(
        vehicle_count=1,
        vehicle_capacities=[100000],
        customer_count=problem_tsp.node_count,
        customers=[
            Customer2D(
                name=str(i),
                demand=0,
                x=problem_tsp.list_points[i].x,
                y=problem_tsp.list_points[i].y,
            )
            for i in range(len(problem_tsp.list_points))
        ],
        start_indexes=[problem_tsp.start_index],
        end_indexes=[problem_tsp.end_index],
    )
    print(problem)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=False, cut_transition=False)
    p = ParametersCP.default_cpsat()
    p.nb_process = 10
    p.time_limit = 10
    res = solver.solve(parameters_cp=p)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    plot_vrp_solution(vrp_model=problem, solution=sol)
    sol_tsp = SolutionTSP(
        problem=problem_tsp,
        start_index=problem_tsp.start_index,
        end_index=problem_tsp.end_index,
        permutation=sol.list_paths[0],
    )
    assert problem_tsp.satisfy(sol_tsp)
    plt.show()


if __name__ == "__main__":
    run_cpsat_vrp_on_tsp()
