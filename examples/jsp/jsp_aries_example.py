#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.jsp.job_shop_parser import get_data_available, parse_file
from discrete_optimization.jsp.job_shop_problem import JobShopProblem
from discrete_optimization.jsp.solvers.aries_jsp_solver import AriesJspSolver

logging.basicConfig(level=logging.INFO)


def run_aries_jsp():
    file_path = get_data_available()[1]
    # file_path = [f for f in get_data_available() if "abz6" in f][0]
    problem = parse_file(file_path)
    print("File path ", file_path)
    print(
        "Problem with ",
        problem.n_jobs,
        " jobs, ",
        problem.n_all_jobs,
        " subjobs, and ",
        problem.n_machines,
        " machines",
    )
    solver = AriesJspSolver(problem=problem)
    res = solver.solve(time_limit=20, search="learning-rate")
    sol = res.get_best_solution_fit()[0]
    print(problem.evaluate(sol))
    assert problem.satisfy(res.get_best_solution_fit()[0])


if __name__ == "__main__":
    run_aries_jsp()
