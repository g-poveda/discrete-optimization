#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solver.toulbar_solver import ToulbarRCPSPSolver


def test_toolb():
    import logging

    logging.basicConfig(level=logging.DEBUG)
    files_available = get_data_available()
    file = [f for f in files_available if "j301_2.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = ToulbarRCPSPSolver(problem=rcpsp_problem)
    solver.init_model()
    res = solver.solve()
    sol = res.get_best_solution()
    print(sol)


if __name__ == "__main__":
    run_toolb()
