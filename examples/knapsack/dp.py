#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.dp import DpKnapsackSolver, dp


def run():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver

    solver = GreedyBestKnapsackSolver(problem=knapsack_problem)
    sol, fit = solver.solve().get_best_solution_fit()
    print(fit, " current sol")
    solver = DpKnapsackSolver(problem=knapsack_problem)
    solver.init_model()
    # solver.set_warm_start(solution=sol)
    res = solver.solve(solver=dp.LNBS, time_limit=100)
    # initial_solution=solver.initial_transitions)
    sol = res.get_best_solution()
    print(knapsack_problem.satisfy(sol))
    print(knapsack_problem.max_capacity)
    print(sol, "\n", sol)


def run_optuna():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    from discrete_optimization.generic_tools.optuna.utils import (
        generic_optuna_experiment_monoproblem,
    )

    study = generic_optuna_experiment_monoproblem(
        problem=knapsack_problem, solvers_to_test=[DpKnapsackSolver]
    )


if __name__ == "__main__":
    run()