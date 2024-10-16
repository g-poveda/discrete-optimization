import logging

from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.solvers.asp import AspMisSolver
from discrete_optimization.maximum_independent_set.solvers.decomposition import (
    MisProblem,
)
from discrete_optimization.maximum_independent_set.solvers.toulbar import (
    ToulbarMisSolver,
)


def run_toulbar_solver():
    small_example = [f for f in get_data_available() if "1dc.1024" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = ToulbarMisSolver(problem=mis_model)
    solver.init_model()  # (UB=-160)
    res = solver.solve(time_limit=300)
    sol, fit = res.get_best_solution_fit()
    print(mis_model.satisfy(sol))
    print(mis_model.evaluate(sol))


if __name__ == "__main__":
    run_toulbar_solver()
