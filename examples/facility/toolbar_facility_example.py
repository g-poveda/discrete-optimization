from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.facility.solvers.facility_toulbar_solver import (
    ToulbarFacilitySolve,
)


def run_baseline_solve():
    import logging

    from discrete_optimization.facility.solvers.facility_lp_solver import (
        LP_Facility_Solver_CBC,
        ParametersMilp,
    )

    logging.basicConfig(level=logging.DEBUG)
    file = [f for f in get_data_available() if "fl_16_1" in f][0]
    model: FacilityProblem = parse_file(file)
    solver = LP_Facility_Solver_CBC(facility_problem=model)
    solver.init_model()
    params_milp = ParametersMilp.default()
    params_milp.time_limit = 100
    res = solver.solve(parameters_milp=params_milp)
    sol = res.get_best_solution()
    print(model.evaluate(sol))


def run_toolbar():
    import logging

    logging.basicConfig(level=logging.DEBUG)
    file = [f for f in get_data_available() if "fl_16_1" in f][0]
    model: FacilityProblem = parse_file(file)
    solver = ToulbarFacilitySolve(problem=model)
    res = solver.solve(upper_bound=10e7)
    print("RES = ", res)


if __name__ == "__main__":
    # run_solve()
    run_toolbar()
