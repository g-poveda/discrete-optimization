#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from didppy import BeamParallelizationMethod

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.jsp.job_shop_parser import get_data_available, parse_file
from discrete_optimization.jsp.job_shop_problem import JobShopProblem
from discrete_optimization.jsp.job_shop_utils import transform_jsp_to_rcpsp
from discrete_optimization.jsp.solvers.did_jsp_solver import DidJspSolver, dp
from discrete_optimization.rcpsp.solver.did_rcpsp_solver import DidRCPSPSolver

logging.basicConfig(level=logging.INFO)


def debug():
    model = dp.Model()
    cur_time_per_machine = [model.add_int_var(target=0) for m in range(4)]
    cur_time_per_job = [model.add_int_var(target=1) for m in range(2)]
    max_total = model.add_int_var(target=0)
    t = dp.Transition(
        name="t",
        cost=dp.IntExpr.state_cost() + 1,
        preconditions=[],
        effects=[
            (
                cur_time_per_machine[0],
                dp.max(cur_time_per_machine[0] + 2, cur_time_per_job[0] + 2),
            ),
            (
                cur_time_per_job[0],
                dp.max(cur_time_per_machine[0] + 2, cur_time_per_job[0] + 2),
            ),
            (
                max_total,
                dp.max(
                    max_total,
                    dp.max(cur_time_per_machine[0] + 2, cur_time_per_job[0] + 2),
                ),
            ),
        ],
    )
    state = model.target_state
    # preconditions = t.preconditions
    # preconditions[0].eval(state, model)
    print("machine", t[cur_time_per_machine[0]].eval(state, model))
    print("cur time job", t[cur_time_per_job[0]].eval(state, model))
    print("Max totoal", t[max_total].eval(state, model))
    t[max_total] = max_total + 1
    print(t[max_total].eval(state, model))


def run_did_jsp():
    # file_path = get_data_available()[1]
    file_path = [f for f in get_data_available() if "ta68" in f][0]
    problem = parse_file(file_path)
    print("File path ", file_path)
    solver = DidJspSolver(problem=problem)
    res = solver.solve(
        solver=dp.LNBS,
        time_limit=100,
        max_beam_size=2048,
        keep_all_layers=False,
        parallelization_method=BeamParallelizationMethod.Hdbs2,
    )
    sol = res.get_best_solution_fit()[0]
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))


def run_did_of_rcpsp():
    file_path = [f for f in get_data_available() if "ta68" in f][0]
    problem = parse_file(file_path)
    rcpsp_problem = transform_jsp_to_rcpsp(problem)
    solver = DidRCPSPSolver(rcpsp_problem)
    solver.init_model_multimode()
    res = solver.solve(
        solver=dp.LNBS,
        time_limit=100,
        max_beam_size=2048,
        keep_all_layers=False,
        parallelization_method=BeamParallelizationMethod.Hdbs2,
    )
    sol = res.get_best_solution_fit()[0]
    assert rcpsp_problem.satisfy(sol)
    print(rcpsp_problem.evaluate(sol))


if __name__ == "__main__":
    run_did_of_rcpsp()