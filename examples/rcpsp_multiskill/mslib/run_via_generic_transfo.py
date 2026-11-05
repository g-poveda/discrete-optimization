#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tasks_tools.solvers.cpmpy.auto_impl import (
    GenericSchedulingAutoCpmpyImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.optal.auto_impl import (
    GenericSchedulingAutoOptalImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.rcpsp_multiskill.parser_mslib import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspProblem
from discrete_optimization.rcpsp_multiskill.transformations.generic_scheduling_impl import (
    MultiskillRcpspToGenericSchedulingTransformation,
)

logging.basicConfig(level=logging.INFO)


def run_generic(
    solver_backend: str = "cpmpy", problem: MultiskillRcpspProblem = None
) -> ResultStorage:
    solver_cls = None
    kwargs = {}
    p = ParametersCp.default_cpsat()
    match solver_backend:
        case "cpsat":
            solver_cls = GenericSchedulingAutoCpSatImplSolver
            kwargs = {
                "time_limit": 100,
                "parameters_cp": p,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            }
        case "cpmpy":
            solver_cls = GenericSchedulingAutoCpmpyImplSolver
            kwargs = {
                "time_limit": 100,
                "parameters_cp": p,
                "solver_name": "ortools",
                "log_search_progress": True,
                "verbose": True,
            }
        case "optal":
            solver_cls = GenericSchedulingAutoOptalImplSolver
            kwargs = {
                "time_limit": 100,
                "parameters_cp": p,
            }
    solver = TransformationSolver(
        transformation=MultiskillRcpspToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            solver_cls,
            kwargs,
        ),
        source_problem=problem,
    )
    res = solver.solve(
        callbacks=[
            ObjectiveGapStopper(0, 0),
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
            BasicStatsCallback(),
        ],
    )
    return res


def run_via_generic():
    files_dict = get_data_available()
    file = [f for f in files_dict["MSLIB4"] if "MSLIB_Set4_1003.msrcp" in f][0]
    problem = parse_file(file, skill_level_version=False)
    res = run_generic(solver_backend="cpmpy", problem=problem)
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    assert problem.satisfy(sol)


if __name__ == "__main__":
    run_via_generic()
