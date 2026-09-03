#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os

import pandas as pd

from discrete_optimization.generic_tools.callbacks.loggers import (
    NbIterationTracker,
    ProblemEvaluateLogger,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.study import (
    SolverConfig,
    Study,
)
from discrete_optimization.generic_tools.study.database import is_empty_metrics
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.problem import AllocSchedulingProblem
from discrete_optimization.workforce.scheduling.solvers import ObjectivesEnum
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat_auto import (
    CPSatAutoAllocSchedulingSolver,
)

logging.basicConfig(level=logging.INFO)
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates


def run_study():
    instances = [os.path.basename(p) for p in get_data_available()]

    # instances = ["instance_42.json",
    #             "instance_170.json",
    #             "instance_252.json"]
    def load_instance(instance) -> AllocSchedulingProblem:
        file = [f for f in get_data_available() if instance in f][0]
        return parse_json_to_problem(file)

    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    p_mono_worker = ParametersCp.default_cpsat()
    p_mono_worker.nb_process = 1
    solver_configs = dict()
    solver_configs["original-cpsat"] = SolverConfig(
        cls=CPSatAllocSchedulingSolver,
        kwargs={
            "parameters_cp": p,
            "objectives": [ObjectivesEnum.NB_TEAMS],
            "adding_redundant_cumulative": True,
            "time_limit": 20,
        },
    )

    solver_configs["cpsat-auto-no-optional"] = SolverConfig(
        cls=CPSatAutoAllocSchedulingSolver,
        kwargs={
            "parameters_cp": p,
            "avoid_interval_optional": True,
            "objectives": [ObjectivesEnum.NB_TEAMS],
            "adding_redundant_cumulative": True,
            "time_limit": 20,
        },
    )
    solver_configs["cpsat-auto-optional"] = SolverConfig(
        cls=CPSatAutoAllocSchedulingSolver,
        kwargs={
            "parameters_cp": p,
            "avoid_interval_optional": False,
            "objectives": [ObjectivesEnum.NB_TEAMS],
            "adding_redundant_cumulative": True,
            "time_limit": 20,
        },
    )

    study = Study(
        name="cpsat-vs-auto",
        instances=instances,
        solver_configs=solver_configs,
        overwrite=overwrite,
        max_retry=2,
        problem_factory=load_instance,
    )
    for problem, solver, solver_kwargs in study:
        try:
            stats_cb = StatsWithBoundsCallback()
            result_store = solver.solve(
                callbacks=[
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    ProblemEvaluateLogger(logging.INFO, logging.INFO),
                ],
                **solver_kwargs,
            )
        except Exception as e:
            # failed experiment
            metrics = pd.DataFrame([])
            status = StatusSolver.ERROR
            logging.error(e)
            reason = f"{type(e)}: {str(e)}"
            success = False
        else:
            # get metrics and solver status
            status = solver.status_solver
            metrics = stats_cb.get_df_metrics()
            success = not is_empty_metrics(metrics)
            if success:
                logging.info("experiment successful")
            else:
                logging.info("experiment unsuccessful (no metrics found)")
            reason = ""
            logging.info(
                f"Instance {study.get_current_instance()},"
                f"Solver config {study.get_current_config_name()},"
                f"value={solver.get_current_best_internal_objective_value()},"
                f"bound={solver.get_current_best_internal_objective_bound()}"
            )
        # store corresponding experiment
        study.store_current_xp(
            metrics=metrics, status=status, reason=reason, success=success
        )


def run_dashboard():
    from discrete_optimization.generic_tools.dashboard import Dashboard
    from discrete_optimization.generic_tools.study import Hdf5Database

    study_name = "cpsat-vs-auto"

    # retrieve data
    with Hdf5Database(f"{study_name}.h5") as database:
        results = database.load_results()

    # launch dashboard with this data
    app = Dashboard(results=results)
    app.run()


if __name__ == "__main__":
    run_dashboard()
