"""Benchmark study comparing preemptive RCPSP solvers.

This script uses the study framework to compare:
- CpSatPreemptiveRcpspSolver (regular)
- CpSatPreemptiveRcpspSolverUnitTime (unit-time decomposition)

On instances with varying resource calendars.
"""

import logging
import pickle

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
from discrete_optimization.generic_tools.study import SolverConfig
from discrete_optimization.generic_tools.study.database import is_empty_metrics
from discrete_optimization.generic_tools.study.study import Study
from discrete_optimization.rcpsp_preemptive.problem import PreemptiveRcpspProblem
from discrete_optimization.rcpsp_preemptive.solvers.cpsat import (
    CpSatPreemptiveRcpspSolver,
    CpSatPreemptiveRcpspSolverUnitTime,
)

# Study configuration
INSTANCES_FILE = "instances/all_instances.pkl"
TIME_LIMIT = 60
STUDY_NAME = "preemptive-rcpsp-study"
OVERWRITE = False  # Overwrite previous study with same name?
MAX_RETRY = 0  # Retry failed experiments?


def load_all_instances() -> dict[str, PreemptiveRcpspProblem]:
    """Load all benchmark instances from pickle file."""
    with open(INSTANCES_FILE, "rb") as f:
        instances = pickle.load(f)
    return instances


def problem_factory(instance_name: str) -> PreemptiveRcpspProblem:
    """Factory function to create problem instances.

    Args:
        instance_name: Name of the instance

    Returns:
        PreemptiveRcpspProblem instance
    """
    all_instances = load_all_instances()
    return all_instances[instance_name]


# Solver configurations
p = ParametersCp.default_cpsat()
p.nb_process = 6

solver_configs = {
    "regular_30s": SolverConfig(
        cls=CpSatPreemptiveRcpspSolver,
        kwargs=dict(
            time_limit=30,
            parameters_cp=p,
            max_nb_preemption=10,
        ),
    ),
    "regular_60s": SolverConfig(
        cls=CpSatPreemptiveRcpspSolver,
        kwargs=dict(
            time_limit=60,
            parameters_cp=p,
            max_nb_preemption=10,
        ),
    ),
    "unit_time_30s": SolverConfig(
        cls=CpSatPreemptiveRcpspSolverUnitTime,
        kwargs=dict(
            time_limit=30,
            parameters_cp=p,
        ),
    ),
    "unit_time_60s": SolverConfig(
        cls=CpSatPreemptiveRcpspSolverUnitTime,
        kwargs=dict(
            time_limit=60,
            parameters_cp=p,
        ),
    ),
    "unit_time_120s": SolverConfig(
        cls=CpSatPreemptiveRcpspSolverUnitTime,
        kwargs=dict(
            time_limit=120,
            parameters_cp=p,
        ),
    ),
}


def run_study(
    instances: list[str] = None,
    configs: dict[str, SolverConfig] = None,
    study_name: str = STUDY_NAME,
):
    """Run the benchmark study.

    Args:
        instances: List of instance names (None = all)
        configs: Dictionary of solver configurations (None = all)
        study_name: Name of the study
    """
    # Use all instances if not specified
    if instances is None:
        all_instances = load_all_instances()
        instances = list(all_instances.keys())

    # Use all solver configs if not specified
    if configs is None:
        configs = solver_configs

    print(f"Study: {study_name}")
    print(f"Instances: {len(instances)}")
    print(f"Solver configs: {len(configs)}")
    print(f"Total experiments: {len(instances) * len(configs)}\n")

    study = Study(
        name=study_name,
        instances=instances,
        solver_configs=configs,
        overwrite=OVERWRITE,
        max_retry=MAX_RETRY,
        problem_factory=problem_factory,
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
            # Failed experiment
            metrics = pd.DataFrame([])
            status = StatusSolver.ERROR
            logging.error(e)
            reason = f"{type(e).__name__}: {str(e)}"
            success = False
        else:
            # Get metrics and solver status
            status = solver.status_solver
            metrics = stats_cb.get_df_metrics()
            success = not is_empty_metrics(metrics)
            if success:
                logging.info("Experiment successful")
            else:
                logging.info("Experiment unsuccessful (no metrics found)")
            reason = ""
            logging.info(
                f"Instance {study.get_current_instance()}, "
                f"Solver config {study.get_current_config_name()}, "
                f"value={solver.get_current_best_internal_objective_value()}, "
                f"bound={solver.get_current_best_internal_objective_bound()}"
            )

        # Store experiment results
        study.store_current_xp(
            metrics=metrics, status=status, reason=reason, success=success
        )

    print(f"\n✓ Study completed! Results saved to {study.database_filepath}")
    return study


def quick_test():
    """Run a quick test with a few instances."""
    print("=== Quick Test ===\n")

    # Select a few small instances
    instances = [
        "j301_1_sm_periodic_5_1",
        "j301_1_sm_random_20",
    ]

    # Use only fast solver configs
    configs = {
        "regular_30s": solver_configs["regular_30s"],
        "unit_time_60s": solver_configs["unit_time_60s"],
    }

    run_study(
        instances=instances,
        configs=configs,
        study_name="preemptive-test",
    )


def full_study():
    """Run full benchmark study."""
    print("=== Full Benchmark Study ===\n")

    run_study(
        instances=None,  # All instances
        configs=solver_configs,  # All solvers
        study_name=STUDY_NAME,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run preemptive RCPSP benchmark study")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "full"],
        default="quick",
        help="Run mode: quick test or full study",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=STUDY_NAME,
        help="Name of the study",
    )

    args = parser.parse_args()

    if args.mode == "quick":
        quick_test()
    else:
        full_study()
