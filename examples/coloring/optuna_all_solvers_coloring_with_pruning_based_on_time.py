#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to choose a solving method and tune its hyperparameters for coloring.

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""
import logging
import time
from collections import defaultdict
from typing import Any

import optuna
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import Trial, TrialState

from discrete_optimization.coloring.coloring_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.coloring.coloring_solvers import (
    ColoringASPSolver,
    ColoringLP,
    solvers_map,
    toulbar2_available,
)
from discrete_optimization.coloring.solvers.coloring_cp_solvers import ColoringCP
from discrete_optimization.coloring.solvers.coloring_cpsat_solver import (
    ColoringCPSatSolver,
)
from discrete_optimization.coloring.solvers.coloring_toulbar_solver import (
    ToulbarColoringSolver,
)
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.callbacks.optuna import OptunaCallback
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lp_tools import gurobi_available
from discrete_optimization.generic_tools.optuna.timed_percentile_pruner import (
    TimedPercentilePruner,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


seed = 42
optuna_nb_trials = 100
gurobi_full_license_available = False
create_another_study = True  # avoid relaunching the same study, keep the previous ones
max_time_per_solver = 20  # max duration (s)
min_time_per_solver = 5  # min duration before pruning (s)

modelfilename = "gc_70_9"

suffix = f"-{time.time()}" if create_another_study else ""
study_name = f"coloring_all_solvers-auto-pruning-{modelfilename}{suffix}"
storage_path = "./optuna-journal.log"  # NFS path for distributed optimization
elapsed_time_attr = "elapsed_time"  # name of the user attribute used to store duration of trials (updated during intermediate reports)

# Solvers to test and their associated kwargs
solvers_to_remove = {ColoringCP}
if not gurobi_available or not gurobi_full_license_available:
    solvers_to_remove.add(ColoringLP)
if not toulbar2_available:
    solvers_to_remove.add(ToulbarColoringSolver)
solvers_to_test: list[type[SolverDO]] = [
    s for s in solvers_map if s not in solvers_to_remove
]
# solvers_to_test = [ColoringLP]
p = ParametersCP.default_cpsat()
p.nb_process = 6
kwargs_fixed_by_solver: dict[type[SolverDO], dict[str, Any]] = defaultdict(
    dict,  # default kwargs for unspecified solvers
    {
        ColoringCPSatSolver: dict(parameters_cp=p, time_limit=max_time_per_solver),
        ColoringCP: dict(parameters_cp=p, time_limit=max_time_per_solver),
        ColoringLP: dict(time_limit=max_time_per_solver),
        ColoringASPSolver: dict(time_limit=max_time_per_solver),
        ToulbarColoringSolver: dict(time_limit=max_time_per_solver),
    },
)

# we need to map the classes to a unique string, to be seen as a categorical hyperparameter by optuna
# by default, we use the class name, but if there are identical names, f"{cls.__module__}.{cls.__name__}" could be used.
solvers_by_name: dict[str, type[SolverDO]] = {
    cls.__name__: cls for cls in solvers_to_test
}

# problem definition
file = [f for f in get_data_available() if "gc_70_9" in f][0]
problem = parse_file(file)

# sense of optimization
objective_register = problem.get_objective_register()
if objective_register.objective_sense == ModeOptim.MINIMIZATION:
    direction = "minimize"
else:
    direction = "maximize"

# objective names
objs, weights = objective_register.get_list_objective_and_default_weight()


# objective definition
def objective(trial: Trial):
    # hyperparameters to test

    # first parameter: solver choice
    solver_name: str = trial.suggest_categorical("solver", choices=solvers_by_name)
    solver_class = solvers_by_name[solver_name]

    # hyperparameters for the chosen solver
    suggested_hyperparameters_kwargs = solver_class.suggest_hyperparameters_with_optuna(
        trial=trial, prefix=solver_name + "."
    )

    # use existing value if corresponding to a previous complete trial
    states_to_consider = (TrialState.COMPLETE,)
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=states_to_consider
    )
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            logger.warning(
                "Trial with same hyperparameters as a previous complete trial: returning previous fit."
            )
            return t.value

    # prune if corresponding to a previous failed trial
    states_to_consider = (TrialState.FAIL,)
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=states_to_consider
    )
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            raise optuna.TrialPruned(
                "Pruning trial identical to a previous failed trial."
            )

    logger.info(f"Launching trial {trial.number} with parameters: {trial.params}")

    # construct kwargs for __init__, init_model, and solve
    kwargs = dict(kwargs_fixed_by_solver[solver_class])  # copy the frozen kwargs dict
    kwargs.update(suggested_hyperparameters_kwargs)

    # solver init
    solver = solver_class(problem=problem, **kwargs)
    solver.init_model(**kwargs)

    # init timer
    starting_time = time.perf_counter()

    # solve
    res = solver.solve(
        callbacks=[
            OptunaCallback(
                trial=trial,
                starting_time=starting_time,
                elapsed_time_attr=elapsed_time_attr,
                report_time=True,
            ),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
        ],
        **kwargs,
    )

    # store elapsed time
    elapsed_time = time.perf_counter() - starting_time
    trial.set_user_attr(elapsed_time_attr, elapsed_time)

    if len(res) != 0:
        _, fit = res.get_best_solution_fit()
        return fit
    else:
        raise optuna.TrialPruned("Pruned because failed")


# create study + database to store it
storage = JournalStorage(JournalFileStorage(storage_path))
study = optuna.create_study(
    study_name=study_name,
    direction=direction,
    sampler=optuna.samplers.TPESampler(seed=seed),
    pruner=TimedPercentilePruner(  # intermediate values interpolated at same "step"
        percentile=50,  # median pruner
        n_warmup_steps=min_time_per_solver,  # no pruning during first seconds
    ),
    storage=storage,
    load_if_exists=True,
)
study.set_metric_names(["nb_colors"])
study.optimize(objective, n_trials=optuna_nb_trials)
