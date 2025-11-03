#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import math
from enum import Enum
from typing import Any, Optional, Union

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    IntVar,
    LinearExpr,
    LinearExprT,
)

from discrete_optimization.binpack.problem import (
    BinPack,
    BinPackProblem,
    BinPackSolution,
    Item,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    AllocationCpSatSolver,
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)

logger = logging.getLogger(__name__)


class ModelingBinPack(Enum):
    BINARY = 0
    SCHEDULING = 1


class ModelingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message


class CpSatBinPackSolver(
    AllocationCpSatSolver[Item, BinPack], SchedulingCpSatSolver[Item], WarmstartMixin
):
    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingBinPack, default=ModelingBinPack.BINARY
        )
    ]
    problem: BinPackProblem
    modeling: ModelingBinPack

    def __init__(
        self,
        problem: BinPackSolution,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables: dict[
            str, Union[IntVar, list[IntVar], list[dict[int, IntVar]]]
        ] = {}

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        self.modeling = args["modeling"]
        if args["modeling"] == ModelingBinPack.BINARY:
            self.init_model_binary(**args)
        if args["modeling"] == ModelingBinPack.SCHEDULING:
            self.init_model_scheduling(**args)

    def init_model_binary(self, **args: Any):
        super().init_model(**args)
        variables_allocation = {}
        used_bin = {}
        upper_bound = args.get("upper_bound", self.problem.nb_items)
        for bin_ in range(upper_bound):
            used_bin[bin_] = self.cp_model.NewBoolVar(f"used_{bin_}")
            if bin_ >= 1:
                self.cp_model.Add(used_bin[bin_] <= used_bin[bin_ - 1])
        self.used_variables_created = True
        self.used_variables = used_bin
        for i in range(self.problem.nb_items):
            for bin_ in range(upper_bound):
                variables_allocation[(i, bin_)] = self.cp_model.NewBoolVar(
                    f"alloc_{i}_{bin_}"
                )
                # self.cp_model.Add(used_bin[bin_] >= variables_allocation[(i, bin_)])
            self.cp_model.AddExactlyOne(
                [variables_allocation[(i, bin_)] for bin_ in range(upper_bound)]
            )
        for bin_ in used_bin:
            self.cp_model.add_max_equality(
                used_bin[bin_],
                [variables_allocation[(i, bin_)] for i in range(self.problem.nb_items)],
            )
        if self.problem.has_constraint:
            for i, j in self.problem.incompatible_items:
                for bin_ in range(upper_bound):
                    self.cp_model.AddForbiddenAssignments(
                        [
                            variables_allocation[(i, bin_)],
                            variables_allocation[(j, bin_)],
                        ],
                        [(1, 1)],
                    )
        for bin_ in range(upper_bound):
            self.cp_model.Add(
                LinearExpr.weighted_sum(
                    [
                        variables_allocation[(i, bin_)]
                        for i in range(self.problem.nb_items)
                    ],
                    [
                        self.problem.list_items[i].weight
                        for i in range(self.problem.nb_items)
                    ],
                )
                <= self.problem.capacity_bin
            )
        self.variables["allocation"] = variables_allocation
        self.variables["used"] = used_bin
        self.cp_model.Minimize(self.get_nb_unary_resources_used_variable())

    def init_model_scheduling(self, **args: Any):
        super().init_model(**args)
        upper_bound = args.get("upper_bound", self.problem.nb_items)
        starts = {}
        intervals = {}
        for i in range(self.problem.nb_items):
            starts[i] = self.cp_model.NewIntVar(lb=0, ub=upper_bound, name=f"bin_{i}")
            intervals[i] = self.cp_model.NewFixedSizeIntervalVar(
                start=starts[i], size=1, name=f"interval_{i}"
            )
        self.cp_model.AddCumulative(
            [intervals[i] for i in range(self.problem.nb_items)],
            [self.problem.list_items[i].weight for i in range(self.problem.nb_items)],
            self.problem.capacity_bin,
        )
        if self.problem.has_constraint:
            for i, j in self.problem.incompatible_items:
                self.cp_model.Add(starts[i] != starts[j])
        self.variables["starts"] = starts
        makespan = self.cp_model.NewIntVar(lb=1, ub=upper_bound + 1, name="nb_bins")
        self.variables["makespan"] = makespan
        self.cp_model.add_max_equality(makespan, [starts[i] + 1 for i in starts])
        self.cp_model.minimize(makespan)

    def get_task_unary_resource_is_present_variable(
        self, task: Item, unary_resource: BinPack
    ) -> LinearExprT:
        if self.modeling == ModelingBinPack.BINARY:
            return self.variables["allocation"][(task, unary_resource)]
        raise ModelingError(f"No allocation variable with {self.modeling}")

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        if self.modeling == ModelingBinPack.SCHEDULING:
            if start_or_end == StartOrEnd.START:
                return self.variables["starts"][task]
            else:
                return self.variables["ends"][task]
        raise ModelingError(f"No start or end variable with {self.modeling}")

    def set_warm_start(self, solution: BinPackSolution) -> None:
        if self.modeling == ModelingBinPack.SCHEDULING:
            self.cp_model.ClearHints()
            for i in range(self.problem.nb_items):
                self.cp_model.AddHint(
                    self.variables["starts"][i], solution.allocation[i]
                )
            self.cp_model.AddHint(
                self.variables["makespan"], max(solution.allocation) + 1
            )
        if self.modeling == ModelingBinPack.BINARY:
            self.cp_model.ClearHints()
            for i, bin_ in self.variables["allocation"]:
                if solution.allocation[i] == bin_:
                    self.cp_model.AddHint(self.variables["allocation"][(i, bin_)], 1)
                else:
                    self.cp_model.AddHint(self.variables["allocation"][(i, bin_)], 0)
            bins = set(solution.allocation)
            for bin_ in self.variables["used"]:
                if bin_ in bins:
                    self.cp_model.AddHint(self.variables["used"][bin_], 1)
                else:
                    self.cp_model.AddHint(self.variables["used"][bin_], 0)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        logger.info(
            f"Obj={cpsolvercb.objective_value}, Bound={cpsolvercb.best_objective_bound}"
        )
        allocation = [None for i in range(self.problem.nb_items)]
        if self.modeling == ModelingBinPack.BINARY:
            for i, j in self.variables["allocation"]:
                if cpsolvercb.Value(self.variables["allocation"][(i, j)]) == 1:
                    allocation[i] = j
        if self.modeling == ModelingBinPack.SCHEDULING:
            for i in self.variables["starts"]:
                allocation[i] = cpsolvercb.Value(self.variables["starts"][i])
        return BinPackSolution(problem=self.problem, allocation=allocation)
