#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#
# Toulbar2 model is an adaptation of :
# https://forgemia.inra.fr/thomas.schiex/cost-function-library/-/blob/master/crafted/rcpsp/scripts/rcpsp.py
# Thanks to INRAE team for the help.
from typing import Optional

import pytoulbar2

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution


class ToulbarRCPSPSolver(SolverDO):
    def __init__(
        self,
        problem: RCPSPModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )
        self.model: pytoulbar2.CFN = None

    def init_model(self, **args):
        N = self.problem.n_jobs
        horizon = args.get(
            "max_time", self.problem.horizon
        )  # give a better initial upper-bound if known
        Problem = pytoulbar2.CFN(horizon + 1)
        for i in range(N):
            Problem.AddVariable("x" + str(i + 1), range(horizon + 1))

        # first job starts at 0
        Problem.AddFunction(
            [self.problem.index_task[self.problem.source_task]],
            [0 if a == 0 else horizon for a in range(horizon + 1)],
        )

        # precedence constraints
        job_durations = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        for i in range(N):
            task = self.problem.tasks_list[i]
            for task_suc in self.problem.successors[task]:
                index_suc = self.problem.index_task[task_suc]
                Problem.AddFunction(
                    [i, index_suc],
                    [
                        (0 if a + job_durations[i] <= b else horizon)
                        for a in range(horizon + 1)
                        for b in range(horizon + 1)
                    ],
                )

        # for each ressource and each time slot,
        # we post a linear constraint on all the jobs that require this ressource to not overcome the ressoure capacity
        resources_list = self.problem.resources_list
        max_capacity = {
            r: self.problem.get_max_resource_capacity(r) for r in resources_list
        }
        for resource in max_capacity:
            for a in range(horizon + 1):
                params = ""
                realscope = []
                for i in range(N):
                    task = self.problem.tasks_list[i]
                    req = self.problem.mode_details[task][1].get(resource, 0)
                    if req > 0:
                        paramval = ""
                        nbval = 0
                        for b in range(horizon + 1):
                            if b <= a < b + job_durations[i]:
                                nbval += 1
                                paramval += " " + str(b) + " " + str(-req)
                        if nbval > 0:
                            params += " " + str(nbval) + paramval
                            realscope.append(i)
                if len(params) > 0:
                    Problem.CFN.wcsp.postKnapsackConstraint(
                        realscope, str(-max_capacity[resource]) + params, False, True
                    )

        # minimize makespan, i.e., the completion time of the last job
        Problem.AddFunction([N - 1], [a for a in range(horizon + 1)])
        self.model = Problem

    def solve(self, **kwargs) -> ResultStorage:
        if self.model is None:
            self.init_model()
        solution = self.model.Solve(showSolutions=1)
        rcpsp_sol = RCPSPSolution(
            problem=self.problem,
            rcpsp_schedule={
                self.problem.tasks_list[i]: {
                    "start_time": solution[0][i],
                    "end_time": solution[0][i]
                    + self.problem.mode_details[self.problem.tasks_list[i]][1][
                        "duration"
                    ],
                }
                for i in range(self.problem.n_jobs)
            },
        )
        fit = self.aggreg_sol(rcpsp_sol)
        return ResultStorage(
            list_solution_fits=[(rcpsp_sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )
