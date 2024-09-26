#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Use of Aries solver https://github.com/plaans/aries

import logging
import os
import re
import subprocess
import time
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.do_solver import ResultStorage, SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.jsp.job_shop_problem import JobShopProblem, SolutionJobshop

aries_path = os.environ.get("ARIES_SCHEDULING_PATH", None)
logger = logging.getLogger(__name__)


class AriesJspSolver(SolverDO):
    problem: JobShopProblem
    hyperparameters = [
        CategoricalHyperparameter(
            name="search",
            choices=[
                "activity",
                # "est", "parallel", TODO doesn't work
                "learning-rate",
            ],
            default="learning-rate",
        )
    ]

    def __init__(self, problem: Problem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.file_path = None

    def init_model(self, **args):
        now = time.time()
        file_path = f"tmp_file_{now}.jsp"
        with open(file_path, "w") as f:
            lines = ["nb_jobs nb_machines\n"]
            lines.append(f"{self.problem.n_jobs} {self.problem.n_machines} 0 0 0 0\n")
            lines.append("Times\n")
            for job in self.problem.list_jobs:
                lines.append(
                    " ".join(str(subjob.processing_time) for subjob in job) + "\n"
                )
            lines.append("Machines\n")
            for job in self.problem.list_jobs:
                lines.append(
                    " ".join(str(subjob.machine_id + 1) for subjob in job) + "\n"
                )
            f.writelines(lines)
        self.file_path = file_path

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: int = 20,
        **kwargs: Any,
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if self.file_path is None:
            self.init_model(**kwargs)
        if aries_path is None:
            logger.warning(
                'The environment variable "ARIES_SCHEDULING_PATH" is not defines, '
                "you need to define it as the path to the scheduling executable"
            )
            raise Exception("Need to define ARIES_SCHEDULING_PATH environment variable")
        now = time.time()
        output_tmp_file = f"res_{now}.txt"
        command = (
            aries_path + f" jobshop {self.file_path} --search {kwargs['search']}"
            f" --output {output_tmp_file} --timeout {time_limit}"
        )
        logger.info("Launching command line of aries solver")
        try:
            result = subprocess.run(
                command, shell=True, check=True, timeout=time_limit + 50
            )
            with open(output_tmp_file, "r") as file:
                text = file.read()
                # Split the text by lines
                lines = text.splitlines()
                rebuild_schedule = {}
                for line in lines:
                    # Split each line by ':' to separate the machine ID from the tuples
                    numbers = re.findall(r"\d+", line)
                    # Convert the extracted numbers to integers
                    numbers = list(map(int, numbers))
                    # machine_id = numbers[0]
                    for i in range(0, len(numbers), 3):
                        index_job = numbers[i]
                        index_subjob = numbers[i + 1]
                        start = numbers[i + 2]
                        end = (
                            start
                            + self.problem.list_jobs[index_job][
                                index_subjob
                            ].processing_time
                        )
                        rebuild_schedule[index_job, index_subjob] = (start, end)

            sol = SolutionJobshop(
                problem=self.problem,
                schedule=[
                    [
                        rebuild_schedule[(i, j)]
                        for j in range(len(self.problem.list_jobs[i]))
                    ]
                    for i in range(len(self.problem.list_jobs))
                ],
            )
            fit = self.aggreg_from_sol(sol)
            os.remove(output_tmp_file)
            os.remove(self.file_path)
            return self.create_result_storage([(sol, fit)])
        except Exception as e:
            logger.error(f"Exception raised {e}")
            raise e
