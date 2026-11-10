#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os

from discrete_optimization.generic_tools.asp_tools import AspClingoSolver, clingo
from discrete_optimization.rcpsp.problem import RcpspSolution
from discrete_optimization.rcpsp.solvers import RcpspSolver

this_folder = os.path.dirname(os.path.abspath(__file__))
modeling_folder = os.path.join(this_folder, "modeling_asp")
file_path = os.path.join(modeling_folder, "rcpsp.lp")
from typing import Any


class AspRcpspSolver(AspClingoSolver, RcpspSolver):
    """Solver based on Answer Set Programming formulation and clingo solver."""

    def init_model(self, **kwargs: Any) -> None:
        basic_model = """"""
        with open(file_path, "r") as f:
            basic_model = f.read()
        max_models = kwargs.get("max_models", 1)
        nb_process = kwargs.get("nb_process", None)

        command = ["--warn=no-atom-undefined", "--opt-mode=optN"]
        if nb_process is not None and nb_process > 1:
            command.append("--parallel-mode=8")
        self.ctl = clingo.Control(command)
        self.ctl.add("base", [], basic_model)
        string_data_input = self.build_string_data_input()
        self.ctl.add("base", [], string_data_input)

    def build_string_data_input(self) -> str:
        """Build ASP facts from RCPSP problem data.

        Generates ASP facts for:
        - Resource capacities: res_capacity(R, C)
        - Task durations: dur(T, D)
        - Precedence constraints: prec(T1, T2)
        - Resource requirements: res(T, R, X)

        Note: Only supports single-mode RCPSP (uses mode 1 for all tasks).
        """
        facts = []

        def format_identifier(identifier):
            """Format identifier for ASP (quote strings, keep numbers as-is)."""
            if isinstance(identifier, int):
                return str(identifier)
            else:
                # Quote string identifiers to handle uppercase and special chars
                return f'"{identifier}"'

        # Resource capacities
        for resource in self.problem.resources_list:
            cap_value = self.problem.get_max_resource_capacity(resource)
            res_id = format_identifier(resource)
            facts.append(f"res_capacity({res_id}, {cap_value}).")

        # Task durations and resource requirements (mode 1 only)
        for task in self.problem.tasks_list:
            # Get mode 1 details (assuming single-mode RCPSP)
            modes = sorted(self.problem.mode_details[task].keys())
            mode = modes[0] if modes else 1
            mode_details = self.problem.mode_details[task][mode]

            duration = mode_details.get("duration", 0)
            task_id = self.problem.get_index_from_task(task)
            facts.append(f"dur({task_id}, {duration}).")

            # Resource requirements for this task
            for resource in self.problem.resources_list:
                consumption = mode_details.get(resource, 0)
                if consumption > 0:
                    res_id = format_identifier(resource)
                    facts.append(f"res({task_id}, {res_id}, {consumption}).")

        # Precedence constraints
        for task, successors in self.problem.successors.items():
            task_id = self.problem.get_index_from_task(task)
            for succ in successors:
                succ_id = self.problem.get_index_from_task(succ)
                facts.append(f"prec({task_id}, {succ_id}).")

        return "\n".join(facts)

    def retrieve_solution(self, model: clingo.Model) -> RcpspSolution:
        """Extract RCPSP solution from ASP model.

        Parses 'done((T,D), I)' symbols to build the schedule dictionary.
        Handles both integer and string task IDs.
        """
        # symbols = model.symbols(atoms=True)
        symbols = model.symbols(shown=True)
        schedule = {}

        for symbol in symbols:
            if symbol.name == "done":
                # symbol.arguments[0] is a tuple (task, duration)
                # symbol.arguments[1] is the completion time
                task_tuple = symbol.arguments[0]
                task_index = int(task_tuple.arguments[0].number)
                task_id = self.problem.get_task_from_index(task_index)
                completion_time = symbol.arguments[1].number
                # Get duration from problem (more reliable than from symbol)
                modes = sorted(self.problem.mode_details[task_id].keys())
                mode = modes[0] if modes else 1
                duration = self.problem.mode_details[task_id][mode]["duration"]

                start_time = completion_time - duration
                schedule[task_id] = {
                    "start_time": start_time,
                    "end_time": completion_time,
                }
        if len(schedule) > 0:
            # Build mode dictionary (all mode 1 for single-mode RCPSP)
            modes = [1] * self.problem.n_jobs_non_dummy
            schedule[self.problem.source_task] = {"start_time": 0, "end_time": 0}
            max_time = max([schedule[x]["end_time"] for x in schedule])
            schedule[self.problem.sink_task] = {
                "start_time": max_time,
                "end_time": max_time,
            }
            sol = RcpspSolution(
                problem=self.problem,
                rcpsp_schedule=schedule,
                rcpsp_modes=modes,
            )
            # print(self.problem.evaluate(sol), self.problem.satisfy(sol))
            return sol
        return None
