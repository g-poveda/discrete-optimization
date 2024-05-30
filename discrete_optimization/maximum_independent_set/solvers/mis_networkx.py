from typing import Any, List, Optional

import networkx as nx

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.mis_model import MisSolution
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver


class MisNetworkXSolver(MisSolver):
    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        sol = nx.approximation.maximum_independent_set(self.problem.graph_nx)
        chosen = [
            1 if self.problem.nodes[i] in sol else 0
            for i in range(self.problem.number_nodes)
        ]
        solution = MisSolution(problem=self.problem, chosen=chosen)
        fit = self.aggreg_from_sol(solution)
        return ResultStorage(
            [(solution, fit)], mode_optim=self.params_objective_function.sense_function
        )
