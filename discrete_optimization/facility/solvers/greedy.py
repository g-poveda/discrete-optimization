#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers import FacilitySolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import ResultStorage


class GreedyFacilitySolver(FacilitySolver):
    """
    build a trivial solution
    pack the facilities one by one until all the customers are served
    """

    def solve(self, **kwargs: Any) -> ResultStorage:
        solution = [-1] * self.problem.customer_count
        capacity_remaining = [f.capacity for f in self.problem.facilities]
        facility_index = 0
        for index in range(len(self.problem.customers)):
            customer = self.problem.customers[index]
            if capacity_remaining[facility_index] >= customer.demand:
                solution[index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
            else:
                facility_index += 1
                assert capacity_remaining[facility_index] >= customer.demand
                solution[index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
        sol = FacilitySolution(problem=self.problem, facility_for_customers=solution)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )


class DistanceBasedGreedyFacilitySolver(FacilitySolver):
    def __init__(
        self,
        problem: FacilityProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.matrix_cost = np.zeros(
            (self.problem.customer_count, self.problem.facility_count)
        )
        for k in range(self.problem.customer_count):
            for j in range(self.problem.facility_count):
                self.matrix_cost[k, j] = self.problem.evaluate_customer_facility(
                    facility=self.problem.facilities[j],
                    customer=self.problem.customers[k],
                )
        self.min_distance: npt.NDArray[np.float64] = np.min(self.matrix_cost, axis=1)
        self.sorted_distance: npt.NDArray[np.int_] = np.argsort(
            self.matrix_cost, axis=1
        )
        self.sorted_customers: npt.NDArray[np.int_] = np.argsort(-self.min_distance)
        # sort the customers based on the minimum distance of facility (so the first element is the one which is the furthest to a facility
        self.available_demands = np.array(
            [
                self.problem.facilities[k].capacity
                for k in range(self.problem.facility_count)
            ]
        )

    def solve(self, **kwargs: Any) -> ResultStorage:
        solution = [-1] * self.problem.customer_count
        capacity_remaining = np.copy(self.available_demands)
        for customer in self.sorted_customers:
            prior_customers = kwargs.get("prio", {}).get(
                customer, self.sorted_distance[customer, :]
            )
            if any(x % 1 != x for x in prior_customers):
                prior_customers = self.sorted_distance[customer, :]
            for f in prior_customers:
                f = int(f)
                if capacity_remaining[f] >= self.problem.customers[customer].demand:
                    solution[customer] = f
                    capacity_remaining[f] -= self.problem.customers[customer].demand
                    break
        sol = FacilitySolution(problem=self.problem, facility_for_customers=solution)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )
