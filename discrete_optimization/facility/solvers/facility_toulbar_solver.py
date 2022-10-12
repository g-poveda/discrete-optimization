from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import pytoulbar2

from discrete_optimization.facility.facility_model import FacilityProblem
from discrete_optimization.facility.solvers.facility_lp_solver import prune_search_space
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class FacilityTBModelType(Enum):
    INTEGER = 0
    BOOLEAN = 1


class ToulbarFacilitySolve(SolverDO):
    def __init__(
        self,
        problem: FacilityProblem,
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

    def init_model(
        self, model_type: FacilityTBModelType = FacilityTBModelType.INTEGER, **kwargs
    ):
        if model_type == FacilityTBModelType.INTEGER:
            self.init_model_integer_variable(**kwargs)
        if model_type == FacilityTBModelType.BOOLEAN:
            self.init_model_boolean_variable(**kwargs)

    def init_model_boolean_variable(self, **kwargs):
        """ """
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        Problem = pytoulbar2.CFN(kwargs.get("upper_bound", 10e8))
        x: Dict[Tuple[int, int], Union[int, Any]] = {}
        key_to_index = {}
        index = 0
        # n_shortest = kwargs.get("n_shortest", 10)
        # n_cheapest = kwargs.get("n_cheapest", 10)
        matrix_fc_indicator, matrix_length = prune_search_space(
            self.problem, n_cheapest=nb_facilities, n_shortest=nb_facilities
        )
        for f in range(nb_facilities):
            for c in range(nb_customers):
                x[f, c] = Problem.AddVariable(name=f"x_{(f,c)}", values=[0, 1])
                Problem.AddFunction([f"x_{(f,c)}"], [0, matrix_length[f, c]])
                key_to_index[(f, c)] = index
                index += 1
        for c in range(nb_customers):
            param_c = " ".join(
                [str(1) + " " + str(1) + " " + str(-1) for f in range(nb_facilities)]
            )
            print(param_c)
            Problem.CFN.wcsp.postKnapsackConstraint(
                [key_to_index[(f, c)] for f in range(nb_facilities)],
                str(-int(1)) + " " + param_c,
                False,
                True,
            )
            param_c = " ".join(
                [str(1) + " " + str(1) + " " + str(1) for f in range(nb_facilities)]
            )
            Problem.CFN.wcsp.postKnapsackConstraint(
                [key_to_index[(f, c)] for f in range(nb_facilities)],
                str(int(1)) + " " + param_c,
                False,
                True,
            )
        used_var = {}
        index_var = index
        for i in range(nb_facilities):
            used_var[i] = Problem.AddVariable(f"used_{i}", [0, 1])
            index_var += 1
            Problem.AddFunction(
                [f"used_{i}"], [0, self.problem.facilities[i].setup_cost]
            )
            for c in range(nb_customers):
                Problem.CFN.wcsp.postKnapsackConstraint(
                    [index_var - 1, key_to_index[(i, c)]], "0 1 1 1 1 1 -1", False, True
                )

                Problem.AddLinearConstraint(
                    [1, -1], [f"used_{i}", f"x_{i, c}"], operand=">=", rightcoef=0
                )
                # constraint triggering the used facility binary variable
        for i in range(nb_facilities):
            param_c = ""
            for c in range(nb_customers):
                param_c += (
                    " "
                    + str(1)
                    + " "
                    + str(1)
                    + " "
                    + str(-int(self.problem.customers[c].demand))
                )
            Problem.CFN.wcsp.postKnapsackConstraint(
                [key_to_index[(i, c)] for c in range(nb_customers)],
                str(-self.problem.facilities[i].capacity) + param_c,
                False,
                True,
            )
            # Capacity/Knapsack constraint on facility.
        self.model = Problem

    def init_model_integer_variable(self, **kwargs):
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        Problem = pytoulbar2.CFN(kwargs.get("upper_bound", 10e8))
        index = 0
        matrix_fc_indicator, matrix_length = prune_search_space(
            self.problem, n_cheapest=nb_facilities, n_shortest=nb_facilities
        )
        client_var_to_index = {}
        for c in range(nb_customers):
            Problem.AddVariable(f"x_{c}", values=range(nb_facilities))
            Problem.AddFunction(
                [f"x_{c}"], [int(matrix_length[f, c]) for f in range(nb_facilities)]
            )
            client_var_to_index[c] = index
            index += 1
        index_var = index
        used_var_to_index = {}
        for i in range(nb_facilities):
            Problem.AddVariable(f"used_{i}", [0, 1])
            used_var_to_index[i] = index_var
            Problem.AddFunction(
                [f"used_{i}"], [0, self.problem.facilities[i].setup_cost]
            )
            for c in range(nb_customers):
                Problem.AddFunction(
                    [f"used_{i}", f"x_{c}"],
                    [
                        10e8 if b == 0 and f == i else 0
                        for b in [0, 1]
                        for f in range(nb_facilities)
                    ],
                )
            index_var += 1
            # Somehow force that when x_{c} == i, used_i = 1
        # capacity constraint on facility.
        for i in range(nb_facilities):
            # params_constraints = ""
            # for c in range(nb_customers):
            #     params_constraints += " "+str(1)+" "+str(i)+" "+str(-int(self.problem.customers[c].demand))
            Problem.AddGeneralizedLinearConstraint(
                [
                    (f"x_{c}", i, int(self.problem.customers[c].demand))
                    for c in range(nb_customers)
                ],
                "<=",
                int(self.problem.facilities[i].capacity),
            )
            # Problem.CFN.wcsp.postKnapsackConstraint([client_var_to_index[c] for c in range(nb_customers)],
            #                                         str(int(-self.problem.facilities[i].capacity))+params_constraints,
            #                                         False, True)
        self.model = Problem

    def solve(self, **kwargs) -> ResultStorage:
        if self.model is None:
            self.init_model(**kwargs)
        solution = self.model.Solve(showSolutions=1)
        return solution
