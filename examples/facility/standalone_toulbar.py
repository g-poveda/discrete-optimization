import pytoulbar2


def create_standalone_input_file():
    from discrete_optimization.facility.facility_model import FacilityProblem
    from discrete_optimization.facility.facility_parser import (
        get_data_available,
        parse_file,
    )
    from discrete_optimization.facility.solvers.facility_lp_solver import (
        prune_search_space,
    )

    file = [f for f in get_data_available() if "fl_16_1" in f][0]
    model: FacilityProblem = parse_file(file)
    nb_facilities = model.facility_count
    nb_customers = model.customer_count
    matrix_fc_indicator, matrix_length = prune_search_space(
        model, n_cheapest=nb_facilities, n_shortest=nb_facilities
    )
    setup_cost = [model.facilities[i].setup_cost for i in range(nb_facilities)]
    demand = [model.customers[i].demand for i in range(nb_customers)]
    capacity = [model.facilities[i].capacity for i in range(nb_facilities)]
    matrix_length = [
        [int(matrix_length[i, j]) for j in range(matrix_length.shape[1])]
        for i in range(matrix_length.shape[0])
    ]
    d = {
        "nb_facilities": nb_facilities,
        "nb_customers": nb_customers,
        "matrix_length": matrix_length,
        "setup_cost": setup_cost,
        "demand": demand,
        "capacity": capacity,
    }
    import json

    json.dump(d, open("data_facility.json", "w"), indent=2)


def run_model_integer_variable():
    import json

    input_file = json.load(open("data_facility.json", "r"))
    nb_facilities = input_file["nb_facilities"]
    nb_customers = input_file["nb_customers"]
    Problem = pytoulbar2.CFN(10e7)
    index = 0
    matrix_length = input_file["matrix_length"]
    setup_cost = input_file["setup_cost"]
    demand = input_file["demand"]
    capacity = input_file["capacity"]
    client_var_to_index = {}
    for c in range(nb_customers):
        Problem.AddVariable(f"x_{c}", values=range(nb_facilities))
        Problem.AddFunction(
            [f"x_{c}"], [int(matrix_length[f][c]) for f in range(nb_facilities)]
        )
        client_var_to_index[c] = index
        index += 1
    index_var = index
    used_var_to_index = {}
    for i in range(nb_facilities):
        Problem.AddVariable(f"used_{i}", [0, 1])
        used_var_to_index[i] = index_var
        Problem.AddFunction([f"used_{i}"], [0, setup_cost[i]])
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
            [(f"x_{c}", i, int(demand[c])) for c in range(nb_customers)],
            "<=",
            int(capacity[i]),
        )
        # Problem.CFN.wcsp.postKnapsackConstraint([client_var_to_index[c] for c in range(nb_customers)],
        #                                         str(int(-self.problem.facilities[i].capacity))+params_constraints,
        #                                         False, True)
    sol = Problem.Solve(showSolutions=1)
    print(sol)


if __name__ == "__main__":
    # create_standalone_input_file()
    run_model_integer_variable()
