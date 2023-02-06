#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

import discrete_optimization.tsp.tsp_parser as tsp_parser
import discrete_optimization.vrp.vrp_parser as vrp_parser
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.pickup_vrp.builders.instance_builders import (
    create_selective_tsp,
)
from discrete_optimization.pickup_vrp.gpdp import (
    GPDPSolution,
    ProxyClass,
    build_pruned_problem,
)
from discrete_optimization.pickup_vrp.solver.lp_solver import (
    LinearFlowSolver,
    ParametersMilp,
    plot_solution,
)

try:
    import gurobipy as grb
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_tsp_new_api():
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    res = linear_flow_solver.solve(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    assert isinstance(res, ResultStorage)
    sol = res.get_best_solution()
    assert isinstance(sol, GPDPSolution)
    assert len(sol.times) == 0


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_tsp_new_api_with_time():
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=True
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    res = linear_flow_solver.solve(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    assert isinstance(res, ResultStorage)
    sol = res.get_best_solution()
    assert isinstance(sol, GPDPSolution)
    assert len(sol.times) == 6


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_tsp():
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_tsp_simplified():
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    gpdp = build_pruned_problem(gpdp, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_vrp():
    files_available = vrp_parser.get_data_available()
    file_path = [f for f in files_available if "vrp_16_3_1" in f][0]
    vrp_model = vrp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_vrp_model_to_gpdp(vrp_model=vrp_model, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


@pytest.mark.skip(reason="build_pruned_problem() is buggy for now.")
@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_vrp_simplified():
    files_available = vrp_parser.get_data_available()
    file_path = [f for f in files_available if "vrp_16_3_1" in f][0]
    vrp_model = vrp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_vrp_model_to_gpdp(vrp_model=vrp_model, compute_graph=True)
    gpdp = build_pruned_problem(gpdp, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()
    p.time_limit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_selective_tsp():
    gpdp = create_selective_tsp(nb_nodes=20, nb_vehicles=1, nb_clusters=4)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=False,
    )
    p = ParametersMilp.default()
    p.time_limit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_selective_vrp():
    gpdp = create_selective_tsp(nb_nodes=20, nb_vehicles=3, nb_clusters=4)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=False,
    )
    p = ParametersMilp.default()
    p.time_limit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_selective_vrp_with_time():
    nb_nodes = 10
    nb_vehicles = 2
    nb_clusters = 2
    gpdp = create_selective_tsp(
        nb_nodes=nb_nodes, nb_vehicles=nb_vehicles, nb_clusters=nb_clusters
    )
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=True,
    )
    p = ParametersMilp.default()
    p.time_limit = 100
    res = linear_flow_solver.solve(
        parameters_milp=p,
        do_lns=False,
        nb_iteration_max=20,
    )
    assert isinstance(res, ResultStorage)
    sol = res.get_best_solution()
    assert isinstance(sol, GPDPSolution)
    print(sol.times)


if __name__ == "__main__":
    test_vrp()
