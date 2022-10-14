#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from minizinc import Status

from discrete_optimization.generic_rcpsp_tools.large_neighborhood_search_scheduling import (
    LargeNeighborhoodSearchScheduling,
)

# from tests_clean.rcpsp.solvers.lns_solver import mix_lot, allocation_handler, mix_two_constraint_handler
from discrete_optimization.generic_rcpsp_tools.neighbor_builder import (
    ObjectiveSubproblem,
    mix_lot,
)
from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import (
    ParamsConstraintBuilder,
)
from discrete_optimization.generic_tools.lns_mip import TrivialInitialSolution
from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
    plot_resource_individual_gantt,
    plt,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solver_mspsp_instlib import (
    CP_MSPSP_MZN,
    CPSolverName,
    ParametersCP,
)
from discrete_optimization.rcpsp_multiskill.solvers.lns_post_process_rcpsp import (
    PostProMSRCPSPPreemptive,
)
from examples.mslib.parse_mslib import get_data_available, parse_file_mslib

strategy = [
    "default_s",
    "start_s",
    "overlap_s",
    "assign_s",
    "contrib_s",
    "assign_Then_start",
    "contrib_Then_start",
    "start_Then_assign",
    "start_Then_contrib",
    "overlap_Then_assign_Then_start",
    "overlap_Then_contrib_Then_start",
    "priority_input_order",
    "priority_smallest",
    "priority_smallest_load",
    "priority_smallest_largest",
    "priority_first_fail",
]


def run_lns(file=None, time_limit=1000):
    if file is None:
        file = (
            "/Users/poveda_g/Documents/git_repos/MSPSP-InstLib/instances/set-1/"
            "set-1b/inst_set1b_sf1_nc2.1_n40_m50_04.dzn"
        )
    model = parse_file_mslib(file_path=file)
    model = model.to_variant_model()
    # init_solution = model.get_dummy_solution(preemptive=False)
    # ls_solver = LS_RCPSP_Solver(model=model,
    #                            ls_solver=LS_SOLVER.SA)
    # result_store = ls_solver.solve(nb_iteration_max=10000,
    #                               temperature=1000,
    #                               init_solution_process=False,
    #                               starting_point=init_solution)

    cp_solver_1 = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.CHUFFED)
    cp_solver_1.init_model(
        output_type=True, ignore_sec_objective=True, add_objective_makespan=True
    )
    cp_solver_1.instance["maxt"] = 150
    cp_solver_1.instance["full_output"] = True
    cp_solver_1.instance.add_string("my_search=priority_smallest;\n")
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 35
    parameters_cp.TimeLimit_iter0 = 25
    parameters_cp.multiprocess = False
    parameters_cp.nb_process = 4
    results = cp_solver_1.solve(parameters_cp=parameters_cp)
    status = cp_solver_1.result_status
    from minizinc import Status

    if status == Status.OPTIMAL_SOLUTION:
        print("Optimal solution..")
        return results

    from discrete_optimization.generic_tools.lns_mip import TrivialInitialSolution

    if len(results.list_solution_fits) > 0:
        initial_solution_provider = TrivialInitialSolution(results)
    else:
        initial_solution_provider = None

    cp_solver = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.CHUFFED)
    cp_solver.init_model(
        output_type=True, ignore_sec_objective=False, add_objective_makespan=False
    )
    cp_solver.instance["maxt"] = 150
    cp_solver.instance["full_output"] = True
    cp_solver.instance.add_string("my_search=priority_smallest;\n")
    constraint_handler = mix_lot(
        model,
        nb_cut_parts=[3, 4, 5, 6],  # [3, 4, 5, 6], #[5, 6, 7],
        fraction_subproblems=[
            0.2,
            0.3,
            0.4,
            0.5,
        ],  # [0.1, 0.2, 0.3, 0.4, 0.5], #list(np.arange(0.05, 0.2, 0.15)),
        params_list=[
            ParamsConstraintBuilder(
                minus_delta_primary=80,
                plus_delta_primary=80,
                minus_delta_secondary=1,
                plus_delta_secondary=1,
                constraint_max_time_to_current_solution=True,
                fraction_of_task_assigned_multiskill=0.45,
                except_assigned_multiskill_primary_set=False,
                first_method_multiskill=True,
                second_method_multiskill=False,
                additional_methods=False,
            )
        ],
        objective_subproblem=ObjectiveSubproblem.SUM_END_SUBTASKS,
        time_windows=False,
        equilibrate_multiskill_v2=False,
        time_window_length=10,
    )
    lns = LargeNeighborhoodSearchScheduling(
        rcpsp_problem=model,
        cp_solver=cp_solver,
        constraint_handler=constraint_handler,
        partial_solution=None,
        post_process_solution=PostProMSRCPSPPreemptive(problem=model),
        initial_solution_provider=initial_solution_provider,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 35
    parameters_cp.TimeLimit_iter0 = 25
    parameters_cp.multiprocess = False
    parameters_cp.nb_process = 4
    parameters_cp.free_search = False
    result_store = lns.solve(
        parameters_cp=parameters_cp,
        nb_iteration_lns=1000,
        nb_iteration_no_improvement=500,
        max_time_seconds=time_limit,
        skip_first_iteration=False,
        stop_first_iteration_if_optimal=False,
    )
    return result_store


def run_lns_cpopt(
    file=None,
    time_limit=1000,
    skill_level_version=True,
    solver_initial_solution: CPSolverName = CPSolverName.CPOPT,
    solver_lns: CPSolverName = CPSolverName.ORTOOLS,
):
    if file is None:
        file = (
            "/Users/poveda_g/Documents/git_repos/MSPSP-InstLib/instances/set-1/"
            "set-1b/inst_set1b_sf1_nc2.1_n40_m50_04.dzn"
        )
    model = parse_file_mslib(file_path=file, skill_level_version=skill_level_version)
    model = model.to_variant_model()
    initial_solution_provider = None
    if False:
        cp_solver_0 = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.ORTOOLS)
        cp_solver_0.init_model(
            output_type=True,
            model_type="mspsp_compatible",
            ignore_sec_objective=True,
            add_objective_makespan=True,
        )
        cp_solver_0.instance["maxt"] = 150
        cp_solver_0.instance["full_output"] = True
        cp_solver_0.instance.add_string("my_search=assign_Then_start;\n")
        parameters_cp = ParametersCP.default()
        parameters_cp.TimeLimit = 35
        parameters_cp.TimeLimit_iter0 = 25
        parameters_cp.multiprocess = False
        parameters_cp.nb_process = 4
        parameters_cp.free_search = False
        results = cp_solver_0.solve(parameters_cp=parameters_cp)
        status = cp_solver_0.result_status
        if status == Status.OPTIMAL_SOLUTION:
            print("Optimal solution..")
            return results
        if len(results.list_solution_fits) > 0:
            initial_solution_provider = TrivialInitialSolution(results)
        else:
            initial_solution_provider = None
    if False and initial_solution_provider is None:
        cp_solver_1 = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.CHUFFED)
        cp_solver_1.init_model(
            output_type=True,
            model_type="mspsp",
            ignore_sec_objective=True,
            add_objective_makespan=True,
        )
        cp_solver_1.instance["maxt"] = 200
        cp_solver_1.instance["full_output"] = True
        cp_solver_1.instance.add_string("my_search=priority_smallest;\n")
        parameters_cp = ParametersCP.default()
        parameters_cp.TimeLimit = 35
        parameters_cp.TimeLimit_iter0 = 25
        parameters_cp.multiprocess = False
        parameters_cp.nb_process = 4
        parameters_cp.free_search = True
        results = cp_solver_1.solve(parameters_cp=parameters_cp)
        status = cp_solver_1.result_status
        if status == Status.OPTIMAL_SOLUTION:
            print("Optimal solution..")
            return results
        if len(results.list_solution_fits) > 0:
            initial_solution_provider = TrivialInitialSolution(results)
        else:
            initial_solution_provider = None
    if initial_solution_provider is None:
        cp_solver_1 = CP_MSPSP_MZN(model, cp_solver_name=solver_initial_solution)
        cp_solver_1.init_model(
            output_type=True,
            model_type="mspsp_compatible",
            ignore_sec_objective=True,
            add_objective_makespan=True,
        )
        cp_solver_1.instance["maxt"] = 200
        cp_solver_1.instance["full_output"] = True
        cp_solver_1.instance.add_string("my_search=assign_Then_start;\n")
        parameters_cp = ParametersCP.default()
        parameters_cp.time_limit = 35
        parameters_cp.time_limit_iter0 = 25
        parameters_cp.multiprocess = True
        parameters_cp.nb_process = 4
        parameters_cp.free_search = True
        results = cp_solver_1.solve(parameters_cp=parameters_cp)
        status = cp_solver_1.result_status
        if status == Status.OPTIMAL_SOLUTION:
            print("Optimal solution..")
            return results
        if len(results.list_solution_fits) > 0:
            initial_solution_provider = TrivialInitialSolution(results)
        else:
            initial_solution_provider = None

    cp_solver = CP_MSPSP_MZN(model, cp_solver_name=solver_lns)
    cp_solver.init_model(
        output_type=True,
        model_type="mspsp_compatible",
        ignore_sec_objective=False,
        add_objective_makespan=False,
    )
    if initial_solution_provider is not None:
        (
            sol,
            f,
        ) = initial_solution_provider.get_starting_solution().get_best_solution_fit()
        eval_sol = model.evaluate(sol)
        cp_solver.instance["maxt"] = eval_sol["makespan"] + 10
    else:
        cp_solver.instance["maxt"] = 170
    cp_solver.instance["full_output"] = True
    cp_solver.instance.add_string("my_search=assign_Then_start;\n")
    constraint_handler = mix_lot(
        model,
        nb_cut_parts=[1, 2, 3, 4],  # [3, 4, 5, 6], #[5, 6, 7],
        fraction_subproblems=[
            0.2,
            0.3,
            0.4,
            0.5,
        ],  # [0.1, 0.2, 0.3, 0.4, 0.5], #list(np.arange(0.05, 0.2, 0.15)),
        params_list=[
            ParamsConstraintBuilder(
                minus_delta_primary=80,
                plus_delta_primary=80,
                minus_delta_secondary=1,
                plus_delta_secondary=1,
                constraint_max_time_to_current_solution=True,
                fraction_of_task_assigned_multiskill=0.1,
                except_assigned_multiskill_primary_set=False,
                first_method_multiskill=True,
                second_method_multiskill=False,
                additional_methods=False,
            )
        ],
        objective_subproblem=ObjectiveSubproblem.MAKESPAN_SUBTASKS,
        time_windows=False,
        equilibrate_multiskill_v2=False,
        time_window_length=10,
    )
    lns = LargeNeighborhoodSearchScheduling(
        rcpsp_problem=model,
        cp_solver=cp_solver,
        constraint_handler=constraint_handler,
        partial_solution=None,
        post_process_solution=PostProMSRCPSPPreemptive(problem=model),
        initial_solution_provider=initial_solution_provider,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 25
    parameters_cp.time_limit_iter0 = 25
    parameters_cp.multiprocess = solver_lns in {
        CPSolverName.ORTOOLS,
        CPSolverName.CPOPT,
    }
    parameters_cp.nb_process = 4
    parameters_cp.free_search = True
    result_store = lns.solve(
        parameters_cp=parameters_cp,
        nb_iteration_lns=1000,
        nb_iteration_no_improvement=500,
        max_time_seconds=time_limit,
        skip_first_iteration=True if initial_solution_provider is None else None,
        stop_first_iteration_if_optimal=False,
    )
    return result_store


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    # run_lns_cpopt(file="../../MSLIB/MSLIB4/Instances4/MSLIB_Set4_58.msrcp")
    files_dict = get_data_available()
    file = files_dict["MSLIB1"][0]
    res = run_lns_cpopt(
        file, skill_level_version=False, time_limit=10, solver_lns=CPSolverName.CHUFFED
    )
    sol, fit = res.get_best_solution_fit()
    plot_resource_individual_gantt(rcpsp_model=sol.problem, rcpsp_sol=sol)
    plt.show()
    # run_lns_second_objective()
    # run_lns()
    # compare_results()
