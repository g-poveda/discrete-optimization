import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import (
    RcpspProblem,
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_preemptive.problem import (
    PreemptiveRcpspProblem,
    PreemptiveRcpspSolution,
    get_rcpsp_problemp_preemptive,
)
from discrete_optimization.rcpsp_preemptive.solvers.cpsat import (
    CpSatCalendarPreemptiveSolver,
    CpSatPreemptiveRcpspSolver,
    CpSatPreemptiveRcpspSolverUnitTime,
    ModelingPreemptive,
)
from discrete_optimization.rcpsp_preemptive.utils import (
    plot_ressource_view,
    plot_task_gantt,
)

logging.basicConfig(level=logging.INFO)


def load_preemptive_rcpsp_problem(
    problem: RcpspProblem = None, frequency: int = 5
) -> PreemptiveRcpspProblem:
    # file = get_data_available()[1]
    if problem is None:
        file = [f for f in get_data_available() if "j1201_1" in f][0]
        problem = parse_file(file)
    for r in problem.resources_list:
        if r not in problem.non_renewable_resources:
            max_capa = problem.get_max_resource_capacity(r)
            problem.resources[r] = [max_capa] * (problem.horizon * 3)
            if True:
                for i in range(len(problem.resources[r])):
                    if i % frequency == 0:
                        problem.resources[r][i] = 0
                    if i % frequency == 1:
                        problem.resources[r][i] = max_capa - 1
        else:
            max_capa = problem.get_max_resource_capacity(r)
            problem.resources[r] = [max_capa] * (problem.horizon * 3)
    problem.horizon = problem.horizon * 3
    problem.update_problem()
    preemptive = get_rcpsp_problemp_preemptive(problem)
    return preemptive


def main_cpsat():
    preemptive = load_preemptive_rcpsp_problem()
    preemptive.preemptive_indicator[6] = False
    solver = CpSatPreemptiveRcpspSolver(preemptive)
    solver.init_model(max_nb_preemption=10)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_cp=p,
        preset="Default",
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        time_limit=100,
    )
    sol = res[-1][0]
    print(preemptive.satisfy(sol), preemptive.evaluate(sol))
    plot_task_gantt(preemptive, res[-1][0])
    plot_ressource_view(preemptive, res[-1][0])
    plt.show()


def main_cpsat_unit():
    preemptive = load_preemptive_rcpsp_problem(frequency=4)
    preemptive.preemptive_indicator[6] = False
    solver = CpSatPreemptiveRcpspSolverUnitTime(preemptive)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_cp=p,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        time_limit=100,
    )
    sol = res[-1][0]
    print(preemptive.satisfy(sol), preemptive.evaluate(sol))
    plot_task_gantt(preemptive, res[-1][0])
    plot_ressource_view(preemptive, res[-1][0])
    plt.show()


def main_cpsat_cal_preemptive():
    from discrete_optimization.generic_tools.do_problem import (
        ModeOptim,
        ObjectiveHandling,
        ParamsObjectiveFunction,
    )

    preemptive = load_preemptive_rcpsp_problem(frequency=4)
    preemptive.preemptive_indicator[6] = False
    solver = CpSatCalendarPreemptiveSolver(
        preemptive,
        params_objective_function=ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan", "nb_preempted_tasks"],
            weights=[1, 10],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    solver.init_model(modeling_calendar_preemptive=ModelingPreemptive.INDICATOR)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        time_limit=120,
    )
    sol: PreemptiveRcpspSolution = res[-1][0]
    print(
        preemptive.satisfy(sol), preemptive.evaluate(sol), sol.get_nb_task_preemption()
    )
    plot_task_gantt(preemptive, sol)
    plot_ressource_view(preemptive, sol)
    plt.show()


def main_cpsat_cal_preemptive_lexico():
    from discrete_optimization.generic_tools.lexico_tools import LexicoSolver

    class WSLexico(Callback):
        def on_step_end(self, step, res, solver: LexicoSolver):
            solver.subsolver.set_warm_start_from_previous_run()

    preemptive = load_preemptive_rcpsp_problem()
    solver = CpSatCalendarPreemptiveSolver(preemptive)
    solver.init_model(modeling_calendar_preemptive=ModelingPreemptive.INDICATOR)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    lexico = LexicoSolver(problem=preemptive, subsolver=solver)
    res = lexico.solve(
        parameters_cp=p,
        callbacks=[WSLexico()],
        objectives=["makespan", "nb_preempted_tasks"],
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        time_limit=30,
    )
    plot_task_gantt(preemptive, res[-1][0])
    plot_ressource_view(preemptive, res[-1][0])
    plt.show()


def main_cpsat_cal_preemptive_pareto():
    from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
    from discrete_optimization.generic_tools.pareto_tools import CpsatParetoSolver

    class WSLexico(Callback):
        def on_step_end(self, step, res, solver: LexicoSolver):
            solver.subsolver.set_warm_start_from_previous_run()

    preemptive = load_preemptive_rcpsp_problem(frequency=8)
    solver = CpSatCalendarPreemptiveSolver(preemptive)
    solver.init_model(modeling_calendar_preemptive=ModelingPreemptive.INDICATOR)
    pareto_solver = CpsatParetoSolver(
        problem=preemptive,
        solver=solver,
        dict_function={
            "makespan": lambda sol: preemptive.evaluate(sol)["makespan"],
            "nb_preempted_tasks": lambda sol: preemptive.evaluate(sol)[
                "nb_preempted_tasks"
            ],
        },
        objective_names=["makespan", "nb_preempted_tasks"],
        delta_ref_improvement=[0, 0],
        delta_abs_improvement=[1, 1],
    )
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    front = pareto_solver.solve(
        obj_vars=[
            solver.variables["objectives"]["makespan"],
            solver.variables["objectives"]["nb_preempted_tasks"],
        ],
        subsolver_kwargs=dict(parameters_cp=p, time_limit=10),
        time_limit=300,
        take_last_solutions=True,
    )
    f1s, f2s = [], []
    for sol, fit in front:
        print(f"  Obj: {fit} | Sol: {sol}")
        f1s.append(preemptive.evaluate(sol)["makespan"])
        f2s.append(preemptive.evaluate(sol)["nb_preempted_tasks"])
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
    # Known optima for Example 9 are (1, 2) and (3, 0)
    plt.xlabel("makespan")
    plt.ylabel("nb_preempted_tasks")
    plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
    plt.grid(True)
    plt.legend()
    plt.show()


def main_cpsat_lexico():
    from discrete_optimization.generic_tools.lexico_tools import LexicoSolver

    class WSLexico(Callback):
        def on_step_end(self, step, res, solver: LexicoSolver):
            solver.subsolver.set_warm_start_from_previous_run()

    preemptive = load_preemptive_rcpsp_problem()
    solver = CpSatPreemptiveRcpspSolver(preemptive)
    solver.init_model(max_nb_preemption=10)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    lexico = LexicoSolver(problem=preemptive, subsolver=solver)
    res = lexico.solve(
        parameters_cp=p,
        callbacks=[WSLexico()],
        objectives=["makespan", "nb_preemption"],
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        time_limit=30,
    )
    plot_task_gantt(preemptive, res[-1][0])
    plot_ressource_view(preemptive, res[-1][0])
    plt.show()


if __name__ == "__main__":
    # main_cpsat_unit()
    main_cpsat_cal_preemptive()
    # main_optal_cal_preemptive()
    # main_cpsat_cal_preemptive_pareto()
