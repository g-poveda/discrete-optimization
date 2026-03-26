#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    ConstraintExtractorList,
    NbChangesAllocationConstraintExtractor,
    ParamsConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderTaskThresholdTime,
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import (
    InitialSolutionFromSolver,
    ReinitModelCallback,
)
from discrete_optimization.rcalbp_l.problem import RCALBPLSolution
from discrete_optimization.rcalbp_l.solvers.cpsat import CpSatRCALBPLSolver
from discrete_optimization.rcalbp_l.solvers.meta_solvers import (
    BackwardSequentialRCALBPLSolver,
)


def test_lns_with_initial_solver(problem):
    """Test LNS with BackwardSequential initial solver."""
    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    # Create initial solver
    initial_solver = BackwardSequentialRCALBPLSolver(
        problem=problem,
        future_chunk_size=1,
        phase2_chunk_size=2,
        time_limit_phase1=5,
        time_limit_phase2=5,
        use_sgs_warm_start=True,
    )

    initial_solution_provider = InitialSolutionFromSolver(
        solver=initial_solver,
        params_objective_function=p,
    )

    # Create constraint handler
    constraints_extractor = ConstraintExtractorList(
        extractors=[
            NbChangesAllocationConstraintExtractor(
                nb_changes_max=int(0.2 * len(problem.tasks_list))
            ),
        ]
    )

    constraint_handler = TasksConstraintHandler(
        problem=problem,
        neighbor_builder=NeighborBuilderMix(
            [
                NeighborRandom(problem=problem, fraction_subproblem=0.3),
                NeighborBuilderTaskThresholdTime(
                    problem=problem, threshold=problem.c_target
                ),
            ],
            [0.5, 0.5],
        ),
        params_constraint_extractor=ParamsConstraintExtractor(
            constraint_to_current_solution_makespan=False,
            margin_rel_to_current_solution_makespan=0.05,
            fix_primary_tasks_modes=False,
            fix_secondary_tasks_modes=False,
        ),
        constraints_extractor=constraints_extractor,
    )

    # Create LNS solver
    subsolver = CpSatRCALBPLSolver(problem=problem)
    subsolver.init_model(add_heuristic_constraint=False)

    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )

    # Solve with limited iterations
    result_storage = solver.solve(
        callbacks=[
            ReinitModelCallback(),
            WarmStartCallback(
                warm_start_best_solution=True, warm_start_last_solution=False
            ),
        ],
        nb_iteration_lns=3,
        time_limit_subsolver_iter0=5,
        time_limit_subsolver=5,
        parameters_cp=p,
    )

    # Check solution quality
    sol: RCALBPLSolution = result_storage.get_best_solution()
    assert problem.satisfy(sol), "LNS solution should be feasible"

    # Verify evaluation works
    evaluation = problem.evaluate(sol)
    assert "ramp_up_duration" in evaluation
    assert "nb_adjustments" in evaluation


def test_lns_simple(problem):
    """Test simple LNS without initial solver."""
    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    # Create constraint handler
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        neighbor_builder=NeighborRandom(problem=problem, fraction_subproblem=0.4),
        params_constraint_extractor=ParamsConstraintExtractor(
            constraint_to_current_solution_makespan=False,
        ),
        constraints_extractor=ConstraintExtractorList(extractors=[]),
    )

    # Create LNS solver
    subsolver = CpSatRCALBPLSolver(problem=problem)
    subsolver.init_model(add_heuristic_constraint=False)

    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )

    # Solve with very limited iterations for speed
    # Skip initial solution provider and let first iteration find initial solution
    result_storage = solver.solve(
        nb_iteration_lns=2,
        time_limit_subsolver_iter0=10,
        time_limit_subsolver=5,
        parameters_cp=p,
        skip_initial_solution_provider=True,
    )

    # Should have at least one solution
    assert len(result_storage) > 0

    sol: RCALBPLSolution = result_storage.get_best_solution()
    assert problem.satisfy(sol)
