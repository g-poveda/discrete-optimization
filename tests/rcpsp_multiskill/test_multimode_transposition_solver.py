#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  In this example script we're parsing dzn files that are retrieved from https://github.com/youngkd/MSPSP-InstLib
#  And run CP solver with different mzn models.


from typing import Dict, List, Set

from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CPSolverName,
    ParametersCP,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel_Variant,
    MS_RCPSPSolution,
    SkillDetail,
)
from discrete_optimization.rcpsp_multiskill.solvers.multimode_transposition import (
    MultimodeTranspositionSolver,
)


def create_toy_msrcpsp():
    skills_set: Set[str] = {"S1", "S2", "S3"}
    resources_set: Set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: Dict[int, Employee] = {
        1: Employee(
            dict_skill={"S1": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
    }
    employee = {
        1: Employee(
            dict_skill={
                "S1": SkillDetail(10, 0, 0),
                "S2": SkillDetail(10, 0, 0),
                "S3": SkillDetail(10, 0, 0),
            },
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(10, 0, 0), "S3": SkillDetail(10, 0, 0)},
            calendar_employee=[True] * 100,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(10, 0, 0)}, calendar_employee=[True] * 100
        ),
    }
    index = 5
    for emp in sorted(employee):
        indexes = [index + 8 * i for i in range(10)] + [
            index + 1 + 8 * i for i in range(10)
        ]
        for i in indexes:
            employee[emp].calendar_employee[i] = True
        index += 1

    employees_availability: List[int] = [3] * 1000
    mode_details: Dict[int, Dict[int, Dict[str, int]]] = {
        1: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
        2: {
            1: {"S1": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 2},
            2: {"S2": 1, "R1": 0, "R2": 0, "R3": 0, "duration": 3},
        },
        3: {1: {"S2": 1, "R1": 1, "R2": 2, "R3": 0, "duration": 4}},
        4: {1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        5: {1: {"R1": 2, "R2": 0, "R3": 0, "S1": 1, "duration": 5}},
        6: {1: {"S3": 1, "S2": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        7: {
            1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 1},
            2: {"R1": 2, "R2": 0, "R3": 0, "duration": 2},
        },
        8: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
    }
    successors: Dict[int, List[int]] = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [6, 7],
        6: [8],
        7: [8],
        8: [],
    }
    model = MS_RCPSPModel_Variant(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        horizon_multiplier=1,
    )
    return model


def test_multi_mode_transposition():
    model_msrcpsp = create_toy_msrcpsp()
    solver = MultimodeTranspositionSolver(
        problem=model_msrcpsp, solver_multimode_rcpsp=CP_MRCPSP_MZN
    )
    p = ParametersCP.default_cpsat()
    result_storage = solver.solve(cp_solver_name=CPSolverName.CHUFFED, parameters_cp=p)
    solution: MS_RCPSPSolution = result_storage.get_best_solution()
    assert model_msrcpsp.satisfy(solution)
