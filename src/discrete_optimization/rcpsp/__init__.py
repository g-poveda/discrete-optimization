#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Union

from discrete_optimization.rcpsp.problem import RcpspProblem

if TYPE_CHECKING:
    from discrete_optimization.rcpsp_preemptive.problem import PreemptiveRcpspProblem
    from discrete_optimization.rcpsp_preemptive.problem_specialized_constraints import (
        SpecialConstraintsPreemptiveRcpspProblem,
    )

    GENERIC_CLASS = Union[
        RcpspProblem,
        PreemptiveRcpspProblem,
        SpecialConstraintsPreemptiveRcpspProblem,
    ]
