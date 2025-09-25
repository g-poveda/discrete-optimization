#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
from enum import Enum


class DoProblemEnum(Enum):
    JSP = 0
    FJSP = 1
    RCPSP = 2
    MRCPSP = 3


this_folder = os.path.abspath(os.path.dirname(__file__))
models_folder = os.path.join(this_folder, "models/")
# This will be updated when new model enrich our collection
problem_to_script_path = {
    DoProblemEnum.JSP: os.path.join(models_folder, "jsp/jobshop.mts"),
    DoProblemEnum.FJSP: os.path.join(models_folder, "fjsp/flexible-jobshop.mts"),
    DoProblemEnum.RCPSP: os.path.join(models_folder, "rcpsp/rcpsp.mts"),
    DoProblemEnum.MRCPSP: os.path.join(models_folder, "rcpsp/mrcpsp.mts"),
}
