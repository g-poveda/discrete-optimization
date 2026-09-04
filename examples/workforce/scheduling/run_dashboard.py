#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.study import Hdf5Database

study_name = "tempo_comparison_teams"
study_name = "cpsat-vs-auto-multiobj"

if __name__ == "__main__":
    # retrieve data
    with Hdf5Database(f"{study_name}.h5") as database:
        results = database.load_results()
        for r in results:
            if "nb_teams" in r:
                r["obj"] = 10000 * r["nb_teams"] + r["workload_dispersion"]

    # launch dashboard with this data
    app = Dashboard(results=results)
    app.run()
