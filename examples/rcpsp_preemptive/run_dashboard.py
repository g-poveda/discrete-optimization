"""Launch interactive dashboard to visualize benchmark results."""

import argparse

from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.study.database import Hdf5Database

# Default study name
DEFAULT_STUDY = "preemptive-test"


def launch_dashboard(study_name: str = DEFAULT_STUDY, port: int = 8050):
    """Launch the dashboard for a given study.

    Args:
        study_name: Name of the study to visualize
        port: Port to run the dashboard on (default: 8050)
    """
    # Load data from database
    db_path = f"{study_name}.h5"

    print(f"Loading study: {study_name}")
    print(f"Database: {db_path}")

    with Hdf5Database(db_path) as database:
        results = database.load_results()

    print(f"Loaded {len(results)} experiments")
    print(f"\nLaunching dashboard on http://localhost:{port}")
    print("Press Ctrl+C to stop")

    # Launch dashboard
    app = Dashboard(results=results)
    app.run(port=port, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch interactive dashboard for benchmark results"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=DEFAULT_STUDY,
        help="Name of the study to visualize",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run dashboard on (default: 8050)",
    )

    args = parser.parse_args()

    launch_dashboard(study_name=args.study_name, port=args.port)
