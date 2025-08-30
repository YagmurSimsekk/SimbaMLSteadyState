"""This script defines the legacy CLI for SimbaML (backward compatibility).

For the new modern CLI, use: simba-ml
For legacy compatibility, use: python -m simba_ml.cli
"""
import click

from simba_ml.cli import generate_data
from simba_ml.cli import start_prediction
from simba_ml.cli.problem_viewer import run_problem_viewer
from simba_ml.cli import legacy_adapters


@click.group()
def main() -> None:
    """CLI for SimbaML (Legacy Interface).

    This is the legacy CLI interface. For the modern interface with better UX, use:

        simba-ml --help
    """


# Legacy commands - keep original names for backward compatibility
main.add_command(generate_data.generate_data, name="generate-data")
main.add_command(start_prediction.start_prediction, name="start-prediction")
main.add_command(run_problem_viewer.run_problem_viewer, name="run-problem-viewer")

# New commands - use modern implementations but keep legacy names
main.add_command(legacy_adapters.parse_sbml, name="parse-sbml")
main.add_command(legacy_adapters.biomodels, name="biomodels")
main.add_command(legacy_adapters.steady_state, name="steady-state")


if __name__ == "__main__":
    main()
