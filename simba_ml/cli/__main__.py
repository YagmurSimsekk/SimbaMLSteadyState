"""This script defines a command line interface (CLI) for the SimbaML."""
import click

from simba_ml.cli import generate_data
from simba_ml.cli import start_prediction
from simba_ml.cli.problem_viewer import run_problem_viewer


@click.group()
def main() -> None:
    """CLI for SimbaML."""


main.add_command(generate_data.generate_data)
main.add_command(start_prediction.start_prediction)
main.add_command(run_problem_viewer.run_problem_viewer)


if __name__ == "__main__":
    main()
