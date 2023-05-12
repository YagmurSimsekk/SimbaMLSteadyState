"""Runs the application by starting the user interface."""
import typing
import click

from streamlit.web import bootstrap

from simba_ml.cli.problem_viewer import problem_viewer


@click.command()
@click.option(
    "--module", type=str, required=True, help="Module containing the system model"
)
def run_problem_viewer(module: str) -> None:
    """Starts the problem viewer on a local webserver.

    The user interface is documented in `problem_viewer.py.`

    Args:
        module: Module containing the system model.
    """
    file = problem_viewer.__file__
    flag_options: dict[str, typing.Any] = {}
    args = ["--module", module]
    bootstrap.run(file, None, args, flag_options)
