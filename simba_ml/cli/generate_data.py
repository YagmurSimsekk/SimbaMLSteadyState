"""Defines CLI command for generating data."""
import importlib
import os
import sys

import click

from simba_ml.simulation import generators
from simba_ml.simulation import random_generator


GENERATORS: dict[str, type[generators.GeneratorInterface]] = {
    "TimeSeriesGenerator": generators.TimeSeriesGenerator,
    "SteadyStateGenerator": generators.SteadyStateGenerator,
}


def __normalize_module_name(name: str) -> str:
    """Normalizes the module name.

    Args:
        name: Name of the module.

    Returns:
        Normalized module name.
    """
    return name.removesuffix(".py").replace("/", ".").replace("\\", ".")


@click.command()
@click.option(
    "--generator",
    default="TimeSeriesGenerator",
    type=click.Choice(list(GENERATORS.keys())),
    help="Which generator to use.",
)
@click.option(
    "--config-module",
    required=True,
    help="Config module containing a System Model called sm.",
)
@click.option("-n", default=100, type=int, help="Number of samples to generate.")
@click.option(
    "--output-dir",
    default="./data/",
    type=str,
    required=True,
    help="Path to the output directory.",
)
@click.option(
    "--seed",
    required=False,
    default=0,
    type=int,
    help="The random seed for the simulation part.",
)
def generate_data(
    generator: str, config_module: str, n: int, output_dir: str, seed: int
) -> None:
    """Command for generating data.

    Args:
        generator: Type of generator
        config_module: Path to the config file, that contains a System Model called sm.
        n: Number of samples to generate.
        output_dir: Path to the output directory.
        seed: The random seed
    """
    sys.path.append(os.getcwd())
    random_generator.set_seed(seed)
    sm = importlib.import_module(__normalize_module_name(config_module)).sm
    GENERATORS[generator](sm).generate_csvs(n, output_dir)
