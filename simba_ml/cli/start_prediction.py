"""Module for starting a prediction pipeline."""
import os
import sys

import typing
import logging
import datetime

import pandas as pd
import click

from simba_ml.prediction.time_series.pipelines import synthetic_data_pipeline
from simba_ml.prediction.time_series.pipelines import transfer_learning_pipeline
from simba_ml.prediction.time_series.pipelines import mixed_data_pipeline

logger = logging.getLogger(__name__)


class Pipeline(typing.Protocol):
    """Protocol for a prediction pipeline."""

    def __call__(self, config_path: str) -> pd.DataFrame:
        """Runs the Pipeline.

        Args:
            config_path: path to the config file.
        """


PIPELINES: typing.Dict[str, Pipeline] = {
    "synthetic_data": synthetic_data_pipeline.main,
    "transfer_learning": transfer_learning_pipeline.main,
    "mixed_data": mixed_data_pipeline.main,
}
AVAILABLE_PIPELINES: list[str] = list(PIPELINES.keys())


@click.command()
@click.argument("pipeline", type=click.Choice(AVAILABLE_PIPELINES))
@click.option(
    "--output-path",
    type=str,
    default=f"results{datetime.datetime.now()}.csv",
    help="Path to the output file.",
)
@click.option(
    "--config-path",
    type=str,
    default="config.toml",
    help="Path to the config file.",
)
def start_prediction(pipeline: str, output_path: str, config_path: str) -> None:
    """Start a prediction pipeline.

    Args:
        pipeline: name of the pipeline.
        output_path: path to the output file.
        config_path: path to the config file.
    """
    sys.path.append(os.getcwd())
    results = PIPELINES[pipeline](config_path)
    create_dir_if_not_exists(output_path)
    results.to_csv(output_path)


def create_dir_if_not_exists(path: str) -> None:
    """Creates a directory if not exists.

    Args:
        path: A filepath
    """
    output_dir = os.path.dirname(path)
    if output_dir != "" and not os.path.exists(output_dir):
        os.mkdir(output_dir)
