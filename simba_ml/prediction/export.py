"""Exports the input and output batches to csv files for furthe exploration."""

import os

import pandas as pd
import numpy as np
from numpy import typing as npt


def export_batches(
    data: npt.NDArray[np.float64],
    features: list[str],
    export_path: str,
    file_name: str,
) -> None:
    """Exports the batches to csv files for furthe exploration.

    Args:
        data: the data to export.
        features: the features of the data.
        export_path: the path to export the data to.
        file_name: the name of the file to export the data to.
    """
    create_path_if_not_exist(os.path.join(os.getcwd(), export_path))
    for i in range(data.shape[0]):
        pd.DataFrame(data[i], columns=features).to_csv(
            os.path.join(os.getcwd(), export_path, f"{file_name}-{i}.csv"),
            index=False,
        )


def create_path_if_not_exist(path: str) -> None:
    """Creates a path if it does not exist.

    Args:
        path: the path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
