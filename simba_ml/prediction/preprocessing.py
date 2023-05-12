"""Module with preprocessing funtionalities."""
import random
import os

import pandas as pd
from numpy import typing as npt
import numpy as np


def read_dataframes_from_csvs(path_to_csvs: str) -> list[pd.DataFrame]:
    """Reads all csv files in a given directory and returns a list of pd.Dataframes.

    Constraint: defined folder must not be empty.

    Args:
        path_to_csvs: path to the directory containing the csv files.

    Raises:
        ValueError: if the given directory is empty.
        ValueError: if the given directory contains no csvs.

    Returns:
        data: numpy array containing the data.
    """
    path_to_csvs = path_to_csvs if path_to_csvs[-1] == "/" else f"{path_to_csvs}/"
    if len(os.listdir(path_to_csvs)) == 0:
        raise ValueError("The given directory is empty.")
    if data_list := [
        pd.read_csv(path_to_csvs + file, header=0)
        for file in os.listdir(path_to_csvs)
        if file.endswith(".csv")
    ]:
        return data_list
    raise ValueError("No csvs in the specified folder.")


def convert_dataframe_to_numpy(
    data: list[pd.DataFrame],
) -> list[npt.NDArray[np.float64]]:
    """Converts a list of dataframes to a list of numpy arrays.

    Args:
        data: list of dataframes.

    Returns:
        list of numpy arrays.
    """
    return [(df.to_numpy()).astype("float64") for df in data]


# sourcery skip: require-parameter-annotation
def mix_data(
    *,
    observed_data: list[pd.DataFrame],
    synthetic_data: list[pd.DataFrame],
    ratio: float = 1,
) -> list[pd.DataFrame]:
    """Mixes real and synthetic data according to a ratio.

    Args:
        observed_data: observed data.
        synthetic_data: synthetic data.
        ratio: Ratio of synthethic to observed data.

    Raises:
        ValueError: If number of observed datapoints is not sufficient to fulfill ratio.

    Returns:
        The mixed data.
    """
    datapoints_in_observed_data = sum((len(df) for df in observed_data))
    datapoints_in_synthetic_data = sum((len(df) for df in synthetic_data))
    if datapoints_in_synthetic_data < datapoints_in_observed_data * ratio:
        raise ValueError(
            f"Synthetic data has {datapoints_in_synthetic_data} datapoints, "
            f"but {datapoints_in_observed_data * ratio} are required to fulfill"
            f"the defined ratio."
            f"Please generate more synthetic data."
        )
    synthetic_data_in_mix = []
    datapoints_in_synthetic_data_in_mix = 0
    for df in synthetic_data:
        if (
            datapoints_in_synthetic_data_in_mix + len(df)
            < datapoints_in_observed_data * ratio
        ):
            synthetic_data_in_mix.append(df)
            datapoints_in_synthetic_data_in_mix += len(df)
        else:
            synthetic_data_in_mix.append(
                df[
                    : int(
                        datapoints_in_observed_data * ratio
                        - datapoints_in_synthetic_data_in_mix
                    )
                ]
            )
            break

    res = observed_data + synthetic_data_in_mix
    random.shuffle(res)
    return res
