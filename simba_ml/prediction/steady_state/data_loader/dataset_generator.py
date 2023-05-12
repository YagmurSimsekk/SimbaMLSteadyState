"""Generates a dataset for steady state prediction from a list of np.ndarrays."""

from numpy import typing as npt
import numpy as np
import pandas as pd


def split_features_and_labels(
    samples: list[pd.DataFrame],
    start_value_params: list[str],
    prediction_params: list[str],
) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:
    """Splits a list of training samples into X and y.

    Args:
        samples: The list of training samples.
        start_value_params: The start values of the physical system
            that are used as features for the steady state prediction.
        prediction_params: The parameters that show the steady state
            (used as labels for the steady state prediction).

    Returns:
        A tuple of X and y.
    """
    x = [sample[start_value_params].to_numpy(dtype=np.float64) for sample in samples]
    y = [sample[prediction_params].to_numpy(dtype=np.float64) for sample in samples]
    return x, y


def create_dataset(
    data_list: list[pd.DataFrame],
    start_value_params: list[str],
    prediction_params: list[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Creates a dataset for steady state prediction from a list of dataframes.

    Args:
        data_list: The list of dataframes.
        start_value_params: The parameters to use as start values.
        prediction_params: The parameters to predict.

    Returns:
        A tuple of x and y.
    """
    x, y = split_features_and_labels(data_list, start_value_params, prediction_params)
    return np.concatenate(x), np.concatenate(y)
