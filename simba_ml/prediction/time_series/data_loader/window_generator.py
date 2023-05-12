"""Module that generates specifiable windows from time-series data."""
import numpy as np
import numpy.typing as npt
import pandas as pd

from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def create_array_window(
    data: pd.DataFrame, input_length: int, output_length: int
) -> npt.NDArray[np.float64]:
    """Creates a 3 dimensional array of windows out of a single time series.

    Args:
        data: Time series of the shape (n,m), with n observations and m attributes.
        input_length: Length of the input window.
        output_length: Length of the output window.

    Returns:
        window: Time series transformed into an 3 dimensional windowing array,
        (x,y,z) with x: windows, y: time points, z: attributes
    """
    window = data.to_numpy().reshape((data.shape[0], 1, data.shape[1]))
    i = 0
    total_width = input_length + output_length
    for i in range(total_width - 1):
        new_column = np.roll((window[:, i, :]), -1, axis=0).reshape(
            window.shape[0], 1, window.shape[2]
        )
        window = np.append(window, new_column, 1)
    # Drop all rows that include missing data
    window = np.delete(
        window, [window.shape[0] - i for i in range(1, total_width)], axis=0
    )
    return window


def create_window_dataset(
    data_list: list[pd.DataFrame], config: time_series_config.TimeSeriesConfig
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Creates dataset in the form of a 3 dimensional array of windows.

    Args:
        data_list: 3D array of time series in the shape (x,y,z),
            with x observations y timestamps and m attributes.
        config: a time_series_config

    Returns:
        X: Input data of the shape (x,y,z),
            with x: windows, y: time points, z: attributes
        y: Output data of the shape (x,y,z),
            with x: windows, y: time points, z: attributes

    Raises:
        ValueError: if input/output has no intersection with available features.
    """
    if config.input_features is not None and not set(data_list[0].columns) & set(
        config.input_features
    ):
        raise ValueError(
            "Input features should have an intersection with available features."
        )
    if config.output_features is not None and not set(data_list[0].columns) & set(
        config.output_features
    ):
        raise ValueError(
            "Output features should have an intersection with available features."
        )

    windows = np.concatenate(
        [
            create_array_window(data, config.input_length, config.output_length)
            for data in data_list
        ],
        axis=0,
    )

    x_features_indices = [
        i
        for i, column in enumerate(data_list[0].columns)
        if column in config.input_features
    ]

    y_features_indices = [
        i
        for i, column in enumerate(data_list[0].columns)
        if column in config.output_features
    ]

    X = windows[:, : config.input_length, x_features_indices]
    y = windows[:, config.input_length :, y_features_indices]
    return X, y
