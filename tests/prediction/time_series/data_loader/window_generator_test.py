"""Test methods for testing window_generator.py."""
import pytest
import pandas as pd
import numpy as np

from simba_ml.prediction.time_series.data_loader import window_generator
from simba_ml.prediction.time_series.config import time_series_config


def test_create_array_window_one_dimensional():
    data = pd.DataFrame(np.random.default_rng().normal(0, 1, 1000).reshape(-1, 1))
    input_length = 5
    output_length = 2
    output = window_generator.create_array_window(
        data, input_length=input_length, output_length=output_length
    )
    assert output.shape[0] == data.shape[0] - (input_length + output_length - 1)
    assert output.shape[1] == (input_length + output_length)
    assert output.shape[2] == data.shape[1]


def test_create_array_window_shift():
    data = pd.DataFrame(np.array(np.linspace(0, 9, 10).reshape(-1, 1)))
    input_length = 5
    output_length = 2
    actual = window_generator.create_array_window(
        data, input_length=input_length, output_length=output_length
    )
    expected = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]],
            [[2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]],
            [[3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]],
        ]
    )
    assert np.array_equal(expected, actual)


def test_create_array_window_multi_dimensional():
    data = pd.DataFrame(np.random.default_rng().normal(0, 1, 1000).reshape(-1, 4))
    input_length = 5
    output_length = 2
    output = window_generator.create_array_window(
        data, input_length=input_length, output_length=output_length
    )
    assert output.shape[0] == data.shape[0] - (input_length + output_length - 1)
    assert output.shape[1] == (input_length + output_length)
    assert output.shape[2] == data.shape[1]


def test_create_window_dataset():
    rng = np.random.default_rng()
    shape = (250, 4)
    data = [pd.DataFrame(rng.normal(0, 1, shape)) for _ in range(10)]
    input_length = 5
    output_length = 2
    X, y = window_generator.create_window_dataset(
        data,
        time_series_config.TimeSeriesConfig(
            input_features=[0, 1, 2, 3],
            output_features=[0, 1, 2, 3],
            input_length=input_length,
            output_length=output_length,
        ),
    )
    assert X.shape[0] == len(data) * (shape[0] - (input_length + output_length - 1))
    assert X.shape[1] == input_length
    assert X.shape[2] == shape[1]
    assert y.shape[0] == len(data) * (shape[0] - (output_length + input_length - 1))
    assert y.shape[1] == output_length
    assert y.shape[2] == shape[1]


def test_create_window_dataset_with_selective_features():
    rng = np.random.default_rng()
    shape = (250, 4)
    data = [pd.DataFrame(rng.normal(0, 1, shape)) for _ in range(10)]
    input_length = 5
    output_length = 2
    X, y = window_generator.create_window_dataset(
        data,
        time_series_config.TimeSeriesConfig(
            input_features=[0, 1, 3],
            output_features=[2, 3],
            input_length=input_length,
            output_length=output_length,
        ),
    )
    assert X.shape[0] == len(data) * (shape[0] - (input_length + output_length - 1))
    assert X.shape[1] == input_length
    assert X.shape[2] == 3
    assert y.shape[0] == len(data) * (shape[0] - (output_length + input_length - 1))
    assert y.shape[1] == output_length
    assert y.shape[2] == 2


def test_create_window_dataset_different_length():
    first_time_series = pd.DataFrame(np.array(np.linspace(0, 9, 10).reshape(-1, 1)))
    second_time_series = pd.DataFrame(np.array(np.linspace(0, 8, 9).reshape(-1, 1)))
    data = [first_time_series, second_time_series]
    input_length = 5
    output_length = 2
    X, y = window_generator.create_window_dataset(
        data,
        time_series_config.TimeSeriesConfig(
            input_features=list(first_time_series.columns),
            output_features=list(first_time_series.columns),
            input_length=input_length,
            output_length=output_length,
        ),
    )
    X_expected = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            [[2.0], [3.0], [4.0], [5.0], [6.0]],
            [[3.0], [4.0], [5.0], [6.0], [7.0]],
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            [[2.0], [3.0], [4.0], [5.0], [6.0]],
        ]
    )
    y_expected = np.array(
        [
            [[5.0], [6.0]],
            [[6.0], [7.0]],
            [[7.0], [8.0]],
            [[8.0], [9.0]],
            [[5.0], [6.0]],
            [[6.0], [7.0]],
            [[7.0], [8.0]],
        ]
    )
    assert np.array_equal(X_expected, X)
    assert np.array_equal(y_expected, y)


def test_raises_error_if_input_features_not_in_available_features():
    first_time_series = pd.DataFrame(np.array(np.linspace(0, 9, 10).reshape(-1, 1)))
    data = [first_time_series]
    input_length = 5
    output_length = 2
    with pytest.raises(ValueError):
        window_generator.create_window_dataset(
            data,
            time_series_config.TimeSeriesConfig(
                input_features=["08", "15"],
                output_features=["08", "15"],
                input_length=input_length,
                output_length=output_length,
            ),
        )


def test_raises_error_if_output_features_not_in_available_features():
    first_time_series = pd.DataFrame(np.array(np.linspace(0, 9, 10).reshape(-1, 1)))
    data = [first_time_series]
    input_length = 5
    output_length = 2
    with pytest.raises(ValueError):
        window_generator.create_window_dataset(
            data,
            time_series_config.TimeSeriesConfig(
                input_features=[0],
                output_features=["08", "15"],
                input_length=input_length,
                output_length=output_length,
            ),
        )
