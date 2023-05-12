import numpy as np
import pytest
import pandas as pd

from simba_ml.prediction import normalizer
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_normalizer():
    n = normalizer.Normalizer()
    train_data = [
        pd.DataFrame(np.array([[1], [1], [1], [1], [1], [2], [2], [2], [2], [2]]))
    ]
    expected_train_data = [
        np.array([[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1]])
    ]
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    actual_train_data = n.normalize_train_data(train_data, time_series_params=ts_config)
    assert np.array_equal(expected_train_data[0], actual_train_data[0])

    test_data = np.array([[[0], [1], [2], [3], [4], [5], [6]]])
    actual_test_data = n.normalize_test_data(test_data)
    expected_test_data = np.array([[[-3], [-1], [1], [3], [5], [7], [9]]])
    assert np.array_equal(expected_test_data, actual_test_data)

    prediction_data = expected_test_data
    actual_prediction_data = n.denormalize_prediction_data(prediction_data)
    assert np.array_equal(test_data, actual_prediction_data)


def test_normalizer_raises_errors_if_test_is_called_uninitialized():
    n = normalizer.Normalizer()
    data = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    with pytest.raises(normalizer.NotInitializedError):
        n.normalize_test_data(data)


def test_normalizer_raises_errors_if_denormalize_is_called_uninitialized():
    n = normalizer.Normalizer()
    data = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    with pytest.raises(normalizer.NotInitializedError):
        n.denormalize_prediction_data(data)


def test_normalizer_multidemensional_data():
    n = normalizer.Normalizer()
    X_train = [pd.DataFrame(np.array([[1, 10], [1, 20]]))]
    expected_normalized_data = [pd.DataFrame(np.array([[0, -1], [0, 1]]))]
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    normalize_train_data = n.normalize_train_data(X_train, time_series_params=ts_config)
    assert np.array_equal(expected_normalized_data, normalize_train_data)

    X_test = np.array([[[1, 10], [1, 20]]])
    normalize_test_data = n.normalize_test_data(X_test)
    expected_normalized_X_test = np.array([[[0, -1], [0, 1]]])
    assert np.array_equal(expected_normalized_X_test, normalize_test_data)
    denormalize_test_data = n.denormalize_prediction_data(normalize_test_data)
    assert np.array_equal(X_test, denormalize_test_data)


def test_normalizer_constant_feature():
    n = normalizer.Normalizer()
    data = [pd.DataFrame(np.array([[1, 10], [1, 20]]))]
    expected_normalized_data = [np.array([[0, -1], [0, 1]])]
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    normalize_train_data = n.normalize_train_data(data, time_series_params=ts_config)
    assert np.array_equal(expected_normalized_data, normalize_train_data)
    test_data = np.array([[[1, 10], [2, 20]]])
    expected_normalized_test_data = np.array([[[0, -1], [1, 1]]])
    normalize_test_data = n.normalize_test_data(test_data)
    assert np.array_equal(expected_normalized_test_data, normalize_test_data)
    denormalize_test_data = n.denormalize_prediction_data(normalize_test_data)
    assert np.array_equal(test_data, denormalize_test_data)


def test_normalizer_correct_output_dimension_multivariate_prediction_select_y():
    n = normalizer.Normalizer()
    X_train = [pd.DataFrame(np.array([[1, 10], [1, 20]]))]
    expected_normalized_data = [np.array([[0, -1], [0, 1]])]
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0], input_length=1, output_length=1
    )
    normalize_train_data = n.normalize_train_data(X_train, time_series_params=ts_config)
    assert np.array_equal(expected_normalized_data, normalize_train_data)

    X_test = np.array([[[1, 10], [1, 20]]])
    expected_normalized_X_test = np.array([[[0, -1], [0, 1]]])
    normalized_X_test = n.normalize_test_data(X_test)
    assert np.array_equal(expected_normalized_X_test, normalized_X_test)
    y_pred = np.array([[[0], [0]]])
    denormalized_y_pred = n.denormalize_prediction_data(y_pred)
    expected_normalized_y_pred = np.array([[[1], [1]]])
    assert np.array_equal(denormalized_y_pred, expected_normalized_y_pred)


def test_normalizer_correct_output_dimension_multivariate_prediction_select_x():
    n = normalizer.Normalizer()
    X_train = [pd.DataFrame(np.array([[1, 10], [1, 20]]))]
    expected_normalized_data = [pd.DataFrame(np.array([[0, -1], [0, 1]]))]
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0], output_features=[0], input_length=1, output_length=1
    )
    normalize_train_data = n.normalize_train_data(X_train, time_series_params=ts_config)
    assert np.array_equal(expected_normalized_data, normalize_train_data)

    X_test = np.array([[[1], [1]]])
    expected_normalized_X_test = np.array([[[0], [0]]])
    normalized_X_test = n.normalize_test_data(X_test)
    assert np.array_equal(expected_normalized_X_test, normalized_X_test)
    y_pred = np.array([[[0], [0]]])
    denormalized_y_pred = n.denormalize_prediction_data(y_pred)
    expected_normalized_y_pred = np.array([[[1], [1]]])
    assert np.array_equal(denormalized_y_pred, expected_normalized_y_pred)
