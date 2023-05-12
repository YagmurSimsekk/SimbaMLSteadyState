import pandas as pd
import numpy as np

from simba_ml.prediction.time_series.models.sk_learn import decision_tree_regressor
from simba_ml.prediction.time_series.models.sk_learn import nearest_neighbors_regressor
from simba_ml.prediction.time_series.models.sk_learn import (
    support_vector_machine_regressor,
)
from simba_ml.prediction.time_series.config import (
    time_series_config,
)
from simba_ml.prediction.time_series.models.sk_learn import random_forest_regressor
from simba_ml.prediction.time_series.models.sk_learn import linear_regressor


def create_train_test_expected_data(input_length: int = 5, output_length: int = 2):
    train_data = [
        pd.DataFrame([[(i + 2 * j), (2 * i + 5 * j)] for i in range(1, 11)])
        for j in [1, 2, 3, 5, 7, 8, 9, 10]
    ]
    test_data = np.array(
        [
            [[(i + 2 * j), (2 * i + 5 * j)] for i in range(1, input_length + 1)]
            for j in [4, 6]
        ]
    )
    expected = np.array(
        [
            [
                [(i + 2 * j), (2 * i + 5 * j)]
                for i in range(input_length + 1, input_length + output_length + 1)
            ]
            for j in [4, 6]
        ]
    )
    return train_data, test_data, expected


def test_decision_tree_regressor_correct_prediction_one_attribute_dimension():
    config = decision_tree_regressor.DecisionTreeRegressorConfig()
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0], output_features=[0], input_length=5, output_length=2
    )
    train_data = [pd.DataFrame([[i] for i in range(1, 11)])] * 2
    model = decision_tree_regressor.DecisionTreeRegressorModel(ts_config, config)
    model.train(train=train_data)
    test_data = np.array([train_data[0].to_numpy()[:5]])
    expected = [train_data[0].to_numpy()[5:7]]
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_decision_tree_regressor_correct_prediction_two_attribute_dimension():
    config = decision_tree_regressor.DecisionTreeRegressorConfig()
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    train_data, test_data, expected = create_train_test_expected_data()
    model = decision_tree_regressor.DecisionTreeRegressorModel(ts_config, config)
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_decision_tree_regressor_correct_prediction_set_config():
    config = decision_tree_regressor.DecisionTreeRegressorConfig()
    config.criterion = decision_tree_regressor.Criterion.absolute_error
    config.splitter = decision_tree_regressor.Splitter.random
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    train_data, test_data, expected = create_train_test_expected_data()
    model = decision_tree_regressor.DecisionTreeRegressorModel(ts_config, config)
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_nearest_neighbors_regressor_correct_prediction_two_attribute_dimension():
    config = nearest_neighbors_regressor.NearestNeighborsConfig()
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    model = nearest_neighbors_regressor.NearestNeighborsRegressorModel(
        ts_config, config
    )
    train_data, test_data, expected = create_train_test_expected_data()
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_nearest_neighbors_regressor_correct_prediction_set_config():
    config = nearest_neighbors_regressor.NearestNeighborsConfig()
    config.weights = nearest_neighbors_regressor.Weights.distance
    config.n_neighbors = 2
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    model = nearest_neighbors_regressor.NearestNeighborsRegressorModel(
        ts_config, config
    )
    train_data, test_data, expected = create_train_test_expected_data()
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_support_vector_machine_regressor_correct_prediction_two_attribute_dimension():
    config = support_vector_machine_regressor.SVMRegressorConfig()
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    model = support_vector_machine_regressor.SVMRegressorModel(ts_config, config)
    train_data, test_data, expected = create_train_test_expected_data()
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_support_vector_machine_regressor_correct_prediction_set_config():
    config = support_vector_machine_regressor.SVMRegressorConfig()
    config.kernel = support_vector_machine_regressor.Kernel.rbf
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    model = support_vector_machine_regressor.SVMRegressorModel(ts_config, config)
    train_data, test_data, expected = create_train_test_expected_data()
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_random_forest_regressor_correct_prediction_two_attribute_dimension():
    config = random_forest_regressor.RandomForestRegressorConfig()
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    model = random_forest_regressor.RandomForestRegressorModel(ts_config, config)
    train_data, test_data, expected = create_train_test_expected_data()
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_random_forest_regressor_correct_prediction_set_config():
    config = random_forest_regressor.RandomForestRegressorConfig()
    config.n_estimators = 2
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    model = random_forest_regressor.RandomForestRegressorModel(ts_config, config)
    train_data, test_data, expected = create_train_test_expected_data()
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_linear_regressor_correct_prediction_two_attribute_dimension():
    config = linear_regressor.LinearRegressorConfig()
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    model = linear_regressor.LinearRegressorModel(ts_config, config)
    train_data, test_data, expected = create_train_test_expected_data()
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_linear_regressor_correct_prediction_set_config():
    config = linear_regressor.LinearRegressorConfig()
    config.n_jobs = 2
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    model = linear_regressor.LinearRegressorModel(ts_config, config)
    train_data, test_data, expected = create_train_test_expected_data()
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_decision_tree_regressor_correct_prediction_without_normalizing():
    config = decision_tree_regressor.DecisionTreeRegressorConfig(normalize=False)
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1], input_length=5, output_length=2
    )
    model = decision_tree_regressor.DecisionTreeRegressorModel(ts_config, config)
    train_data, test_data, expected = create_train_test_expected_data()
    model.train(train=train_data)
    prediction = model.predict(test_data)
    np.testing.assert_allclose(prediction, expected, rtol=0.33)


def test_decision_tree_regressor_correct_output_shape_multivariate_prediction():
    config = decision_tree_regressor.DecisionTreeRegressorConfig()
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0], input_length=5, output_length=2
    )
    train_data, test_data, _ = create_train_test_expected_data()
    model = decision_tree_regressor.DecisionTreeRegressorModel(ts_config, config)
    model.train(train=train_data)
    assert model.predict(test_data).shape == (2, 2, 1)
