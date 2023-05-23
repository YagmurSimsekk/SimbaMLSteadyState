"""Test file for abstract time series model."""
import numpy as np
import pytest
import pandas as pd

from simba_ml.prediction.time_series.models import last_value_predictor
from simba_ml.prediction.time_series.models import model_to_transfer_learning_model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)
from simba_ml.prediction.time_series.models.pytorch_lightning import (
    dense_neural_network,
)


def test_transfer_model():
    train = np.array([[1, 2], [1, 2]])
    config = last_value_predictor.LastValuePredictorConfig()
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=[], output_features=[]
    )
    new_model_type = model_to_transfer_learning_model.model_to_transfer_learning_model_with_pretraining(  # pylint: disable=line-too-long
        last_value_predictor.LastValuePredictor
    )
    model = new_model_type(time_series_params, config)
    with pytest.raises(ValueError):
        model.predict(train)

    synthetic_train = np.array([[1, 2], [1, 2]])
    observed_train = np.array([[1, 2], [1, 2]])
    model.train(synthetic_train, observed_train)
    test_set_wrong_format = np.array([[1, 2], [1, 2]])
    with pytest.raises(ValueError):
        model.predict(test_set_wrong_format)
    test = np.array([[1], [2]])
    results = model.predict(test)
    expected = [[1], [2]]
    assert np.array_equal(results, expected)
    assert model.name == "Last Value Predictor"


def test_pytorch_lightning_dense_neural_network_as_transfer_learning_model():
    config = dense_neural_network.DenseNeuralNetworkConfig(finetuning=True)
    config.training_params.finetuning_epochs = 1
    config.training_params.finetuning_learning_rate = 0.0001
    data = pd.DataFrame(np.random.default_rng().normal(0, 1, size=(500, 2)))
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0]
    )
    config.training_params.epochs = 1
    transfer_learning_model = model_to_transfer_learning_model.model_to_transfer_learning_model_with_pretraining(  # pylint: disable=line-too-long
        dense_neural_network.DenseNeuralNetwork
    )
    model = transfer_learning_model(time_series_params, config)
    model.train(synthetic=[data], observed=[data])
    test_data = np.random.default_rng().normal(0, 1, size=(2, 1))
    assert model.predict(test_data).shape == (2, 1, 1)
