import numpy as np
import tensorflow as tf
import pandas as pd

from simba_ml.prediction.time_series.data_loader import splits
from simba_ml.prediction.time_series.models.keras import dense_neural_network
from simba_ml.prediction.time_series.models.keras import keras_model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_dense_neural_network_returns_history():
    config = dense_neural_network.DenseNeuralNetworkConfig(
        architecture_params=keras_model.ArchitectureParams(units=[32, 64, 128]),
    )
    data = pd.DataFrame(np.random.default_rng().normal(0, 1, size=(500, 1)))
    data.columns = ["Gaussian"]
    train, _ = splits.train_test_split_vertical([data], test_split=0.2, input_length=1)
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=["Gaussian"], output_features=["Gaussian"]
    )
    dense = dense_neural_network.DenseNeuralNetwork(time_series_params, config)
    assert len(dense.get_model(time_series_params, config).layers) == 6
    dense.train(train=train)
    assert isinstance(dense.history, tf.keras.callbacks.History)


def test_dense_neural_neural_network_predicts_correct_output_shape():
    config = dense_neural_network.DenseNeuralNetworkConfig(
        training_params=keras_model.TrainingParams(epochs=1),
    )
    data = pd.DataFrame(np.random.default_rng().normal(0, 1, size=(500, 2)))
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=[0, 1],
        output_features=[0],
        input_length=1,
        output_length=1,
    )
    dense = dense_neural_network.DenseNeuralNetwork(time_series_params, config)
    dense.train(train=[data])
    assert dense.predict(data).shape == (500, 1, 1)
