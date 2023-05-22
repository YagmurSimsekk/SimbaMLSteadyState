import pandas as pd
import numpy as np
import pytest

from simba_ml.prediction.time_series.models.pytorch_lightning import (
    dense_neural_network,
)
from simba_ml.prediction.time_series.models.pytorch_lightning import (
    pytorch_lightning_model,
)
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_pytorch_lightning_dense_neural_network_predicts_correct_output_shape():
    config = dense_neural_network.DenseNeuralNetworkConfig()
    train = pd.DataFrame(np.random.default_rng().normal(0, 1, size=(500, 2)))
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0, 1]
    )
    config.training_params.epochs = 1
    model = dense_neural_network.DenseNeuralNetwork(time_series_params, config)
    model.train(train=[train])
    assert model.predict(train.to_numpy()).shape == (500, 1, 2)


def test_pytorch_lightning_dense_neural_network_multivariate_prediction():
    config = dense_neural_network.DenseNeuralNetworkConfig()
    train = pd.DataFrame(np.random.default_rng().normal(0, 1, size=(500, 2)))
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=[0, 1], output_features=[0]
    )
    config.training_params.epochs = 1
    model = dense_neural_network.DenseNeuralNetwork(time_series_params, config)
    model.train(train=[train])
    assert model.predict(train.to_numpy()).shape == (500, 1, 1)


def test_check_params_with_only_finetuning_set():
    cfg = pytorch_lightning_model.PytorchLightningModelConfig(
        finetuning=True,
    )
    with pytest.raises(ValueError):
        pytorch_lightning_model.check_funetuning_params(cfg)


def test_check_params_finetuning_and_epochs_set():
    cfg = pytorch_lightning_model.PytorchLightningModelConfig(
        finetuning=True,
    )
    cfg.training_params.finetuning_epochs = 1
    with pytest.raises(ValueError):
        pytorch_lightning_model.check_funetuning_params(cfg)


def test_check_params_finetuning_and_lr_set():
    cfg = pytorch_lightning_model.PytorchLightningModelConfig(
        finetuning=True,
    )
    cfg.training_params.finetuning_learning_rate = 0.0001
    with pytest.raises(ValueError):
        pytorch_lightning_model.check_funetuning_params(cfg)


def test_check_params_all_set_for_finetuning():
    cfg = pytorch_lightning_model.PytorchLightningModelConfig(
        finetuning=True,
    )
    cfg.training_params.finetuning_learning_rate = 0.0001
    cfg.training_params.finetuning_epochs = 1
    pytorch_lightning_model.check_funetuning_params(cfg)
