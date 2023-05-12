"""Registers the pytorch lightning models as plugins."""

from simba_ml.prediction.time_series.models import factory, transfer_learning_factory
from simba_ml.prediction.time_series.models import model_to_transfer_learning_model
from simba_ml.prediction.time_series.models.pytorch_lightning import (
    dense_neural_network,
)


def register() -> None:
    """Registers the pytorch lightning models."""
    factory.register(
        "PytorchLightningDenseNeuralNetwork",
        dense_neural_network.DenseNeuralNetworkConfig,
        dense_neural_network.DenseNeuralNetwork,
    )
    transfer_learning_factory.register(
        "PytorchLightningDenseNeuralNetworkTransferLearning",
        dense_neural_network.DenseNeuralNetworkConfig,
        model_to_transfer_learning_model.model_to_transfer_learning_model_with_pretraining(  # pylint: disable=line-too-long
            dense_neural_network.DenseNeuralNetwork
        ),
    )
