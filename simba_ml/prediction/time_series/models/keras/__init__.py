"""Registers the keras models as plugins."""

from simba_ml.prediction.time_series.models import factory
from simba_ml.prediction.time_series.models import transfer_learning_factory
from simba_ml.prediction.time_series.models.keras import dense_neural_network
from simba_ml.prediction.time_series.models import model_to_transfer_learning_model


def register() -> None:
    """Registers the keras models."""
    factory.register(
        "KerasDenseNeuralNetwork",
        dense_neural_network.DenseNeuralNetworkConfig,
        dense_neural_network.DenseNeuralNetwork,
    )
    transfer_learning_factory.register(
        "KerasDenseNeuralNetworkTransferLearning",
        dense_neural_network.DenseNeuralNetworkConfig,
        model_to_transfer_learning_model.model_to_transfer_learning_model_with_pretraining(  # pylint: disable=line-too-long
            dense_neural_network.DenseNeuralNetwork
        ),
    )
