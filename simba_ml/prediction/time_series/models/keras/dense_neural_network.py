"""Provides a model, which predictits next steps with a dense architecture."""
import dataclasses

import tensorflow as tf

from simba_ml.prediction.time_series.models.keras import keras_model

from simba_ml.prediction.time_series.config import (
    time_series_config,
)


@dataclasses.dataclass
class DenseNeuralNetworkConfig(keras_model.KerasModelConfig):
    """Defines the configuration for the DenseNeuralNetwork."""

    name: str = "Keras Dense Neural Network"


class DenseNeuralNetwork(keras_model.KerasModel):
    """Defines a dense neural network to predict the next timestamps.

    Args:
        history: History documenting the training process of the model.
    """

    def get_model(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: keras_model.KerasModelConfig,
    ) -> tf.keras.Model:
        """Returns the model.

        Args:
            time_series_params: parameters of time series that affects
                training and architecture of the model
            model_params: configuration for the model.

        Returns:
            The uncompiled model.
        """
        return tf.keras.Sequential(
            [tf.keras.layers.Flatten()]
            + [
                tf.keras.layers.Dense(units=units, activation="relu")
                for units in self.model_params.architecture_params.units
            ]
            + [
                tf.keras.layers.Dense(
                    units=time_series_params.output_length
                    * len(time_series_params.output_features)
                ),
                tf.keras.layers.Reshape([time_series_params.output_length, -1]),
            ]
        )
