"""Provides a model that predicts next timesteps from with a keras architecture."""
import abc
import dataclasses

import numpy as np
import numpy.typing as npt
import pandas as pd

from simba_ml.prediction.time_series.data_loader import window_generator
from simba_ml.prediction.time_series.models import model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)
from simba_ml.prediction import normalizer

try:  # pragma: no cover
    import tensorflow as tf
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Tensorflow is not installed. Please install it to use the KerasModel."
    ) from e

if tuple(int(v) for v in tf.version.VERSION.split(".")) < (
    2,
    10,
    0,
):  # pragma: no cover
    raise ImportError(
        "Tensorflow version 2.10.0 or higher is required for the KerasModel."
    )


@dataclasses.dataclass
class ArchitectureParams:
    """Defines the parameters for the architecture."""

    units: list[int] = dataclasses.field(default_factory=lambda: [32])
    activation: str = "relu"


@dataclasses.dataclass
class TrainingParams:
    """Defines the parameters for the training."""

    epochs: int = 10
    patience: int = 5
    batch_size: int = 32
    validation_split: float = 0.2
    verbose: int = 0


@dataclasses.dataclass
class KerasModelConfig(model.ModelConfig):
    """Defines the configuration for the KerasModel."""

    architecture_params: ArchitectureParams = dataclasses.field(
        default_factory=ArchitectureParams
    )
    training_params: TrainingParams = dataclasses.field(default_factory=TrainingParams)
    name: str = "Keras Model"
    normalize: bool = True
    seed: int = 42


class KerasModel(model.Model):
    """Defines a Keras model to predict the next timestamps.

    Args:
        history: History documenting the training process of the model.
    """

    model_params: KerasModelConfig

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: KerasModelConfig,
    ) -> None:
        """Initializes the model.

        Args:
            time_series_params: parameters of the time series that influence
                the training and archicture of the model.
            model_params: configuration for the model.


        Raises:
            TypeError: if input_length or output_length is not an integer.
        """
        super().__init__(time_series_params, model_params)
        self.set_seed(self.model_params.seed)
        self.history = None
        self.model_params = model_params
        self.model = self.get_model(time_series_params, model_params)
        if self.model_params.normalize:
            self.normalizer = normalizer.Normalizer()

    @abc.abstractmethod
    def get_model(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: KerasModelConfig,
    ) -> tf.keras.Model:
        """Returns the model.

        Args:
            time_series_params: parameters of the time series that influence
                the training and archicture of the model.
            model_params: configuration for the model.
        """

    def set_seed(self, seed: int) -> None:
        """Sets the seed for the model.

        Args:
            seed: seed to set.
        """
        tf.random.set_seed(seed)

    def train(self, train: list[pd.DataFrame]) -> None:
        """Trains the model with the given data.

        Args:
            train: training data.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.model_params.training_params.patience,
            mode="min",
        )
        self.model.compile(
            optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
        )
        if self.model_params.normalize:
            train = self.normalizer.normalize_train_data(train, self.time_series_params)
        X_train, y_train = window_generator.create_window_dataset(
            train, self.time_series_params
        )
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.model_params.training_params.epochs,
            callbacks=[early_stopping],
            validation_split=self.model_params.training_params.validation_split,
            verbose=self.model_params.training_params.verbose,
        )

    def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predicts the next timestamps for every row (time series).

        Args:
            data: np.array, where each dataframe is a time series.

        Returns:
            np.array, where each value is a time series.
        """
        if self.model_params.normalize:
            data = self.normalizer.normalize_test_data(data)
        prediction = self.model.predict(
            data, verbose=self.model_params.training_params.verbose
        )
        if self.model_params.normalize:
            prediction = self.normalizer.denormalize_prediction_data(prediction)
        return prediction
