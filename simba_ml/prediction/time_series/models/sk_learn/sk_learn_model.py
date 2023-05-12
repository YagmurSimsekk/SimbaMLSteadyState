"""Provides an arbitrary sk learn model architecture for prediction."""
import abc
import dataclasses

import pandas as pd
import sklearn as sk
import numpy as np
import numpy.typing as npt

from simba_ml.prediction.time_series.models import model
from simba_ml.prediction import normalizer
from simba_ml.prediction.time_series.data_loader import window_generator
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


@dataclasses.dataclass
class SkLearnModelConfig(model.ModelConfig):
    """Defines the configuration for the SkLearnModel."""

    name: str = "SkLearn Model"
    normalize: bool = True
    seed: int = 42


class SkLearnModel(model.Model):
    """Defines a model that uses the scikit learn library for prediction.

    Args:
        config: configuration for the model.
    """

    model_params: SkLearnModelConfig
    normalizer: normalizer.Normalizer

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: SkLearnModelConfig,
    ) -> None:
        """Initializes the model.

        Args:
            time_series_params: configuration for time series tasks.
            model_params: configuration for the model.

        """
        super().__init__(
            time_series_params,
            model_params,
        )
        self.set_seed(self.model_params.seed)
        if self.model_params.normalize:
            self.normalizer = normalizer.Normalizer()
        self.model = self.get_model(model_params)

    @abc.abstractmethod
    def get_model(self, model_params: SkLearnModelConfig) -> sk.base.BaseEstimator:
        """Returns the model.

        Args:
            model_params: configuration for the model.
        """

    def set_seed(self, seed: int) -> None:
        """Sets the seed for the model.

        Args:
            seed: seed to set.
        """
        np.random.seed(seed)

    def train(self, train: list[pd.DataFrame]) -> None:
        """Trains the model with the train data flattened to two dimensions.

        Args:
            train: training data.
        """
        if self.model_params.normalize:
            self.normalizer = normalizer.Normalizer()
            train = self.normalizer.normalize_train_data(train, self.time_series_params)
        X_train, y_train = window_generator.create_window_dataset(
            train, self.time_series_params
        )
        X_train = np.reshape(
            X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2])
        )
        y_train = np.reshape(
            y_train, (y_train.shape[0], y_train.shape[1] * y_train.shape[2])
        )
        self.model.fit(X_train, y_train)

    def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predicts the next time steps.

        Args:
            data: 3 dimensional numpy array.

        Returns:
            The predicted next time steps.
        """
        if self.model_params.normalize:
            data = self.normalizer.normalize_test_data(data)
        original_data_shape = data.shape
        data = np.reshape(
            data,
            (
                data.shape[0],
                data.shape[1] * data.shape[2],
            ),
        )
        prediction = self.model.predict(data)
        prediction = np.reshape(
            prediction,
            (
                original_data_shape[0],
                self.time_series_params.output_length,
                len(self.time_series_params.output_features),
            ),
        )
        if self.model_params.normalize:
            prediction = self.normalizer.denormalize_prediction_data(prediction)
        return prediction
