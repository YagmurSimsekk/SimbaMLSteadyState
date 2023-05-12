"""Provides an abstract Model."""
import abc

import numpy as np
import numpy.typing as npt

from simba_ml import error_handler
from simba_ml.prediction.time_series.config import (
    time_series_config,
)
from simba_ml.prediction.time_series.models import model


class TransferLearningModel(abc.ABC):
    """Defines the abstract model."""

    @property
    def name(self) -> str:
        """Returns the models name.

        Returns:
            The models name.
        """
        return self.model_params.name

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: model.ModelConfig,
    ):
        """Inits the model.

        Args:
            time_series_params: Time-series parameters that affect
                the training and architecture of models
            model_params: configuration for the model.

        Raises:
            TypeError: if input_length or output_length is not an integer.
        """
        error_handler.confirm_param_is_int(
            param=time_series_params.input_length, param_name="input_length"
        )
        error_handler.confirm_param_is_int(
            param=time_series_params.output_length, param_name="output_length"
        )
        self.time_series_params = time_series_params
        self.model_params = model_params

    @abc.abstractmethod
    def train(
        self,
        synthetic: list[npt.NDArray[np.float64]],
        observed: list[npt.NDArray[np.float64]],
    ) -> None:
        """Trains the model with the given data.

        Args:
            synthetic: synthetic time-series training data.
            observed: observed time-series training data.
        """

    @abc.abstractmethod
    def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predicts the next timesteps.

        Args:
            data: 3 dimensional numpy array.
                First dimension contains time-series.
                Second dimension contains time steps of a time-series.
                Third dimension contains the attributes at a single timestep.
        """

    def validate_prediction_input(self, data: npt.NDArray[np.float64]) -> None:
        """Validates the input of the `predict` function.

        Args:
            data: a single dataframe containing the input data,
                where the output will be predicted.

        Raises:
            ValueError: if data has incorrect shape (row length does not equal )
        """
        if data.shape[1] != self.time_series_params.input_length:
            raise ValueError(
                f"Row length ({data.shape}) should be equal"
                f"to input_length ({self.time_series_params.input_length})"
            )
