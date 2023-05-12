"""This is an example-plugin which can be used in tests.

The plugin adds the ZeroPredictor model"""

import dataclasses

import pandas as pd
import numpy as np
import numpy.typing as npt

from simba_ml.prediction.time_series.models import model
from simba_ml.prediction.time_series.models import factory
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


@dataclasses.dataclass
class ZeroPredictorConfig(model.ModelConfig):
    """Defines the configuration for the DenseNeuralNetwork."""

    name: str = "Zero Predictor"


class ZeroPredictor(model.Model):
    """Defines a model, which predicts the average of the train data."""

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: ZeroPredictorConfig,
    ):
        """Inits the `AveragePredictor`.

        Args:
            input_length: the length of the input data.
            output_length: the length of the output data.
            config: the config for the model
        """
        super().__init__(time_series_params, model_params)

    def set_seed(self, seed: int) -> None:
        pass

    def train(self, train: list[pd.DataFrame]) -> None:
        pass

    def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.validate_prediction_input(data)
        return np.full(
            (data.shape[0], self.time_series_params.output_length, data.shape[2]), 0.0
        )


def register() -> None:
    factory.register("ZeroPredictor", ZeroPredictorConfig, ZeroPredictor)
