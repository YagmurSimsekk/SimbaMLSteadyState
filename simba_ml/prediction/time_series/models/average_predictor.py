"""Provides a model, which predicts the average of the train data."""
import dataclasses
import statistics

import pandas as pd
import numpy as np
import numpy.typing as npt

from simba_ml.prediction.time_series.models import model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


@dataclasses.dataclass
class AveragePredictorConfig(model.ModelConfig):
    """Defines the configuration for the DenseNeuralNetwork."""

    name: str = "Average Predictor"


class AveragePredictor(model.Model):
    """Defines a model, which predicts the average of the train data."""

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: model.ModelConfig,
    ):
        """Inits the `AveragePredictor`.

        Args:
            time_series_params: parameters of the time series that influence
                the training and archicture of the model.
            model_params: configuration for the model.
        """
        super().__init__(time_series_params, model_params)
        self.avg = 0.0

    def set_seed(self, seed: int) -> None:
        """Sets the seed for the model.

        Args:
            seed: seed to set.
        """

    def train(self, train: list[pd.DataFrame]) -> None:
        """Trains the model with the given data.

        Args:
            train: data, that can be used for training.
        """
        self.avg = statistics.mean([df.mean().mean() for df in train])

    def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Predicts the next timestamps for every row.

        Args:
            data: 3 dimensional numpy array.
                First dimension contains time-series.
                Second dimension contains time steps of a time-series.
                Third dimension contains the attributes at a single timestep.

        Returns:
            A 3 dimensional numpy array, with the predicted values.

        Example:
            >>> import numpy as np
            >>> from simba_ml.prediction.time_series.models import average_predictor
            >>> from simba_ml.prediction.time_series.config import time_series_config
            >>> train = np.array([[[1,2], [1,2]], [[2,5], [2,6]], [[10, 11], [12,12]]])

            >>> train.shape
            (3, 2, 2)
            >>> model_config = average_predictor.AveragePredictorConfig()
            >>> ts_config = time_series_config.TimeSeriesConfig(
            ...     input_features=["1"],
            ...     output_features=["1"],
            ...     input_length=2
            ... )
            >>> model = average_predictor.AveragePredictor(ts_config, model_config)
            >>> model.train(train=train)
            >>> model.avg
            5.5
            >>> test_input = np.array([[[10, 10], [20, 20]], [[15, 15], [15, 16]]])
            >>> print(test_input)
            [[[10 10]
              [20 20]]
            <BLANKLINE>
             [[15 15]
              [15 16]]]
            >>> print(model.predict(test_input))
            [[[5.5 5.5]]
            <BLANKLINE>
             [[5.5 5.5]]]


        """
        self.validate_prediction_input(data)
        return np.full(
            (data.shape[0], self.time_series_params.output_length, data.shape[2]),
            self.avg,
        )
