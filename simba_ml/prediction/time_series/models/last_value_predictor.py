"""Provides a model, which predicts the last given input value."""
import dataclasses

import pandas as pd
import numpy as np
import numpy.typing as npt

from simba_ml.prediction.time_series.models import model


@dataclasses.dataclass
class LastValuePredictorConfig(model.ModelConfig):
    """Defines the configuration for the DenseNeuralNetwork."""

    name: str = "Last Value Predictor"
    seed: int = 42


class LastValuePredictor(model.Model):
    """Defines a model, which predicts the previous value."""

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

    def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predicts the next timestamps for every row.

        Args:
            data: 3 dimensional numpy array.
                First dimension contains time-series.
                Second dimension contains time steps of a time-series.
                Third dimension contains the attributes at a single timestep.

        Returns:
            A 3 dimensional numpy array, with the predicted values.

        Example:
            >>> import numpy as np
            >>> from simba_ml.prediction.time_series.models import last_value_predictor
            >>> from simba_ml.prediction.time_series.config import time_series_config
            >>> train = np.array([[[1,2], [1,2]], [[2,5], [2,6]], [[10, 11], [12,12]]])
            >>> train.shape
            (3, 2, 2)
            >>> model_config = last_value_predictor.LastValuePredictorConfig()
            >>> ts_config = time_series_config.TimeSeriesConfig(
            ...     input_features=["1", "2"],
            ...     output_features=["1", "2"],
            ...     input_length=2
            ... )
            >>> model = last_value_predictor.LastValuePredictor(ts_config, model_config)
            >>> model.train(train=train)
            >>> test_input = np.array([[[10, 10], [20, 20]], [[15, 15], [15, 16]]])
            >>> print(test_input)
            [[[10 10]
              [20 20]]
            <BLANKLINE>
             [[15 15]
              [15 16]]]
            >>> print(model.predict(test_input))
            [[[20 20]]
            <BLANKLINE>
             [[15 16]]]
        """
        self.validate_prediction_input(data)
        return np.array(
            [
                [ts[-1] for _ in range(self.time_series_params.output_length)]
                for ts in data
            ]
        )
