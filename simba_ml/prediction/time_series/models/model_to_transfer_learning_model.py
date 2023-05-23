"""Converts a model to a transfer learning model."""

import numpy as np
from numpy import typing as npt

from simba_ml.prediction.time_series.models import model as model_module
from simba_ml.prediction.time_series.models import transfer_learning_model
from simba_ml.prediction.time_series.config import time_series_config


def model_to_transfer_learning_model_with_pretraining(
    model: type[model_module.Model],
) -> type[transfer_learning_model.TransferLearningModel]:
    """Converts a model to a transfer learning model.

    Args:
        model: model to convert.

    Returns:
        Transfer learning model.
    """

    class NewModel(transfer_learning_model.TransferLearningModel):
        """Transfer learning model."""

        def __init__(
            self,
            time_series_params: time_series_config.TimeSeriesConfig,
            model_params: model_module.ModelConfig,
        ) -> None:
            """Inits the new Transfer Learning model.

            Args:
                time_series_params: time series parameters.
                model_params: model parameters.
            """
            super().__init__(time_series_params, model_params)
            self.model = model(time_series_params, model_params)

        def train(
            self,
            synthetic: list[npt.NDArray[np.float64]],
            observed: list[npt.NDArray[np.float64]],
        ) -> None:
            """Trains the model.

            Args:
                synthetic: synthetic data.
                observed: observed data.
            """
            self.model.train(synthetic)
            self.model.model_params.finetuning = True
            self.model.train(observed)

        def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Predicts.

            Args:
                data: test data.

            Returns:
                Predictions for the test data.
            """
            self.validate_prediction_input(data)
            return self.model.predict(data)

    return NewModel
