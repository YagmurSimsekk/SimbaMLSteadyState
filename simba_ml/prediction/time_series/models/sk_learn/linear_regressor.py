"""Provides a model, which predicts next timesteps from with a linear regressor."""
import dataclasses

import sklearn as sk
from sklearn import linear_model

from simba_ml.prediction.time_series.models.sk_learn import sk_learn_model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


@dataclasses.dataclass
class LinearRegressorConfig(sk_learn_model.SkLearnModelConfig):
    """Defines the configuration for the LinearRegressor.

    Attributes:
        name: name of the model.
        criterion: the function to measure the quality of a split.
        splitter: the strategy used to choose the split at each node.
    """

    name: str = "Linear Regressor"
    fit_intercept: bool = True
    n_jobs: int | None = None
    positive: bool = False


class LinearRegressorModel(sk_learn_model.SkLearnModel):
    """Defines a model, which uses a Linear Regressor to predict the next timestamps."""

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: LinearRegressorConfig,
    ) -> None:
        """Initializes the configuration for the LinearRegressor.

        Args:
            time_series_params: Time-series parameters that affect
                the training and architecture of models
            model_params: configuration for the model.
        """
        super().__init__(time_series_params, model_params)

    def get_model(  # type: ignore
        self, model_params: LinearRegressorConfig
    ) -> sk.base.BaseEstimator:
        """Returns the model.

        Args:
            model_params: configuration for the model.

        Returns:
            The model.
        """
        return linear_model.LinearRegression(
            fit_intercept=model_params.fit_intercept,
            n_jobs=model_params.n_jobs,
            positive=model_params.positive,
        )
