"""Provides a decision tree regressor for prediction."""
import enum
import dataclasses
import sklearn as sk
from sklearn import tree

from simba_ml.prediction.time_series.models.sk_learn import sk_learn_model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


class Criterion(enum.Enum):
    """The function to measure the quality of a split."""

    squared_error = "squared_error"
    """Mean squared error, which is equal to variance reduction as feature selection
        criterion and minimizes the L2 loss using the mean of each terminal node."""
    friedman_mse = "friedman_mse"
    """Mean squared error with Friedman's improvement score, which uses mean squared
        error with Friedman's improvement score for potential splits."""
    absolute_error = "absolute_error"
    """mean absolute error for the mean absolute error, which minimizes the L1 loss
        using the median of each terminal node."""
    poisson = "poisson"
    """reduction in Poisson deviance."""


class Splitter(enum.Enum):
    """The strategy used to choose the split at each node."""

    best = "best"
    """Chooses always the best split."""
    random = "random"
    """Choose randomly from the distribution of the used criterion."""


@dataclasses.dataclass
class DecisionTreeRegressorConfig(sk_learn_model.SkLearnModelConfig):
    """Defines the configuration for the DecisionTreeRegressor.

    Attributes:
        name: name of the model.
        criterion: the function to measure the quality of a split.
        splitter: the strategy used to choose the split at each node.
    """

    name: str = "Decision Tree Regressor"
    criterion: Criterion = Criterion.squared_error
    splitter: Splitter = Splitter.best


class DecisionTreeRegressorModel(sk_learn_model.SkLearnModel):
    """Defines a decisision tree regressor model for predictions."""

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: DecisionTreeRegressorConfig,
    ) -> None:
        """Initializes the configuration for the DecisionTreeRegressor.

        Args:
            time_series_params: Time-series parameters that affect
                the training and architecture of models
            model_params: configuration for the model.
        """
        super().__init__(time_series_params, model_params=model_params)

    def get_model(  # type: ignore
        self, model_params: DecisionTreeRegressorConfig
    ) -> sk.base.BaseEstimator:
        """Returns the model.

        Args:
            model_params: configuration for the model.

        Returns:
            The model.
        """
        return tree.DecisionTreeRegressor(
            criterion=model_params.criterion.value, splitter=model_params.splitter.value
        )
