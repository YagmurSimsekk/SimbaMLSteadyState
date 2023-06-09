"""Provides a model, which predicts next timesteps with a random forest regressor."""
import enum
import dataclasses
import sklearn as sk
from sklearn import ensemble

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
    """Mean squared error with Friedman's improvement score, which uses
    mean squared error with Friedman's improvement score for potential splits."""
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
class RandomForestRegressorConfig(sk_learn_model.SkLearnModelConfig):
    """Defines the configuration for the RandomForestRegressor.

    Attributes:
        name: name of the model.
        criterion: the function to measure the quality of a split.
        splitter: the strategy used to choose the split at each node.
    """

    name: str = "Random Forest Regressor"
    criterion: Criterion = Criterion.squared_error
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0


class RandomForestRegressorModel(sk_learn_model.SkLearnModel):
    """Defines a model, which uses a Random Forest regressor."""

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: RandomForestRegressorConfig,
    ) -> None:
        r"""Initializes the configuration for the RandomForestRegressor.

        Args:
            time_series_params: Time-series parameters that affect
                the training and architecture of models
            model_params: configuration for the model.
        """
        super().__init__(time_series_params, model_params)

    def get_model(  # type: ignore
        self, model_params: RandomForestRegressorConfig
    ) -> sk.base.BaseEstimator:
        """Returns the model.

        Args:
            model_params: configuration for the model.

        Returns:
            The model.
        """
        return ensemble.RandomForestRegressor(
            criterion=model_params.criterion.value,
            n_estimators=model_params.n_estimators,
            max_depth=model_params.max_depth,
            min_samples_split=model_params.min_samples_split,
            min_samples_leaf=model_params.min_samples_leaf,
            min_weight_fraction_leaf=model_params.min_weight_fraction_leaf,
        )
