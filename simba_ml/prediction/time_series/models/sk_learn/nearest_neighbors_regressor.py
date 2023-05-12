"""Provides a nearest neighbors regressor, which predicts next timesteps."""
import enum
import dataclasses
import sklearn as sk
from sklearn import neighbors

from simba_ml.prediction.time_series.models.sk_learn import sk_learn_model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


class Weights(enum.Enum):
    """The function to weight the neighbors."""

    uniform = "uniform"
    """All points in each neighborhood are weighted equally."""
    distance = "distance"
    """Weight points by the inverse of their distance. in this case, closer neighbors
    of a query point will have a greater influence
    than neighbors which are further away."""


@dataclasses.dataclass
class NearestNeighborsConfig(sk_learn_model.SkLearnModelConfig):
    """Defines the configuration for the NearestNeighborsRegressor.

    Attributes:
        name: name of the model.
        n_neighbors: number of neighbors to use.
        weights: function to weight neighbors.
    """

    name: str = "Nearest Neighbors Regressor"
    n_neighbors: int = 5
    weights: Weights = Weights("uniform")


class NearestNeighborsRegressorModel(sk_learn_model.SkLearnModel):
    """Defines a model, which uses a nearest neighbors regressor."""

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: NearestNeighborsConfig,
    ) -> None:
        """Initializes the configuration for the DecisionTreeRegressor.

        Args:
            time_series_params: Time-series parameters that affect
                the training and architecture of models
            model_params: configuration for the model.
        """
        super().__init__(time_series_params, model_params)

    def get_model(  # type: ignore
        self, model_params: NearestNeighborsConfig
    ) -> sk.base.BaseEstimator:
        """Returns the model.

        Args:
            model_params: configuration for the model.

        Returns:
            The model.
        """
        return neighbors.KNeighborsRegressor(
            n_neighbors=model_params.n_neighbors, weights=model_params.weights.value
        )
