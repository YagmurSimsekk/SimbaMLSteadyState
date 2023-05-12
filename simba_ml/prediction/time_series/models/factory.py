"""Factory for creating a models."""
import dacite.core

from simba_ml.prediction.time_series import models
from simba_ml.prediction.time_series.models import sk_learn
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


class ModelNotFoundError(Exception):
    """Raised when a model type is not found."""


model_config_creation_funcs: dict[
    str, tuple[type[models.model.ModelConfig], type[models.model.Model]]
] = {}


def register(
    model_id: str,
    config_class: type[models.model.ModelConfig],
    model_class: type[models.model.Model],
) -> None:
    """Register a new model type.

    Args:
        model_id: the model type to register.
        config_class: the configuration class for the model.
        model_class: the model class.
    """
    model_config_creation_funcs[model_id] = (config_class, model_class)


def unregister(model_id: str) -> None:
    """Unregister a model type.

    Args:
        model_id: the model type to unregister.
    """
    model_config_creation_funcs.pop(model_id, None)


def create(
    model_id: str,
    model_dict: dict[str, object],
    time_series_params: time_series_config.TimeSeriesConfig,
) -> models.model.Model:
    """Create a model of a specific type, given JSON data.

    Args:
        model_id: the model type to create.
        model_dict: the JSON data to use to create the model.
        time_series_params: the time series configuration.

    Returns:
        The created model if model can be created, None otherwise.

    Raises:
        ModelNotFoundError: if the model type is unknown.
    """
    try:
        config_class, Model_class = model_config_creation_funcs[model_id]
    except KeyError as e:
        raise ModelNotFoundError(f"Model type {model_id} not found") from e

    model_config: models.model.ModelConfig = dacite.from_dict(
        data_class=config_class, data=model_dict, config=dacite.Config(strict=True)
    )
    model = Model_class(time_series_params, model_config)
    return model


register(
    "AveragePredictor",
    models.average_predictor.AveragePredictorConfig,
    models.average_predictor.AveragePredictor,
)
register(
    "LastValuePredictor",
    models.last_value_predictor.LastValuePredictorConfig,
    models.last_value_predictor.LastValuePredictor,
)

register(
    "DecisionTreeRegressor",
    sk_learn.decision_tree_regressor.DecisionTreeRegressorConfig,
    sk_learn.decision_tree_regressor.DecisionTreeRegressorModel,
)

register(
    "LinearRegressor",
    sk_learn.linear_regressor.LinearRegressorConfig,
    sk_learn.linear_regressor.LinearRegressorModel,
)
register(
    "NearestNeighborsRegressor",
    sk_learn.nearest_neighbors_regressor.NearestNeighborsConfig,
    sk_learn.nearest_neighbors_regressor.NearestNeighborsRegressorModel,
)

register(
    "RandomForestRegressor",
    sk_learn.random_forest_regressor.RandomForestRegressorConfig,
    sk_learn.random_forest_regressor.RandomForestRegressorModel,
)

register(
    "SVMRegressor",
    sk_learn.support_vector_machine_regressor.SVMRegressorConfig,
    sk_learn.support_vector_machine_regressor.SVMRegressorModel,
)
