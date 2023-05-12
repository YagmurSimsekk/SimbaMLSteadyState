"""Factory for creating a transfer learning models."""
import dacite.core

from simba_ml.prediction.time_series import models
from simba_ml.prediction.time_series.models import transfer_learning_model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)
from simba_ml.prediction.time_series.models import model_to_transfer_learning_model


class ModelNotFoundError(Exception):
    """Raised when a model type is not found."""


model_config_creation_funcs: dict[
    str,
    tuple[
        type[models.model.ModelConfig],
        type[transfer_learning_model.TransferLearningModel],
    ],
] = {}


def register(
    model_id: str,
    config_class: type[models.model.ModelConfig],
    model_class: type[transfer_learning_model.TransferLearningModel],
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
) -> transfer_learning_model.TransferLearningModel:
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
    "LastValuePredictor",
    models.last_value_predictor.LastValuePredictorConfig,
    model_to_transfer_learning_model.model_to_transfer_learning_model_with_pretraining(
        models.last_value_predictor.LastValuePredictor
    ),
)
