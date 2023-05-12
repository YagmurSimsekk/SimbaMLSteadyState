"""Provides Models."""

# pylint: disable=only-importing-modules-is-allowed
from simba_ml.prediction.time_series.models.model import Model
from simba_ml.prediction.time_series.models.average_predictor import AveragePredictor
from simba_ml.prediction.time_series.models.last_value_predictor import (
    LastValuePredictor,
)

__all__ = [
    "LastValuePredictor",
    "Model",
    "AveragePredictor",
]
