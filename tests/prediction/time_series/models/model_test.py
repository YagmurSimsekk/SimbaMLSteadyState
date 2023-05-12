"""Test file for abstract time series model."""
import numpy as np
import pytest

from simba_ml.prediction.time_series.models import last_value_predictor
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_validate_prediction_input():
    train = np.array([[1, 2], [1, 2]])
    config = last_value_predictor.LastValuePredictorConfig()
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=[], output_features=[]
    )
    model = last_value_predictor.LastValuePredictor(time_series_params, config)
    with pytest.raises(ValueError):
        model.predict(train)
