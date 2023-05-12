"""Test file for abstract time series model."""
import numpy as np
import pytest

from simba_ml.prediction.time_series.models import last_value_predictor
from simba_ml.prediction.time_series.models import model_to_transfer_learning_model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_transfer_model():
    train = np.array([[1, 2], [1, 2]])
    config = last_value_predictor.LastValuePredictorConfig()
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=[], output_features=[]
    )
    new_model_type = model_to_transfer_learning_model.model_to_transfer_learning_model_with_pretraining(  # pylint: disable=line-too-long
        last_value_predictor.LastValuePredictor
    )
    model = new_model_type(time_series_params, config)
    with pytest.raises(ValueError):
        model.predict(train)

    synthetic_train = np.array([[1, 2], [1, 2]])
    observed_train = np.array([[1, 2], [1, 2]])
    model.train(synthetic_train, observed_train)
    test_set_wrong_format = np.array([[1, 2], [1, 2]])
    with pytest.raises(ValueError):
        model.predict(test_set_wrong_format)
    test = np.array([[1], [2]])
    results = model.predict(test)
    expected = [[1], [2]]
    assert np.array_equal(results, expected)
    assert model.name == "Last Value Predictor"
