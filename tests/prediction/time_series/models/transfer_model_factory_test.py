import pytest
from simba_ml.prediction.time_series.models import transfer_learning_factory as factory
from simba_ml.prediction.time_series.models import last_value_predictor
from simba_ml.prediction.time_series.models import model_to_transfer_learning_model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_unknown_model_cant_be_created():
    """Test function for the model_factory function."""
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=["Test"], output_features=["Test 2"]
    )
    with pytest.raises(factory.ModelNotFoundError):
        factory.create("MyModel", {}, time_series_params)


def test_model_can_be_set():
    """Test function for the model_factory function."""
    new_model_type = model_to_transfer_learning_model.model_to_transfer_learning_model_with_pretraining(  # pylint: disable=line-too-long
        last_value_predictor.LastValuePredictor
    )
    factory.register(
        "LastValuePredictor",
        last_value_predictor.LastValuePredictorConfig,
        new_model_type,
    )
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=["Test"], output_features=["Test 2"]
    )
    factory.create("LastValuePredictor", {}, time_series_params)
    factory.unregister("LastValuePredictor")
    with pytest.raises(factory.ModelNotFoundError):
        factory.create("LastValuePredictor", {}, time_series_params)
