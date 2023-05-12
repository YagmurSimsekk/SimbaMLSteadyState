import pytest
from simba_ml.prediction.time_series.models import factory
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_unknown_model_cant_be_created():
    """Test function for the model_factory function."""
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=[], output_features=[]
    )
    with pytest.raises(factory.ModelNotFoundError):
        factory.create("MyModel", {}, time_series_params)


def test_unregistered_model_cant_be_created():
    """Test function for the model_factory function."""
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=[], output_features=[]
    )
    factory.unregister("DenseNeuralNetwork")
    with pytest.raises(factory.ModelNotFoundError):
        factory.create("DenseNeuralNetwork", {}, time_series_params)
