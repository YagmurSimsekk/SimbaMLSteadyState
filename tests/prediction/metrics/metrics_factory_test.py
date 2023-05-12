import pytest
from simba_ml.prediction.time_series.metrics import factory


def test_unknown_metric_cant_be_created():
    with pytest.raises(factory.MetricNotFoundError):
        factory.create("MyMetric")


def test_unregistered_metric_cant_be_created():
    factory.unregister("r_square")
    with pytest.raises(factory.MetricNotFoundError):
        factory.create("r_square")
