import numpy as np

from simba_ml.prediction.time_series.config.synthetic_data_pipeline import (
    data_config,
)
from simba_ml.prediction.time_series.data_loader import synthetic_data_loader
from simba_ml.prediction.time_series.config import time_series_config


def test_x_test_does_not_change_when_called_multiple_times():
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
    )
    loader = synthetic_data_loader.SyntheticDataLoader(cfg)
    before = loader.X_test
    after = loader.X_test
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.X_test
    assert np.array_equal(before, after)


def test_y_test_does_not_change_when_called_multiple_times():
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
    )
    loader = synthetic_data_loader.SyntheticDataLoader(cfg)
    before = loader.y_test
    after = loader.y_test
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.y_test
    assert np.array_equal(before, after)


def test_train_sets_do_not_change_when_called_multiple_times():
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
    )
    loader = synthetic_data_loader.SyntheticDataLoader(cfg)
    before = loader.train
    after = loader.train
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.train
    assert np.array_equal(before, after)
