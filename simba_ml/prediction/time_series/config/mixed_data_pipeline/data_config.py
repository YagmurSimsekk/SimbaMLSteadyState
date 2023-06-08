"""Provides the configuration for the data."""
import dataclasses
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


@dataclasses.dataclass
class DataConfig:
    """Config for time-series data."""

    ratios: list[float]
    time_series: time_series_config.TimeSeriesConfig
    observed: str | None = None
    synthetic: str | None = None
    test_split: float = 0.2
    split_axis: str = "vertical"
    input_features: list[str] | None = None
    output_features: list[str] | None = None
    export_path: str | None = None
