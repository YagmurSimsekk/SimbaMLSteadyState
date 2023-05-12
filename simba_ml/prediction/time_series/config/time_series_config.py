"""Provides the configuration for logging."""
import dataclasses


@dataclasses.dataclass
class TimeSeriesConfig:
    """Config for the time-series parameters."""

    input_features: list[str]
    output_features: list[str]
    input_length: int = 1
    output_length: int = 1
