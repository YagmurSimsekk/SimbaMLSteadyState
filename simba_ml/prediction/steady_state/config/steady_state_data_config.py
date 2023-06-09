"""Provides the configuration for the data."""
import dataclasses


@dataclasses.dataclass
class DataConfig:
    """Config for steady state data."""

    start_value_params: list[str]
    prediction_params: list[str]
    mixing_ratios: list[float]
    observed: str | None = None
    synthethic: str | None = None
    test_split: float = 0.2
    k_cross_validation: int = 5
