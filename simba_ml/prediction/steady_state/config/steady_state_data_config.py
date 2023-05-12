"""Provides the configuration for the data."""
import dataclasses
from typing import Optional


@dataclasses.dataclass
class DataConfig:
    """Config for steady state data."""

    start_value_params: list[str]
    prediction_params: list[str]
    mixing_ratios: list[float]
    observed: Optional[str] = None
    synthethic: Optional[str] = None
    test_split: float = 0.2
    k_cross_validation: int = 5
