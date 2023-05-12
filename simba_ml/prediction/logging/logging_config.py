"""Provides the configuration for logging."""
import dataclasses


@dataclasses.dataclass
class LoggingConfig:
    """Config for the data."""

    project: str
    entity: str
