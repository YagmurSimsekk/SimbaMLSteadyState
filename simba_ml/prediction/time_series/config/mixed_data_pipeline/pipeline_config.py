"""Provides the configuration for the pipeline."""
import dataclasses

from simba_ml.prediction.time_series.config.mixed_data_pipeline import (
    data_config,
)
from simba_ml.prediction.logging import logging_config
from simba_ml.prediction.time_series.metrics import factory as metrics_factory
from simba_ml.prediction.time_series.metrics import metrics as metrics_module


@dataclasses.dataclass
class PipelineConfig:
    """Config for the Pipeline."""

    models: list[dict[str, object]]
    metrics: list[str]
    data: data_config.DataConfig
    plugins: list[str] = dataclasses.field(default_factory=list)
    metric_functions: dict[str, metrics_module.Metric] = dataclasses.field(init=False)
    logging: logging_config.LoggingConfig | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        """Inits the PipelineConfig.

        Creates a dict mapping the given metric_ids to their respective functions.
        """
        self.metric_functions = {
            metric_id: metrics_factory.create(metric_id) for metric_id in self.metrics
        }
