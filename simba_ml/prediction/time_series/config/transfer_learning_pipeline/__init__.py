"""Provides the configurations for the synthetic data pipeline."""

# pylint: disable=only-importing-modules-is-allowed
# fmt: off
from simba_ml.prediction.time_series.config.transfer_learning_pipeline.data_config \
    import DataConfig
from simba_ml.prediction.time_series.config.transfer_learning_pipeline.pipeline_config \
    import PipelineConfig

__all__ = [
    "DataConfig",
    "PipelineConfig",
]
