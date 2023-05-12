"""Provides the configurations for the pipeline."""

# pylint: disable=only-importing-modules-is-allowed
# fmt: off
from simba_ml.prediction.time_series.config.synthetic_data_pipeline.data_config \
    import DataConfig
from simba_ml.prediction.time_series.config.synthetic_data_pipeline.pipeline_config \
    import PipelineConfig

__all__ = [
    "DataConfig",
    "PipelineConfig",
]
