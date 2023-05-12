"""Provides all the utilities needed for generating prediction_tasks via simulation."""

# pylint: disable=only-importing-modules-is-allowed
from simba_ml.simulation.sparsifier.sparsifier import Sparsifier
from simba_ml.simulation.sparsifier.interval_sparsifier import IntervalSparsifier
from simba_ml.simulation.sparsifier.random_sample_sparsifier import (
    RandomSampleSparsifier,
)
from simba_ml.simulation.sparsifier.no_sparsifier import NoSparsifier
from simba_ml.simulation.sparsifier.keep_extreme_values_sparsifier import (
    KeepExtremeValuesSparsifier,
)
from simba_ml.simulation.sparsifier.constant_suffix_remover import ConstantSuffixRemover
from simba_ml.simulation.sparsifier.sequential_sparsifier import SequentialSparsifier

__all__ = [
    "Sparsifier",
    "IntervalSparsifier",
    "RandomSampleSparsifier",
    "NoSparsifier",
    "KeepExtremeValuesSparsifier",
    "ConstantSuffixRemover",
    "SequentialSparsifier",
]
