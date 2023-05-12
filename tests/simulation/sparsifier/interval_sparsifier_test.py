import pytest
from simba_ml.simulation import sparsifier


def test_interval_sparsifier_raises_error_on_floats_greater_one():
    with pytest.raises(ValueError):
        sparsifier.interval_sparsifier.IntervalSparsifier(
            (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0), 1.2),
            (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(1), 5.7),
            (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0), 2.11),
        )


def test_interval_sparsifier_raises_error_on_mixed_interval_endings():
    with pytest.raises(ValueError):
        sparsifier.interval_sparsifier.IntervalSparsifier(
            (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0), 2.2),
            (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(1), 5),
            (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0), 10),
        )


def test_interval_sparsifier_raises_type_error_if_sparsifier_is_not_sparsifier():
    with pytest.raises(TypeError):
        sparsifier.interval_sparsifier.IntervalSparsifier(
            (4, 2),
            (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(1), 5),
            (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0), 10),
        )
