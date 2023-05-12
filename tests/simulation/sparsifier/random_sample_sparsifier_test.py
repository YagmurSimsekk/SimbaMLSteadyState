import pandas as pd

import pytest
from simba_ml.simulation.sparsifier import random_sample_sparsifier


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.125, 0.2, 0.25, 0.5, 1.0])
def test_random_sample_sparsifier(frac):
    """Check that the `RandomSampleSparsifier` sparsifies the input."""
    before = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5, 6, 7, 8], "B": [2, 3, 4, 5, 6, 7, 8, 9]}
    )
    after = random_sample_sparsifier.RandomSampleSparsifier(frac=frac).sparsify(before)
    assert after.shape[0] == int(before.shape[0] * frac)


def test_random_sample_sparsifier_throws_value_error_when_frac_not_between_0_and_1():
    """Check that the `RandomSampleSparsifier` throws a TypeError
    if frac is not between 0 and 1."""
    with pytest.raises(ValueError):
        random_sample_sparsifier.RandomSampleSparsifier(frac=-2.0)
