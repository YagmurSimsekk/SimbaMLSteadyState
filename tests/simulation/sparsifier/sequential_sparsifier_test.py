import pandas as pd

from simba_ml.simulation import sparsifier


def test_sequential_sparsifier():
    """Check that the `RandomSampleSparsifier` sparsifies the input."""
    before = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5, 6, 7, 8], "B": [2, 3, 4, 5, 6, 7, 8, 9]}
    )
    sparsifier1 = sparsifier.RandomSampleSparsifier(frac=0.5)
    sparsifier2 = sparsifier.RandomSampleSparsifier(frac=0.5)
    after = sparsifier.SequentialSparsifier([sparsifier1, sparsifier2]).sparsify(before)
    assert after.shape[0] == 2
