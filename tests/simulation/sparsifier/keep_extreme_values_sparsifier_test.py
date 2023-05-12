import pandas as pd
from simba_ml.simulation import sparsifier


def test_keep_extreme_values_sparsifier_does_not_override_extreme_values_column():
    """Check that the `KeepExtremeValuesSparsifier` sparsifies the input."""
    signal = pd.DataFrame(
        {
            "extreme_value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [1, 3, 5, 7, 9, 10, 8, 6, 5, 2],
        }
    )
    actual = (
        sparsifier.keep_extreme_values_sparsifier.KeepExtremeValuesSparsifier(
            sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0)
        )
        .sparsify(signal)
        .sort_index()
    )
    expected = pd.DataFrame(
        {"extreme_value": [1, 6, 10], "b": [1, 10, 2]}, index=[0, 5, 9]
    )
    assert actual.equals(expected)
