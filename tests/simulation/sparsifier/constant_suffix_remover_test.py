import pytest

import pandas as pd
from simba_ml.simulation import sparsifier


def test_constant_suffix_remover_absolute():
    """Check that the `KeepExtremeValuesSparsifier` sparsifies the input."""
    signal = pd.DataFrame(
        {
            "a": [4, 5, 4, 5, 4, 5, 3, 1, 2, 1, 3, 3, 1],
            "b": [4, 5, 4, 5, 4, 1, 3, 1, 2, 1, 3, 3, 1],
        }
    )
    actual = (
        sparsifier.ConstantSuffixRemover(n=2, epsilon=1, mode="absolute")
        .sparsify(signal)
        .sort_index()
    )
    assert actual.equals(signal[:6])


def test_constant_suffix_remover_relative():
    """Check that the `KeepExtremeValuesSparsifier` sparsifies the input."""
    signal = pd.DataFrame({"a": [2, 2, 2, 1.5, 1.1, 1.0, 0.95, 0.9]})
    actual = (
        sparsifier.ConstantSuffixRemover(n=2, epsilon=0.1, mode="relative")
        .sparsify(signal)
        .sort_index()
    )
    assert actual.equals(signal[:4])


def test_constant_suffix_remover_raises_error_if_invalid_mode():
    with pytest.raises(ValueError):
        sparsifier.ConstantSuffixRemover(mode="invalid")


def test_direct_return_if_no_constant_suffix():
    signal = pd.DataFrame({"a": [2, 2, 2, 1.5, 1.1, 1.0, 0.95, 0.9]})
    actual = (
        sparsifier.ConstantSuffixRemover(n=10, epsilon=0.1, mode="relative")
        .sparsify(signal)
        .sort_index()
    )
    assert actual.equals(signal)
