import pandas as pd
from simba_ml.simulation.sparsifier.no_sparsifier import NoSparsifier


def test_no_sparsifier():
    """Check that the `Sparsifier` sparsifies the input"""
    before = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5, 6, 7, 8], "B": [2, 3, 4, 5, 6, 7, 8, 9]}
    )
    after = NoSparsifier().sparsify(before)
    assert after.equals(before)
