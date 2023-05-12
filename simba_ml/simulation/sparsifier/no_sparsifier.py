"""Provides a Dummy-Sparsifier which removes no samples from a signal."""

import pandas as pd
from simba_ml.simulation.sparsifier import sparsifier


class NoSparsifier(sparsifier.Sparsifier):
    """A dummy sparsifier that just returns the incoming signal."""

    def sparsify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Mocks to sparfify, but does not remove any sample.

        Args:
            signal: The signal to sparsify.

        Returns:
            pd.DataFrame: The (not) sparsified signal.
        """
        return signal
