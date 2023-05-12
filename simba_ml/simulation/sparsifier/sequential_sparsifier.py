"""Module providing the `SequentialNoiser`."""

import pandas as pd

from simba_ml.simulation.sparsifier import sparsifier


class SequentialSparsifier(sparsifier.Sparsifier):
    """The `SequentialNoiser` applies multiple given `Noiser` sequentially.

    Attributes:
        noisers: A list of `Noiser` to be applied.
    """

    def __init__(self, sparsifiers: list[sparsifier.Sparsifier]):
        """Inits SequentialNoiser with the provided params.

        Args:
            sparsifiers: A list of `Sparsifiers` to be applied.
        """
        self.sparsifiers = sparsifiers

    def sparsify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Sparsifies to the provided signal.

        Args:
            signal: The input data.

        Returns:
            The noised signal.
        """
        for s in self.sparsifiers:
            signal = s.sparsify(signal)
        return signal
