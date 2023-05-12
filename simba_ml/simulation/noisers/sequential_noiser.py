"""Module providing the `SequentialNoiser`."""

import pandas as pd

from simba_ml.simulation.noisers import noiser


class SequentialNoiser(noiser.Noiser):
    """The `SequentialNoiser` applies multiple given `Noiser` sequentially.

    Attributes:
        noisers: A list of `Noiser` to be applied.
    """

    def __init__(self, noisers: list[noiser.Noiser]):
        """Inits SequentialNoiser with the provided params.

        Args:
            noisers: A list of `Noiser` to be applied.
        """
        self.noisers = noisers

    def noisify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies noise to the provided signal.

        Args:
            signal: The input data.

        Returns:
            The noised signal.
        """
        for n in self.noisers:
            signal = n.noisify(signal)
        return signal
