"""Defines multiple classes for applying noise to a signal."""

import pandas as pd

from simba_ml.simulation.noisers import noiser


class NoNoiser(noiser.Noiser):
    """The NoNoiser is a dummy `Noiser`, that applies no noise."""

    def noisify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Returns the input signal.

        Args:
            signal: The input data.

        Returns:
            The (unnoised) signal.
        """
        return signal
