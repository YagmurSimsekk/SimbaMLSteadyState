"""Defines the `MultiNoiser`."""

import numpy as np
import pandas as pd

from simba_ml.simulation.noisers import noiser


class MultiNoiser(noiser.Noiser):
    """Applies one randomly selected `Noiser` to add noise to an input signal.

    Attributes:
        noisers: A list of `Noiser` to choose from.
    """

    def __init__(self, noisers: list[noiser.Noiser]):
        """Inits MultiNoiser with the provided params.

        Args:
            noisers: A list of `Noiser` to choose from.
        """
        self.noisers = noisers

    def noisify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies noise to the provided signal.

        Args:
            signal: The input data.

        Returns:
            The noised signal.
        """
        rng = np.random.default_rng()
        selected_noiser = rng.choice(np.array(self.noisers))
        signal = selected_noiser.noisify(signal)
        return signal
