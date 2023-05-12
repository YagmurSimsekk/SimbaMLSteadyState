"""Removes a given relative amount of samples from a signal."""
import typing

import pandas as pd

from simba_ml import error_handler
from simba_ml.simulation.sparsifier import sparsifier


class RandomSampleSparsifier(sparsifier.Sparsifier):
    """Removes some relative amount of the given samples.

    Attributes:
        frac: The relative amount of samples to keep from the incoming signal.
            0 <= frac <= 1.

    Raises:
        TypeError: If frac is not a float.
        ValueError: If frac not in the interval [0, 1].
    """

    def __init__(self, frac: typing.Union[float, int] = 0.5) -> None:
        """Inits the `RandomSampleSparsiier`.

        Args:
            frac: The relative amount of samples to keep from the incoming signal.
        """
        error_handler.confirm_param_is_float_or_int(frac, "frac")
        error_handler.confirm_number_is_in_interval(frac, 0, 1, param_name="frac")
        self.frac = float(frac)

    def sparsify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Removes some (`1-frac`) samples chosen with a uniform random distributions.

        Args:
            signal: The signal to sparsify.

        Returns:
            DataFrame: The sparsified signal.
        """
        return signal.sample(int(signal.shape[0] * self.frac))
