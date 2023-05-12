"""Defines an abstract definition of `Noiser`."""

import abc

import pandas as pd


class Noiser(abc.ABC):
    """A Noiser puts noise to a sample."""

    @abc.abstractmethod
    def noisify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Noises an incomming signal.

        Args:
            signal: The signal which should be noised.
        """
