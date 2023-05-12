"""Provides an abstract Sparsifier.

Sparsifiers remove samples from a signal.
"""
import abc
import pandas as pd


class Sparsifier(abc.ABC):
    """A sparsifier sparsifies an input signal by removing samples."""

    @abc.abstractmethod
    def sparsify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Removes half of the samples chosen with a uniform random distributions.

        Args:
            signal: The signal to sparsify.
        """
