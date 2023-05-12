"""Provides a constranit, which does nothing."""
import pandas as pd

from simba_ml.simulation.constraints import constraint
from simba_ml.simulation import species as species_module


class SpeciesValueTruncator(constraint.Constraint):
    """Truncates the values to a valid species-specific range."""

    def apply_noisifier(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies the noisifier to a signal but ensures values stay in correct range.

        Args:
            signal: (pd.DataFrame) The signal.

        Returns:
            pd.DataFrame: Signal with applied noise.
        """
        noised_signal = self.sm.apply_noisifier(signal)
        noised_signal = self.__ensure_signal_is_inside_valid_range(noised_signal)
        return noised_signal

    def __put_value_in_valid_range(
        self, x: float, species: species_module.Species
    ) -> float:
        """Puts a value into the valid range for a given species.

        The valid range is defined by `species.min` and `species.max`.

        Args:
            x: a number which will be put to valid range
            species: the species, which defines min and max

        Returns:
            The value in valid range.
        """
        if species.max is not None:
            x = min(x, species.max)
        if species.min is not None:
            x = max(x, species.min)
        return x

    def __ensure_species_is_inside_valid_range(
        self, signal: pd.DataFrame, species: str
    ) -> None:
        signal[species] = signal[species].map(
            lambda x: self.__put_value_in_valid_range(x, self.specieses[species])
        )

    def __ensure_signal_is_inside_valid_range(
        self, signal: pd.DataFrame
    ) -> pd.DataFrame:
        """Puts the species-specific part of a signal into the valid range.

        Args:
            signal: the signal having a column for the species
                which will be put to valid range.

        Returns:
            The signal with species-values in valid range.
        """
        for species in signal.keys():
            self.__ensure_species_is_inside_valid_range(signal, species)
        return signal
