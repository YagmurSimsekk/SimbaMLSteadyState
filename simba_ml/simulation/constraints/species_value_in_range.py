"""Ensures that the values in a signal are in valid range or raises an error."""
import pandas as pd

from simba_ml.simulation.system_model import system_model_interface
from simba_ml.simulation.constraints import constraint


class MaxRetriesReachedError(Exception):
    """Raised when the maximum number of retries is reached."""


class KeepSpeciesRange(constraint.Constraint):
    """Ensures, that the values are in a valid species-specific range."""

    def __init__(
        self, sm: system_model_interface.SystemModelInterface, max_retries: int = 10
    ) -> None:
        """Inits the constraint.

        Args:
            sm: A PredictionTask, on which the constraint will be applied.
            max_retries: The maximum number of retries, if the signal
                is not in the specified range.

        Example:
            >>> from simba_ml.example_problems.sir import sm
            >>> from simba_ml.simulation import constraints
            >>> constrainted_sm = constraints.KeepSpeciesRange(sm)
        """
        super().__init__(sm)
        self.max_retries = max_retries

    def apply_noisifier(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies the noisifier to a signal but ensures, values stay in correct range.

        Args:
            signal: (pd.DataFrame) The signal.

        Returns:
            pd.DataFrame: Signal with applied noise.

        Raises:
            MaxRetriesReachedError: if the maximum number of retries
                was reached without creating an valid signal.
        """
        assert self.__check_species_in_range(
            signal
        ), "Input signal is not in valid range."
        for _ in range(self.max_retries):
            signal = self.sm.apply_noisifier(signal)
            if self.__check_species_in_range(signal):
                return signal
        raise MaxRetriesReachedError(
            f"Could not find a valid signal in {self.max_retries} retries."
        )

    def __check_species_in_range(self, signal: pd.DataFrame) -> bool:
        """Checks if the species are in the valid range.

        Args:
            signal: The signal.

        Returns:
            True iff the species are in the valid range.
        """
        return all(
            (species.min is None or species.min <= signal[name].min())
            and (species.max is None or species.max >= signal[name].max())
            for name, species in self.sm.specieses.items()
        )
