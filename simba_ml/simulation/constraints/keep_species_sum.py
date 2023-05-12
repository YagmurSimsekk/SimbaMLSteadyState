"""Provides a constranit, which does nothing."""
from typing import Optional
import math
import pandas as pd
from simba_ml.simulation.constraints import constraint
from simba_ml.simulation.system_model import system_model_interface


class KeepSpeciesSum(constraint.Constraint):
    """Keeps the sum of specieses constant at every timestep."""

    def __init__(
        self,
        sm: system_model_interface.SystemModelInterface,
        specieses_to_hold_sum: Optional[list[str]] = None,
        species_sum: Optional[int] = None,
    ):
        """Inits `KeepSpeciesSum`.

        Args:
            sm: A SystemModel, on which the constraint will be applied.
            specieses_to_hold_sum: The specieses, for which the sum has to be constant.
                If None, all specieses will be taken into account.
            species_sum: The sum of the given specieses.
                If None, sum on time 0 will be used.
        """
        super().__init__(sm)
        self.specieses_to_hold_sum = specieses_to_hold_sum or self.specieses.keys()
        self.specieses_sum = species_sum

    def __hold_sum(self, signal: pd.DataFrame) -> pd.DataFrame:
        specieses_sum = (
            self.specieses_sum
            if self.specieses_sum is not None
            else signal.iloc[0][self.specieses_to_hold_sum].sum()
        )
        for i, row in signal.iterrows():
            factor = specieses_sum / row[self.specieses_to_hold_sum].sum()
            for species in self.specieses_to_hold_sum:
                signal.at[i, species] *= factor
            assert math.isclose(row[self.specieses_to_hold_sum].sum(), specieses_sum)
        return signal

    def apply_noisifier(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies the noisifier to a signal. Ensures values stay in correct range.

        Args:
            signal: (pd.DataFrame) The signal.

        Returns:
            pd.DataFrame: Signal with applied noise.
        """
        noised_signal = self.sm.apply_noisifier(signal)
        noised_signal = self.__hold_sum(noised_signal)
        return noised_signal
