"""Provides a constranit, which does nothing."""
import typing

import pandas as pd
from simba_ml.simulation.system_model import system_model_interface
from simba_ml.simulation import species
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module


class Constraint:
    """Defines an constraint that does nothing.

    Write own constraints by inheriting from this one.
    """

    def __init__(self, sm: system_model_interface.SystemModelInterface):
        """Inits the constraint.

        Args:
            sm: A `SystemModel`, on which the constraint will be applied.

        Example:
            >>> from simba_ml.example_problems.sir import sm
            >>> from simba_ml.simulation.constraints.constraint import Constraint
            >>> constrainted_pt = Constraint(sm)
        """
        self.sm = sm

    @property
    def name(self) -> str:
        """Returns the name.

        Returns:
            The name.
        """
        return self.sm.name

    @property
    def specieses(self) -> dict[str, species.Species]:
        """Returns the specieses.

        Returns:
            The specieses.
        """
        return self.sm.specieses

    @property
    def deriv(
        self,
    ) -> typing.Callable[
        [float, list[float], dict[str, float]], typing.Tuple[float, ...]
    ]:
        """Returns the deriv.

        Returns:
            The deriv.
        """
        return self.sm.deriv

    @property
    def kinetic_parameters(
        self,
    ) -> typing.Dict[str, kinetic_parameters_module.KineticParameter[typing.Any]]:
        """Returns the kinetic_parameters.

        Returns:
            The kinetic_parameters.
        """
        return self.sm.kinetic_parameters

    def apply_noisifier(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies the objects noisifier to a signal.

        Args:
            signal: (pd.DataFrame) The signal.

        Returns:
            pd.DataFrame: Signal with applied noise.
        """
        return self.sm.apply_noisifier(signal)

    def apply_sparsifier(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies the objects sparsifier to a signal.

        Args:
            signal: (pd.DataFrame) The signal.

        Returns:
            pd.DataFrame: Signal of reduced features.
        """
        return self.sm.apply_sparsifier(signal)

    def get_clean_signal(
        self,
        start_values: dict[str, typing.Any],
        sample_id: int,
        deriv_noised: bool = True,
    ) -> pd.DataFrame:
        """Creates a clean signal.

        Args:
            start_values: Start values for the simulation.
            sample_id: The id of the sample.
            deriv_noised: If the derivative function should be noised.

        Returns:
            pd.DataFrame
        """
        return self.sm.get_clean_signal(
            start_values, sample_id, deriv_noised=deriv_noised
        )

    def sample_start_values_from_hypercube(self, n: int) -> dict[str, typing.Any]:
        """Creates a start_values dict.

        Args:
            n: The number of samples.

        Returns:
            dict[str, typing.Any]: The start_values dict.
        """
        return self.sm.sample_start_values_from_hypercube(n)
