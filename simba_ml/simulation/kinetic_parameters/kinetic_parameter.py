"""Defines an abstract class for kinetic parameters."""

import typing

T_co = typing.TypeVar("T_co", covariant=True)


class KineticParameter(typing.Protocol[T_co]):
    """A KineticParameter is a parameter, that is used in the simulation."""

    def prepare_samples(self, n: int) -> None:
        """Starts the simulation.

        This method is called before a new simulation starts.

        Args:
            n: The number of samples to prepare.
        """

    def get_at_timestamp(self, run: int, t: float) -> T_co:
        """Returns the kinetic parameters at the given timestamp.

        Args:
            t: The timestamp, at which the kinetic parameters are needed.
            run: The run (time series) of the current simulation.
        """
