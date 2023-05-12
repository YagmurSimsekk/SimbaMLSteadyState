"""Defines a kinetic parameter, that is constant over time."""
import typing

import pandas as pd

T_co = typing.TypeVar("T_co", covariant=True)


class ZeroNotSetError(Exception):
    """Raised if no value for the timestamp 0 is provided."""


class DictBasedKineticParameter(typing.Generic[T_co]):
    """A kinetic parameter which value depends on the timestamp and given by a dict.

    Missing value will be interpolated by using the last known value.

    Attributes:
        values: A dictionary mapping the timestamp to the value
            of the kinetic parameter.
    """

    def __init__(self, values: dict[float, T_co]):
        """Initializes a dict based kinetic parameter.

        Args:
            values: A dict mapping the timestamp to the value of the kinetic parameter.

        Raises:
            ZeroNotSetError: If no value for the timestamp 0 is provided.
        """
        if 0 not in values:
            raise ZeroNotSetError()
        self.values = pd.Series(values)

    # pylint: disable=unused-argument; Should match inferface of KineticParameter
    def prepare_samples(self, n: int) -> None:
        """Prepares a sample of the kinetic parameter.

        This method is called before a new simulation starts.

        Args:
            n: The number of samples to prepare.
        """

    # pylint: disable=unused-argument; Should match inferface of KineticParameter
    def get_at_timestamp(self, run: int, t: float) -> T_co:
        """Returns the kinetic parameters at the given timestamp.

        Args:
            t: The timestamp, at which the kinetic parameters are needed.
            run: The run (time series) of the current simulation.

        Returns:
            The kinetic parameters at the given timestamp.
        """
        return self.values[self.values.index <= t].iloc[-1]
