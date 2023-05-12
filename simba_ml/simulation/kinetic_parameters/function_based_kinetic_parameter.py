"""Defines a kinetic parameter, that is constant over time."""
import typing

T_co = typing.TypeVar("T_co", covariant=True)


class FunctionBasedKineticParameter(typing.Generic[T_co]):
    """A kinetic parameter which values are based on a function.

    Attributes:
        function: A function mapping the timestamp to the value
            of the kinetic parameter.
    """

    def __init__(self, function: typing.Callable[[float], T_co]):
        """Initializes a function based kinetic parameter.

        Args:
            function: A function mapping the timestamp to the value
                of the kinetic parameter.
        """
        self.function = function

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
        return self.function(t)
