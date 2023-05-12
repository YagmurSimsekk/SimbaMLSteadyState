"""Defines a kinetic parameter, that is constant over time."""
import typing

from simba_ml.simulation.distributions import distribution as distribution_module


T = typing.TypeVar("T")


class ConstantKineticParameter(typing.Generic[T]):
    """A constant kinetic parameter.

    Attributes:
        samples: The kinetic parameters for each run (time series)
            of the current simulation.
        distribution: The distribution for possible values of the kinetic parameter.
    """

    samples: typing.Optional[list[T]] = None

    def __init__(self, distribution: distribution_module.Distribution[T]):
        """Initializes a constant kinetic parameter.

        Args:
            distribution: The distribution for possible values of the kinetic parameter.
        """
        self.distribution = distribution

    def prepare_samples(self, n: int) -> None:
        """Prepares a sample of the kinetic parameter.

        This method is called before a new simulation starts.

        Args:
            n: The number of samples to prepare.
        """
        self.samples = self.distribution.get_samples_from_hypercube(n)

    # pylint: disable=unused-argument; Should match inferface of KineticParameter
    def get_at_timestamp(self, run: int, t: float) -> T:
        """Returns the kinetic parameters at the given timestamp.

        Args:
            t: The timestamp, at which the kinetic parameters are needed.
            run: The run (time series) of the current simulation.

        Returns:
            The kinetic parameters at the given timestamp.

        Raises:
            RuntimeError: If the the samples have not been prepared.
                Preparation is done by calling the method `prepare_samples`.
        """
        if self.samples is None:
            raise RuntimeError("The samples have not been prepared.")
        if run >= len(self.samples):
            raise RuntimeError("The run index is too large.")
        return self.samples[run]

    def set_for_run(self, run: int, value: T) -> None:
        """Sets the kinetic parameter for the given run.

        Args:
            run: The run number of the current simulation.
            value: The value of the kinetic parameter for the given run.

        Raises:
            RuntimeError: If the the samples have not been prepared.
                Preparation is done by calling the method `prepare_samples`.
            RuntimeError: If the run index is too large.
        """
        if self.samples is None:
            raise RuntimeError(
                "The samples have not been prepared."
                "Prepare them by calling prepare_samples()."
            )
        if run >= len(self.samples):
            raise RuntimeError("The run index is too large.")
        self.samples[run] = value
