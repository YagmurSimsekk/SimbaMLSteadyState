"""Defines a Distribution which only has one value."""

import typing

T = typing.TypeVar("T")


class Constant:
    """An object which represents a constant value.

    Attributes:
        value: The value.

    Raises:
        TypeError: If value is not a float or int.
    """

    def __init__(self, value: T) -> None:
        """Inits the Constant with the provided value.

        Args:
            value: The scalar used as constant value.
        """
        self.value = value

    def get_random_values(self, n: int) -> list[T]:
        """Returns an array of the constant value in the given shape.

        Args:
            n: The number of values.

        Returns:
            np.ndarray[float]
        """
        return [self.value for _ in range(n)]

    def get_samples_from_hypercube(self, n: int) -> list[T]:
        """Samples n values from a hypercube.

        Args:
            n: the number of samples.

        Returns:
            Samples of the distributions, sampled from a hypercube.
        """
        return [self.value for _ in range(n)]
