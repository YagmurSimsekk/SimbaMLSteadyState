"""Defines Vector Distribution."""

from simba_ml import error_handler
from simba_ml.simulation import random_generator


class VectorDistribution:
    """An object which samples values from a list of numbers.

    Attributes:
        values: A list containing values.

    Raises:
        IndexError: If values is empty.
        TypeError: If values contains a value which is not a float or int.
    """

    def __init__(self, values: list[float]) -> None:
        """Inits VectorDistribution with the provided arguments.

        Args:
            values: A list containing all the valid values.
        """
        self.values = values
        error_handler.confirm_sequence_is_not_empty(self.values, "values")
        error_handler.confirm_sequence_contains_only_floats_or_ints(
            self.values, "values"
        )

    def get_random_values(self, n: int) -> list[float]:
        """Samples an array of values from the list of values with the given shape.

        Args:
            n: The number of values.

        Returns:
            np.ndarray[float]
        """
        return random_generator.get_rng().choice(self.values, size=n).tolist()

    def get_samples_from_hypercube(self, n: int) -> list[float]:
        """Samples n values from a hypercube.

        Args:
            n: the number of samples.

        Returns:
            Samples of the distributions, sampled from a hypercube.
        """
        rng = random_generator.get_rng()
        res = (
            self.values * (n // len(self.values)) + self.values[: n % len(self.values)]
        )
        rng.shuffle(res)
        return res
