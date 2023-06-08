"""Defines Continuous Uniform Distribution."""

import numpy as np

from simba_ml import error_handler
from simba_ml.simulation import random_generator


class ContinuousUniformDistribution:
    """An object which samples values from a continuous uniform distributions.

    Attributes:
        min_value: the minimal value of the distributions.
        max_value: the maximal value of the distributions.

    Raises:
        TypeError: If min_value is not float or int.
        TypeError: If max_value is not float or int.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        """Inits ContinuousUniformDistribution with the provided arguments.

        Args:
            min_value: the minimal value of the distributions.
            max_value: the maximal value of the distributions.
        """
        self.min_value = min_value
        self.max_value = max_value
        error_handler.confirm_param_is_float_or_int(self.min_value, "min_value")
        error_handler.confirm_param_is_float_or_int(self.max_value, "max_value")

    def get_random_values(self, n: int) -> list[float]:
        """Samples an array with the given distribution.

        Args:
            n: The number of values.

        Returns:
            np.ndarray[float]
        """
        return (
            random_generator.get_rng()
            .uniform(self.min_value, self.max_value, size=n)
            .tolist()
        )

    def get_samples_from_hypercube(self, n: int) -> list[float]:
        """Samples n values from a hypercube.

        Args:
            n: the number of samples.

        Returns:
            Samples of the distributions, sampled from a hypercube.
        """
        rng = random_generator.get_rng()
        res = [
            rng.uniform(min_value, min_value + (self.max_value - self.min_value) / n)
            for min_value in np.arange(
                self.min_value, self.max_value, (self.max_value - self.min_value) / n
            )
        ]
        rng.shuffle(res)
        return res
