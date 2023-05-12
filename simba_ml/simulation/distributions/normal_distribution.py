"""Defines the Normal Distribution."""

import typing

import numpy as np
from scipy import stats

from simba_ml import error_handler


class NormalDistribution:
    """An object which samples values from a normal distributions.

    Attributes:
        mu: Mean ("centre") of the distributions.
        sigma: Standard deviation (spread or "width") of the distributions.
        Must be non-negative.

    Raises:
        ValueError: If sigma < 0.
        TypeError: If mu is not float or int.
        TypeError: If sigma is not float or int.
    """

    def __init__(
        self, mu: typing.Union[float, int], sigma: typing.Union[float, int]
    ) -> None:
        """Inits NormalDistribution with the provided arguments.

        Args:
            mu: Mean ("centre") of the distributions.
            sigma: Standard deviation (spread or "width") of the distributions.
                Must be non-negative.
        """
        self.mu = mu
        self.sigma = sigma
        error_handler.confirm_param_is_float_or_int(self.mu, "mu")
        error_handler.confirm_param_is_float_or_int(self.sigma, "sigma")
        error_handler.confirm_number_is_greater_or_equal_to_0(self.sigma, "sigma")

    def get_random_values(self, n: int) -> list[float]:
        """Samples an array with the given distribution.

        Args:
            n: The number of values.

        Returns:
            np.ndarray[float]
        """
        return np.random.default_rng().normal(self.mu, self.sigma, size=n).tolist()

    def get_samples_from_hypercube(self, n: int) -> list[float]:
        """Samples n values from a hypercube.

        Args:
            n: the number of samples.

        Returns:
            Samples of the distribution, sampled from a hypercube.
        """
        rv = stats.norm(self.mu, self.sigma)
        p = [np.random.uniform(low=i / n, high=(i + 1) / n) for i in range(n)]
        return rv.ppf(p)
