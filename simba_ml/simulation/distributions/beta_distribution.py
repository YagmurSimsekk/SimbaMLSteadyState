"""Defines Beta Distribution."""

from scipy import stats

from simba_ml import error_handler

from simba_ml.simulation import random_generator


class BetaDistribution:
    """An object which samples values from a beta distributions.

    Attributes:
        alpha: Alpha parameter of the distributions.
        beta: Beta parameter of the distributions.

    Raises:
        TypeError: If alpha is not float or int.
        TypeError: If beta is not float or int.
        ValueError: If alpha <= 0.
        ValueError: If beta <= 0.
    """

    def __init__(self, alpha: float, beta: float) -> None:
        """Inits `BetaDistribution` with the provided arguments.

        Args:
            alpha: Alpha parameter of the distributions.
            beta: Beta parameter of the distributions.
        """
        self.alpha = alpha
        self.beta = beta
        error_handler.confirm_param_is_float_or_int(self.alpha, "alpha")
        error_handler.confirm_param_is_float_or_int(self.beta, "beta")
        error_handler.confirm_number_is_greater_than_0(self.alpha, "alpha")
        error_handler.confirm_number_is_greater_than_0(self.beta, "beta")

    def get_random_values(self, n: int) -> list[float]:
        """Samples an array of values with the given shape from the distributions.

        Args:
            n: The number ofnp.random values.

        Returns:
            an array of randomly sampled values.

        """
        return random_generator.get_rng().beta(self.alpha, self.beta, n).tolist()

    def get_samples_from_hypercube(self, n: int) -> list[float]:
        """Samples n values from a hypercube.

        Args:
            n: the number of samples.

        Returns:
            Samples of the distribution, sampled from a hypercube.
        """
        rv = stats.beta(self.alpha, self.beta)
        p = [
            random_generator.get_rng().uniform(low=i / n, high=(i + 1) / n)
            for i in range(n)
        ]
        rv.random_state = random_generator.get_rng()
        return rv.ppf(p)
