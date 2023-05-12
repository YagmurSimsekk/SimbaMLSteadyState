"""Provides the `AdditiveDerivNoiser`."""

import typing

from simba_ml.simulation import distributions


class AdditiveDerivNoiser:
    """The `AdditiveDerivNoiser` is a noiser, that adds random-values to the derivative.

    Attributes:
        distribution: A distribution to generate random noise.
    """

    def __init__(self, distribution: distributions.Distribution[float]) -> None:
        """Inits `AdditiveDerivNoiser` with the provided params.

        Args:
            distribution: A distribution to generate random noise from.
        """
        self.distribution = distribution

    def noisify(
        self,
        deriv: typing.Callable[
            [float, list[float], dict[str, float]], tuple[float, ...]
        ],
        max_t: float,
    ) -> typing.Callable[[float, list[float], dict[str, float]], tuple[float, ...]]:
        """Applies noise to the provided derivative function.

        Args:
            deriv: Derivative function.
            max_t: Adds noise up to this timestep.

        Returns:
            Noised derivative function.
        """
        noise = distributions.get_random_array_from_distribution(
            self.distribution, (int(max_t),)
        )

        def new_deriv(
            t: float, y: list[float], arguments: dict[str, float]
        ) -> tuple[float, ...]:
            old_vector = deriv(t, y, arguments)
            return (
                old_vector
                if t >= max_t
                else tuple(v + noise[int(t)] for v in old_vector)
            )

        return new_deriv
