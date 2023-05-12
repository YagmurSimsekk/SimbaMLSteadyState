"""Provides the `MultiDerivNoiser`."""

import typing
import numpy as np

from simba_ml.simulation.derivative_noiser import derivative_noiser

KineticParameterType = typing.TypeVar("KineticParameterType")


class MultiDerivNoiser:
    """Applies one randomly selected `DerivNoiser` to noise a derivative function.

    Attributes:
        noisers: A list of `derivative_noiser.DerivNoiser` to choose from.
    """

    def __init__(
        self, noisers: list[derivative_noiser.DerivNoiser[KineticParameterType]]
    ) -> None:
        """Inits MultiDerivNoiser with the provided arguments.

        Args:
            noisers: A list of `derivative_noiser.DerivNoiser` to choose from.
        """
        self.noisers = noisers

    def noisify(
        self,
        deriv: typing.Callable[
            [float, list[float], dict[str, KineticParameterType]], tuple[float, ...]
        ],
        max_t: float,
    ) -> typing.Callable[
        [float, list[float], dict[str, KineticParameterType]], tuple[float, ...]
    ]:
        """Applies noise to the provided derivative function.

        Args:
            deriv: Derivative function.
            max_t: Adds noise up to this timestep.

        Returns:
            Noised derivative function.
        """
        rng = np.random.default_rng()
        noiser = rng.choice(np.array(self.noisers))
        deriv = noiser.noisify(deriv, max_t)
        return deriv
