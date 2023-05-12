"""Provides the `NoDerivNoiser`."""

import typing


KineticParameterType = typing.TypeVar("KineticParameterType")


class NoDerivNoiser:
    """The NoDerivNoiser is a dummy `DerivNoiser`, that applies no noise."""

    def noisify(
        self,
        deriv: typing.Callable[
            [float, list[float], dict[str, KineticParameterType]], tuple[float, ...]
        ],
        _max_t: float,
    ) -> typing.Callable[
        [float, list[float], dict[str, KineticParameterType]], tuple[float, ...]
    ]:
        """Returns the input signal.

        Args:
            deriv: Derivative function.

        Returns:
            Not noised derivative function.
        """
        return deriv
