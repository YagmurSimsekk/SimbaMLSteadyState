"""Defines multiple classes for applying noise to a derivative."""

import abc
import typing

KineticParameterType = typing.TypeVar("KineticParameterType")


class DerivNoiser(typing.Protocol[KineticParameterType]):
    """A DerivNoiser is a Noiser, that noises a derivative function."""

    @abc.abstractmethod
    def noisify(
        self,
        deriv: typing.Callable[
            [float, list[float], dict[str, KineticParameterType]], tuple[float, ...]
        ],
        max_t: float,
    ) -> typing.Callable[
        [float, list[float], dict[str, KineticParameterType]], tuple[float, ...]
    ]:
        """Noises the derivative.

        Args:
            deriv: The derivative function, that needs to be noised.
            max_t: Adds noise up to this timestep.
        """
