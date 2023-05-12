"""Provides functions to sample from distributions."""
import typing
import math
import numpy as np
from numpy import typing as npt

from simba_ml.simulation.distributions import distribution as distribution_module

T = typing.TypeVar("T")


def get_random_value_from_distribution(
    distribution: distribution_module.Distribution[T],
) -> T:
    """Samples a random value from the given distribution.

    Args:
        distribution: The distribution to sample from.

    Returns:
        A random value sampled from the distribution.
    """
    return distribution.get_random_values(n=1)[0]


def get_random_array_from_distribution(
    distribution: distribution_module.Distribution[float], shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Samples a random array from the given distribution.

    Args:
        distribution: The distribution to sample from.
        shape: The shape of the output array.

    Returns:
        A random array sampled from the distribution.
    """
    return np.array(distribution.get_random_values(n=math.prod(shape))).reshape(shape)
