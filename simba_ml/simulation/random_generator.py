"""Provides a random generator."""
import numpy as np

rng = [np.random.default_rng(0)]


def get_rng() -> np.random.Generator:
    """Returns the random generator.

    Returns:
        The random generator
    """
    return rng[0]


def set_seed(seed: int) -> None:
    """Sets the seed.

    Args:
        seed: the random seed
    """
    rng[0] = np.random.default_rng(seed)
