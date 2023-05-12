"""Removes a given relative amount of samples from a signal."""
import typing

import pandas as pd

from simba_ml.simulation.sparsifier import sparsifier as sparsifier_module


class IntervalSparsifier(sparsifier_module.Sparsifier):
    """A `Sparsifier` that sparsifies intervals with different `Sparsifier`.

    The `IntervalSparsifier` takes sparsifiers and interval endings as arguments.
    The sparsifiers are applied to the according intervals.
    """

    def __init__(
        self,
        *kinetic_parameters: tuple[
            sparsifier_module.Sparsifier, typing.Union[int, str]
        ],
    ) -> None:
        """Inits the `IntervalSparsifier`.

        Args:
            *kinetic_parameters: Pairs of (sparsifier, end_of_interval) where the
                end_of_interval is the last timestep, where the sparsifier should
                be applied. end_of_interval can either be represented explicit as
                an integer or relative to the length of the signal as a float.

        Raises:
            ValueError: If interval endings are neither ints nor floats in range [0, 1]
            TypeError: If Sparsifiers are not of type `Sparsifier`.

        Examples:
            >>> import pandas as pd
            >>> from simba_ml.simulation import sparsifier
            >>> signal = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
            >>> sparsifier.interval_sparsifier.IntervalSparsifier(
            ...     (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0), 2),
            ...     (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(1), 7),
            ...     (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0), 11),
            ... ).sparsify(signal).sort_index()
               a
            2  3
            3  4
            4  5
            5  6
            6  7

            >>> sparsifier.interval_sparsifier.IntervalSparsifier(
            ...     (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0),
            ...         0.2),
            ...     (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(1),
            ...         0.5),
            ...     (sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0),
            ...         1.0),
            ... ).sparsify(signal).sort_index()
               a
            2  3
            3  4
            4  5
        """
        self.interval_endings = tuple(arg[1] for arg in kinetic_parameters)
        self.sparsifiers = tuple(arg[0] for arg in kinetic_parameters)

        if not all(isinstance(end, int) for end in self.interval_endings) and not all(
            isinstance(end, float) and 0 <= end <= 1 for end in self.interval_endings
        ):
            raise ValueError(
                f"Interval endings must be either integers or floats"
                f"in the interval [0, 1]. {self.interval_endings} given."
            )
        self.explicit_endings = isinstance(self.interval_endings[0], int)

        if not all(
            isinstance(s, sparsifier_module.Sparsifier) for s in self.sparsifiers
        ):
            raise TypeError("Sparsifiers must be of type `Sparsifier`.")

    def __sparsify_explicit(self, signal: pd.DataFrame) -> pd.DataFrame:
        sparsified_signals = []
        for i, sparsifier in enumerate(self.sparsifiers):
            start = int(self.interval_endings[i - 1]) if i > 0 else 0
            end = int(self.interval_endings[i])
            sparsified_signals.append(sparsifier.sparsify(signal.iloc[start:end]))
        return pd.concat(sparsified_signals)

    def __sparsify_implicit(self, signal: pd.DataFrame) -> pd.DataFrame:
        sparsified_signals = []
        for i, sparsifier in enumerate(self.sparsifiers):
            start = int(self.interval_endings[i - 1] * signal.shape[0]) if i > 0 else 0
            end = int(self.interval_endings[i] * signal.shape[0])
            sparsified_signals.append(sparsifier.sparsify(signal.iloc[start:end]))
        return pd.concat(sparsified_signals)

    def sparsify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Removes some (`1-frac`) samples chosen with a uniform random distributions.

        Args:
            signal: The signal to sparsify.

        Returns:
            DataFrame: The sparsified signal.
        """
        return (
            self.__sparsify_explicit(signal)
            if self.explicit_endings
            else self.__sparsify_implicit(signal)
        )
