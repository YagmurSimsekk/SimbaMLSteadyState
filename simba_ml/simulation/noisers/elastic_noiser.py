"""Defines the `ElasticNoiser`."""

import numpy as np
import pandas as pd

from simba_ml.simulation.noisers import noiser
from simba_ml.simulation import distributions


class ElasticNoiser(noiser.Noiser):
    """The `ElasticNoiser` applies noise elastically.

    Noise is added randomly at every point using a normal distributions
    where the variance increases with t.

    Attributes:
        k: maximal variance of the normal distributions
        invert: If True, noise is added at the beggining of the curve.
        exponential: If True, uses exponentially increasing noise.
            If invert = True, exponentially decreasing.
    """

    def __init__(
        self,
        k: distributions.Distribution[float],
        invert: bool = False,
        exponential: bool = False,
    ) -> None:
        """Inits ElasticNoiser with the provided params.

        Args:
            k: maximal variance of the normal distributions
            invert: If True, noise is added at the beggining of the curve.
            exponential: If True, uses exponentially increasing noise.
                If invert = True, exponentially decreasing.
        """
        self.k = k
        self.invert = invert
        self.exponential = exponential

    def noisify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies noise to the provided signal.

        Applies random gaussian noise with increasing or decreasing variance
        over time to the signal.

        Args:
            signal: The input data.

        Returns:
            pd.DataFrame

        Example:
            >>> import pandas as pd
            >>> from simba_ml.simulation import distributions
            >>> from simba_ml.simulation.noisers.elastic_noiser import ElasticNoiser
            >>> series = [[0]] * 1000
            >>> df = pd.DataFrame(series)
            >>> df.head()
               0
            0  0
            1  0
            2  0
            3  0
            4  0
            >>> noisers = ElasticNoiser(distributions.Constant(10))
            >>> noisers.noisify(df).plot() # doctest: +SKIP

            .. image:: /_static/noiser_examples/elastic_noiser_exponential.png
        """
        rng = np.random.default_rng()
        for key in signal.keys():
            for i in range(len(signal[key])):
                max_variance = distributions.get_random_value_from_distribution(self.k)
                variance = (
                    max_variance ** (i / signal.shape[0])
                    if self.exponential
                    else i / signal.shape[0] * max_variance
                )

                if self.invert:
                    variance = max_variance - variance
                signal.at[i, key] += rng.normal(0, variance)
        return signal
