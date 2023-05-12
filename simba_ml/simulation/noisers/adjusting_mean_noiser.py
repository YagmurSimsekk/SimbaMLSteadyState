"""Defines the `ElasticNoiser`."""

import pandas as pd

from simba_ml.simulation.noisers import noiser
from simba_ml.simulation import distributions


class AdjustingMeanNoiser(noiser.Noiser):
    """The `AdjustingMeanNoiser` adjusts every value to the mean it's column.

    Does a linear interpolation for each cell between it's value and it's columns mean.

    With c_{i;j} being an arbitrary cell in column j and m(j)
        being the mean of the column j, the following equation holds.

    .. math::
        c_{i;j} = c_{i_j} + weight * (m(j) + c_{i;j})

    Attributes:
        weight : Weight of the mean in the linear interpolation.

    Example:
        >>> import pandas as pd
        >>> from simba_ml.simulation import distributions
        >>> from simba_ml.simulation.noisers.\
                adjusting_mean_noiser import AdjustingMeanNoiser
        >>> clean_signal = pd.DataFrame([0, 10, 50])
        >>> clean_signal
            0
        0   0
        1  10
        2  50
        >>> noisers = AdjustingMeanNoiser(distributions.Constant(0.5))
        >>> noisers.noisify(clean_signal)
            0
        0  10
        1  15
        2  35
        >>> noisers = AdjustingMeanNoiser(distributions.Constant(0.2))
        >>> noisers.noisify(clean_signal)
            0
        0   4
        1  12
        2  44
    """

    def __init__(self, weight: distributions.Distribution[float]):
        """Inits the `AdjustingMeanNoiser`.

        Args:
            weight: Interpolation Factor.
                When weight is 1, then the output equals the mean of the column.
                If weight is 0.5 the output equals the mean of the column's mean
                and the cell's value.
        """
        self.weight = weight

    def noisify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies noise to the provided signal.

        Args:
            signal: The input data.

        Returns:
            pd.DataFrame
        """
        noised_signal = signal.copy(deep=True)
        for key in signal.keys():
            mean = signal[key].mean()
            for i in range(len(signal[key])):
                noised_signal.at[i, key] = signal.at[
                    i, key
                ] + distributions.get_random_value_from_distribution(self.weight) * (
                    mean - signal.at[i, key]
                )
        return noised_signal
