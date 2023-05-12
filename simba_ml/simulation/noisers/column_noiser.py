"""Defines the `ColumnNoiser`."""

import pandas as pd

from simba_ml.simulation.noisers import noiser


class ColumnNoiser(noiser.Noiser):
    """The `ColumnNoiser` applies a different `Noiser` to every column of the signal.

    Columns without a provided `Noiser` will be skipped.

    Attributes:
        noisers: A dictionary containing the column names as keys
            and the corresponding `Noiser` as values.

    Example:
        >>> import pandas as pd
        >>> from simba_ml.simulation import distributions
        >>> from simba_ml.simulation import noisers
        >>> clean_signal = pd.DataFrame.from_dict({
        ...     "A": [75, 52, 68],
        ...     "B": [33, 96, 64],
        ...     "C": [57, 5, 13],
        ...     "D": [65, 4, 51],})
        >>> clean_signal
            A   B   C   D
        0  75  33  57  65
        1  52  96   5   4
        2  68  64  13  51
        >>> col_noisers = {
        ...     "A": noisers.AdditiveNoiser(distributions.Constant(2)),
        ...     "B": noisers.MultiplicativeNoiser(distributions.Constant(2)),
        ...     "D": noisers.NoNoiser()}
        >>> noiser = noisers.ColumnNoiser(col_noisers)
        >>> noiser.noisify(clean_signal)
            A    B   C   D
        0  77   66  57  65
        1  54  192   5   4
        2  70  128  13  51
    """

    def __init__(self, noisers: dict[str, noiser.Noiser]):
        """Inits `ColumnNoiser` with the provided params.

        Args:
            noisers: A dictionary containing the column names as keys
                and the corresponding `Noiser` as values.
        """
        self.noisers = noisers

    def noisify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies noise to each column individually.

        Args:
            signal: The input data.

        Returns:
            pd.DataFrame
        """
        for name in signal.columns:
            if name in self.noisers:
                noiser_ = self.noisers[name]
                signal[name] = noiser_.noisify(signal[name])
        return signal
