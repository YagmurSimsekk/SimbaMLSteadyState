"""Provides a Sparsifier which removes constant suffixes."""
import pandas as pd

from simba_ml import error_handler
from simba_ml.simulation.sparsifier import sparsifier


class ConstantSuffixRemover(sparsifier.Sparsifier):
    """A sparsifier which removes constant suffixes.

    Attributes:
        frac: The relative amount of samples to keep from the incoming signal.
        0 <= frac <= 1.

    Raises:
        TypeError: If frac is not a float.
        ValueError: If frac not in the interval [0, 1].
    """

    def __init__(
        self, n: int = 10, epsilon: float = 0.001, mode: str = "relative"
    ) -> None:
        """Inits the `ConstantSuffixRemover`.

        Args:
            n: The minimum length of the suffix to remove.
            epsilon: Relative size of the epsilon-neighbourhood of the suffix.
            mode: Wether the espilon neighborhood is "relative" or "absolute".

        Raises:
            TypeError: If n is not an int.
            ValueError: If n is negative or epsilon is negative.
            TypeError: If epsilon is neither float nor int.
            ValueError: If mode is neither "absolute" nor "relative".
        """
        error_handler.confirm_param_is_float_or_int(epsilon, param_name="epsilon")
        error_handler.confirm_param_is_int(param=n, param_name="n")
        error_handler.confirm_number_is_greater_or_equal_to_0(number=n, param_name="n")
        error_handler.confirm_number_is_greater_or_equal_to_0(
            epsilon, param_name="epsilon"
        )

        if mode not in ["relative", "absolute"]:
            raise ValueError("mode must be either 'relative' or 'absolute'")

        self.n = n
        self.epsilon = epsilon
        self.mode = mode

    def __check_if_suffix_is_constant(
        self, suffix_dict: dict[str, tuple[float, float]]
    ) -> bool:
        """Checks if the suffix_dict is constant.

        Args:
            suffix_dict: The suffix to check. A dict with the column names as keys
                and the min and max value of the suffix as values.

        Returns:
            True if the suffix is constant, False otherwise.
        """
        if self.mode == "absolute":
            for key in suffix_dict:
                mean = (suffix_dict[key][0] + suffix_dict[key][1]) / 2
                if (
                    mean - self.epsilon > suffix_dict[key][0]
                    or mean + self.epsilon < suffix_dict[key][1]
                ):
                    return False

        if self.mode == "relative":
            for key in suffix_dict:
                mean = (suffix_dict[key][0] + suffix_dict[key][1]) / 2
                if (
                    mean * (1 - self.epsilon) > suffix_dict[key][0]
                    or mean * (1 + self.epsilon) < suffix_dict[key][1]
                ):
                    return False

        return True

    def __get_max_n(self, signal: pd.DataFrame) -> int:
        extrems = {
            key: (
                signal[key].iloc[-self.n : -1].min(),
                signal[key].iloc[-self.n : -1].max(),
            )
            for key in signal.columns
        }
        if not self.__check_if_suffix_is_constant(extrems):
            return 0
        min_n = 0
        max_n = signal.shape[0] - self.n - 1
        while min_n < max_n:
            n = (min_n + max_n) // 2
            extrems = {
                key: (signal[key].iloc[n:].min(), signal[key].iloc[n:].max())
                for key in signal.columns
            }
            if self.__check_if_suffix_is_constant(extrems):
                max_n = n - 1
            else:
                min_n = n + 1
        return min_n

    def sparsify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Sparsifies the signal by constant suffixes.

        Args:
            signal: The signal to sparsify.

        Returns:
            The sparsified signal.
        """
        max_n = self.__get_max_n(signal)
        return signal if max_n == 0 else signal.iloc[: -self.__get_max_n(signal) - 1]
