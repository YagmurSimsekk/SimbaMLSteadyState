"""Provides the generator for `PredictionTask` signals."""

import os

import pandas as pd

from simba_ml.simulation.system_model import system_model_interface


class TimeSeriesGenerator:
    """Defines how to generate signals from a PredictionTask."""

    def __init__(self, sm: system_model_interface.SystemModelInterface):
        """Initializes the `PredictionTaskBuilder`.

        Args:
            sm: A `SystemModel`, for which the signals should be built.
        """
        self.sm = sm

    def generate_csv(self, save_dir: str = "./data/") -> None:
        """Generates and saves a signal as csv-file.

        Args:
            save_dir: The name of the directory to save the data.
        """
        self.generate_csvs(n=1, save_dir=save_dir)

    def generate_csvs(self, n: int = 1, save_dir: str = "./data/") -> None:
        """Generates and saves signals as csv-files.

        Args:
            n: The number of csvs that will be generated.
            save_dir: The name of the directory to save the data.
        """
        signals = self.generate_signals(n)
        for i, signal in enumerate(signals):
            self.save_signal(signal, save_dir, save_name=str(i))

    def save_signal(
        self, signal: pd.DataFrame, save_dir: str = "./data/", save_name: str = "0"
    ) -> None:
        """Saves a generated signal as csv-file.

        Args:
            signal(pd.DataFrame): Signal.
            save_dir(str): The name of the directory to save the data.
                Default is './data/'.
            save_name(str): Suffix of the filename. Default is '0'.
        """
        # Create save_dir folder if nonexistent
        os.makedirs(save_dir, exist_ok=True)
        save_dir = save_dir if save_dir[-1] == "/" else f"{save_dir}/"

        signal.to_csv(f"{save_dir}{self.sm.name}_{save_name}.csv", index=False)

    def generate_signal(self) -> pd.DataFrame:
        """Generates a signal.

        Returns:
            A (noised and sparsed) signal.
        """
        return self.generate_signals(1)[0]

    def generate_signals(self, n: int = 100) -> list[pd.DataFrame]:
        """Generates signals.

        Args:
            n: The number of samples.

        Returns:
            A list of (noised and sparsed) signals.

        Raises:
            ValueError: If the method is not 'hypercube' or 'random'.
        """
        start_values = self.sm.sample_start_values_from_hypercube(n)
        signals = []
        for i in range(n):
            clean_signal = self.sm.get_clean_signal(
                start_values=start_values, sample_id=i
            )
            noised_signal = self.sm.apply_noisifier(clean_signal)
            signals.append(self.sm.apply_sparsifier(noised_signal))
        return signals
