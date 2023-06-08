"""Provides the generator for `PredictionTask` signals."""
import os
import typing
import math

import pandas as pd

from simba_ml.simulation.system_model import system_model_interface
from simba_ml.simulation import random_generator


class SteadyStateGenerator:
    """Defines how to generate signals from a PredictionTask."""

    def __init__(self, sm: system_model_interface.SystemModelInterface):
        """Initializes the `PredictionTaskBuilder`.

        Args:
            sm: A `SystemModel`, for which the signals should be built.
        """
        self.sm = sm

    def _is_similar(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Checks if two series are similar.

        Args:
            series1: The first series.
            series2: The second series.

        Returns:
            True if the series are similar, False otherwise.

        Raises:
            ValueError: if the series have different lengths.
        """
        if len(series1) != len(series2):
            raise ValueError("Series have different lengths.")

        return all(
            math.isclose(series1[i], series2[i], rel_tol=1e-05)
            for i in range(len(series1))
        )

    def __check_if_signal_has_steady_state(self, signal: pd.DataFrame) -> bool:
        """Checks if a signal has a steady state.

        Args:
            signal: The signal.

        Returns:
            True if the signal has a steady state, False otherwise.
        """
        return self._is_similar(signal.iloc[-1], signal.iloc[-2])

    def __add_parameters_to_table(
        self, start_values: dict[str, typing.Any], signals_df: pd.DataFrame
    ) -> pd.DataFrame:
        for i in range(len(start_values["timestamps"])):
            for name, kinetic_parameter in self.sm.kinetic_parameters.items():
                signals_df[
                    "kinetic_parameter_" + name
                ] = kinetic_parameter.get_at_timestamp(i, 0)
            for species in self.sm.specieses.values():
                signals_df[species.name + "_start_value"] = float(
                    start_values["specieses"][species.name][i]
                )
        for species_name, species in self.sm.specieses.items():
            if not species.contained_in_output:
                signals_df.drop(columns=[species_name], inplace=True)

        return signals_df

    def __generate_steady_state(
        self, start_values: dict[str, typing.Any], sample_id: int
    ) -> pd.Series:
        """Simulates a prediction task and tests, if it has a steady state.

        Args:
            start_values: The start values.
            sample_id: The sample id.

        Returns:
            The steady state.

        Raises:
            ValueError: if the generated signal has no steady state.
        """
        clean_signal = self.sm.get_clean_signal(
            start_values=start_values, sample_id=sample_id
        )

        for key in start_values["specieses"]:
            start_values["specieses"][key][sample_id] = clean_signal[key].iloc[-1]

        clean_signal = self.sm.get_clean_signal(
            start_values=start_values, sample_id=sample_id, deriv_noised=False
        )

        if not self.__check_if_signal_has_steady_state(clean_signal):
            raise ValueError("Signal has no steady state.")

        pertubation_std = 0.01
        for key in start_values["specieses"]:
            start_values["specieses"][key][sample_id] = clean_signal[key].iloc[
                -1
            ] * random_generator.get_rng().normal(1, pertubation_std)

        pertubated_signal = self.sm.get_clean_signal(
            start_values=start_values, sample_id=sample_id
        )
        if self.__check_if_signal_has_steady_state(pertubated_signal):
            self.sm.apply_noisifier(clean_signal)
            return clean_signal.iloc[-1]
        raise ValueError("Signal has no steady state.")

    def generate_signals(self, n: int = 100) -> pd.DataFrame:
        """Generates signals.

        Args:
            n: The number of samples.

        Returns:
            A list of (noised and sparsed) signals.

        Raises:
            ValueError: if a signal has no steady state.

        Note:
            This method will probably not work for prediction tasks
            using a derivative noiser.
        """
        start_values = self.sm.sample_start_values_from_hypercube(n)
        signals = [self.__generate_steady_state(start_values, i) for i in range(n)]
        signals_df = pd.DataFrame(signals)

        signals_df = self.__add_parameters_to_table(start_values, signals_df)
        return signals_df.reset_index(drop=True)

    def generate_csvs(self, n: int = 1, save_dir: str = "./data/") -> None:
        """Generates signals and saves them to csv.

        Args:
            n: The number of samples.
            save_dir: The directory to save the csv files.

        Raises:
            ValueError: if a signal has no steady state.
        """
        signals = self.generate_signals(n)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        signals.to_csv(f"{save_dir}{self.sm.name}_steady_states.csv", index=False)
