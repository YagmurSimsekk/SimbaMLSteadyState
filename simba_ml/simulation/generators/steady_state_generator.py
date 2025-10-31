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

    def _is_similar(self, series1: pd.Series, series2: pd.Series,
                    abs_tol: float = 1e-8, rel_tol: float = 1e-4) -> bool:
        """Checks if two series are similar using combined absolute and relative tolerance.

        Uses a robust tolerance check that works for both small values (near zero)
        and large values: |a - b| <= abs_tol OR |a - b| / max(|a|, |b|) <= rel_tol

        Args:
            series1: The first series.
            series2: The second series.
            abs_tol: Absolute tolerance threshold (default 1e-8, for near-zero values)
            rel_tol: Relative tolerance threshold (default 1e-4, for proportional differences)

        Returns:
            True if all values are similar within tolerance, False otherwise.

        Raises:
            ValueError: if the series have different lengths.
        """
        if len(series1) != len(series2):
            raise ValueError("Series have different lengths.")

        for i in range(len(series1)):
            val1 = series1.iloc[i]
            val2 = series2.iloc[i]

            # Check absolute tolerance first (good for values near zero)
            abs_diff = abs(val1 - val2)
            if abs_diff <= abs_tol:
                continue

            # Check relative tolerance (good for large values)
            max_abs = max(abs(val1), abs(val2))
            if max_abs > 0:
                rel_diff = abs_diff / max_abs
                if rel_diff <= rel_tol:
                    continue

            # Neither tolerance satisfied
            return False

        return True

    def __check_if_signal_has_steady_state(self, signal: pd.DataFrame) -> bool:
        """Checks if a signal has reached a steady state.

        Verifies convergence by checking that the last 10 time points form a stable
        plateau - i.e., all consecutive pairs are similar. This is more robust than
        comparing just 2 points, as it rules out accidental similarity.

        Args:
            signal: The signal (time series DataFrame).

        Returns:
            True if the signal has reached steady state (last 10 points are stable),
            False otherwise.
        """
        # Need at least 11 rows to check last 10 points
        if len(signal) < 11:
            return False

        # Get the last 10 time points
        last_10_points = signal.iloc[-10:]

        # Check that all consecutive pairs are similar
        # This ensures stability across multiple timesteps, not just 2
        for i in range(len(last_10_points) - 1):
            if not self._is_similar(last_10_points.iloc[i], last_10_points.iloc[i+1]):
                return False

        return True

    def __add_parameters_to_table(
        self, start_values: dict[str, typing.Any], signals_df: pd.DataFrame
    ) -> pd.DataFrame:
        # Add kinetic parameter columns for each sample
        for name, kinetic_parameter in self.sm.kinetic_parameters.items():
            param_values = []
            for i in range(len(start_values["timestamps"])):
                param_values.append(kinetic_parameter.get_at_timestamp(i, 0))
            signals_df["kinetic_parameter_" + name] = param_values

        # Add initial value columns for each sample
        for species in self.sm.specieses.values():
            start_values_list = []
            for i in range(len(start_values["timestamps"])):
                start_values_list.append(float(start_values["specieses"][species.name][i]))
            signals_df[species.name + "_start_value"] = start_values_list
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
        # Try LSODA time integration first
        try:
            clean_signal = self.sm.get_clean_signal(
                start_values=start_values, sample_id=sample_id
            )

            for key in start_values["specieses"]:
                start_values["specieses"][key][sample_id] = clean_signal[key].iloc[-1]

            clean_signal = self.sm.get_clean_signal(
                start_values=start_values, sample_id=sample_id, deriv_noised=False
            )

            if self.__check_if_signal_has_steady_state(clean_signal):
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
        except Exception:
            pass  # Fall through to bounded solver

        # Fallback to bounded solver if LSODA fails
        try:
            from simba_ml.simulation import steady_state_solvers

            # Get kinetic parameters for this sample
            kinetic_params = {
                name: param.get_at_timestamp(sample_id, 0.0)
                for name, param in self.sm.kinetic_parameters.items()
            }

            # Get initial guess from species start values
            dynamic_species = [sp for sp, obj in self.sm.specieses.items() if obj.contained_in_output]
            initial_guess = [start_values["specieses"][sp_name][sample_id] for sp_name in dynamic_species]

            # Use bounded solver
            solution, success, message = steady_state_solvers.find_steady_state(
                deriv_func=self.sm.deriv,
                initial_guess=initial_guess,
                kinetic_params=kinetic_params,
                solver_type='bounded'
            )

            if success:
                # Create result series from solution
                result_dict = {sp_name: solution[i] for i, sp_name in enumerate(dynamic_species)}
                return pd.Series(result_dict)
        except Exception:
            pass

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
