"""Provides the generator for `PredictionTask` signals."""

import typing
import copy
import math

import pandas as pd

from simba_ml.simulation.system_model import system_model_interface
from simba_ml.simulation import noisers
from simba_ml.simulation import kinetic_parameters


class PertubationGenerator:
    """Defines how to generate data for a `PertubationTask`.

    The `PertubationGenerator` generates data for a `PertubationTask`
    by generating a signal for a given system model and check if the signal
    has a steady state.
    The initial values for the species and the kinetic parameters are then pertubed.
    Afterwards the signal is generated again and checked if it has a steady state.
    If each of the signals has steady states, the data is saved
    and a table containing the concrete start values for the species, arguments and
    the according steady-states is returned.
    """

    def __init__(
        self,
        sm: system_model_interface.SystemModelInterface,
        species_start_values_noiser: typing.Optional[noisers.Noiser] = None,
        kinetic_parameters_noiser: typing.Optional[noisers.Noiser] = None,
    ):
        """Initializes the `PertubationGenerator`.

        Note:
            Only the use of a constant_kinetic_parameter is allowed.

        Args:
            sm: The system model.
            species_start_values_noiser: The noiser for the species start values.
            kinetic_parameters_noiser: The noiser for the kinetic parameters.
        """
        self.sm = sm
        self.species_start_values_noiser = (
            species_start_values_noiser or noisers.NoNoiser()
        )
        self.kinetic_parameters_noiser = kinetic_parameters_noiser or noisers.NoNoiser()

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
            or math.isclose(series1[i], series2[i], abs_tol=1e-05)
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
        my_start_values = copy.deepcopy(start_values)
        clean_signal = self.sm.get_clean_signal(
            start_values=my_start_values, sample_id=sample_id
        )
        for key in start_values["specieses"]:
            my_start_values["specieses"][key][sample_id] = clean_signal[key].iloc[-1]

        clean_signal = self.sm.get_clean_signal(
            start_values=my_start_values, sample_id=sample_id, deriv_noised=False
        )

        if not self.__check_if_signal_has_steady_state(clean_signal):
            raise ValueError("Signal has no steady state.")

        return clean_signal.iloc[-1]

    def __noise_parameters(
        self, start_values: dict[str, typing.Any], sample_id: int
    ) -> None:
        """Noises the species start values and the kinetic parameters.

        Args:
            start_values: The start values.
            sample_id: The sample id to noise.

        Raises:
            TypeError: if a kinetic parameter is not of type `ConstantKineticParameter`.
        """
        species_start_values = pd.DataFrame(
            [
                {
                    species: start_values["specieses"][species][sample_id]
                    for species in start_values["specieses"]
                }
            ]
        )
        species_start_values = self.species_start_values_noiser.noisify(
            species_start_values
        )

        for species in start_values["specieses"]:
            start_values["specieses"][species][sample_id] = species_start_values[
                species
            ].iloc[0]

        for kinetic_parameter in self.sm.kinetic_parameters.values():
            arg_df = pd.DataFrame([kinetic_parameter.get_at_timestamp(sample_id, 0)])
            if not isinstance(
                kinetic_parameter, kinetic_parameters.ConstantKineticParameter
            ):
                raise TypeError(
                    "Each kinetic parameter must be of type"
                    "ConstantKineticParameter in this generator."
                )
            kinetic_parameter.set_for_run(
                sample_id, self.kinetic_parameters_noiser.noisify(arg_df)[0].iloc[0]
            )

    def __generate_perturbed_signal(
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
        results = {
            f"unnoised_species_{species}": start_values["specieses"][species][sample_id]
            for species in start_values["specieses"]
        } | {
            f"unnoised_kinetic_parameter_{name}": kinetic_parameter.get_at_timestamp(
                sample_id, 0
            )
            for name, kinetic_parameter in self.sm.kinetic_parameters.items()
        }

        clean_steady_state = self.__generate_steady_state(start_values, sample_id)
        results |= {
            f"unnoised_steady_state_{species}": clean_steady_state[species]
            for species in clean_steady_state.keys()
        }
        self.__noise_parameters(start_values, sample_id)
        perturbed_steady_state = self.__generate_steady_state(start_values, sample_id)
        results |= {
            f"noised_species_{species}": start_values["specieses"][species][sample_id]
            for species in start_values["specieses"]
        }

        results |= {
            f"noised_kinetic_parameter_{name}": kinetic_parameter.get_at_timestamp(
                sample_id, 0
            )
            for name, kinetic_parameter in self.sm.kinetic_parameters.items()
        }

        results |= {
            f"noised_steady_state_{species}": perturbed_steady_state[species]
            for species in perturbed_steady_state.keys()
        }

        return results

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
        signals = [self.__generate_perturbed_signal(start_values, i) for i in range(n)]
        signals_df = pd.DataFrame(signals)
        return signals_df.reset_index(drop=True)
