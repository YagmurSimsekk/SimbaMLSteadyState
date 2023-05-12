import pytest

import pandas as pd

from simba_ml.simulation import generators
from simba_ml.simulation import noisers
from simba_ml.simulation import system_model
from simba_ml.simulation.species import Species
from simba_ml.simulation import distributions
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module


def test_generator_returns_correct_result_when_no_noiser_is_provided():
    sm = get_converging_system_model()
    generator = generators.PertubationGenerator(sm)

    actual = generator.generate_signals(2)
    expected = pd.DataFrame(
        [
            {
                "unnoised_species_y": 1.0,
                "unnoised_kinetic_parameter_sledge": 0.0,
                "unnoised_steady_state_y": 1.0,
                "noised_species_y": 1.0,
                "noised_kinetic_parameter_sledge": 0.0,
                "noised_steady_state_y": 1.0,
            },
            {
                "unnoised_species_y": 1.0,
                "unnoised_kinetic_parameter_sledge": 0.0,
                "unnoised_steady_state_y": 1.0,
                "noised_species_y": 1.0,
                "noised_kinetic_parameter_sledge": 0.0,
                "noised_steady_state_y": 1.0,
            },
        ]
    )
    assert actual.equals(expected)


class DummyNoiser(noisers.Noiser):
    """The dummy noiser is used to test if the noiser is called."""

    def __init__(self):
        self.called = False

    def noisify(self, signal):
        self.called = True
        return signal


def test_generator_returns_correct_result_when_noisers_are_provided():
    sm = get_converging_system_model()
    kinetic_parameters_noiser = DummyNoiser()
    species_noiser = noisers.AdditiveNoiser(distributions.Constant(2.0))
    generator = generators.PertubationGenerator(
        sm,
        kinetic_parameters_noiser=kinetic_parameters_noiser,
        species_start_values_noiser=species_noiser,
    )

    actual = generator.generate_signals(2)
    expected = pd.DataFrame(
        [
            {
                "unnoised_species_y": 1.0,
                "unnoised_kinetic_parameter_sledge": 0.0,
                "unnoised_steady_state_y": 1.0,
                "noised_species_y": 3.0,
                "noised_kinetic_parameter_sledge": 0.0,
                "noised_steady_state_y": 3.0,
            },
            {
                "unnoised_species_y": 1.0,
                "unnoised_kinetic_parameter_sledge": 0.0,
                "unnoised_steady_state_y": 1.0,
                "noised_species_y": 3.0,
                "noised_kinetic_parameter_sledge": 0.0,
                "noised_steady_state_y": 3.0,
            },
        ]
    )

    assert actual.equals(expected)
    assert kinetic_parameters_noiser.called


def test_generator_raises_error_if_there_is_no_steady_state():
    sm = get_non_converging_system_model()
    generator = generators.PertubationGenerator(sm)

    with pytest.raises(ValueError):
        generator.generate_signals(2)


def test_generator_raises_error_if_there_is_no_steady_state_after_noising():
    sm = get_converging_system_model()
    generator = generators.PertubationGenerator(
        sm,
        kinetic_parameters_noiser=noisers.AdditiveNoiser(distributions.Constant(1.0)),
    )

    with pytest.raises(ValueError):
        generator.generate_signals(2)


def test_is_similar_thows_exception_if_series_have_different_length():
    sm = get_converging_system_model()
    generator = generators.PertubationGenerator(sm)
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([1, 2, 3, 4])
    with pytest.raises(ValueError):
        generator._is_similar(s1, s2)  # pylint: disable=protected-access


def test_raises_error_if_non_constant_kinetic_parameter_is_provided():
    sm = get_converging_system_model()
    sm.kinetic_parameters[
        "sledge"
    ] = kinetic_parameters_module.DictBasedKineticParameter({0: 0.0, 10: 0.0})
    generator = generators.PertubationGenerator(sm)

    with pytest.raises(TypeError):
        generator.generate_signals(2)


def get_converging_system_model():
    name = "Converging System Model"
    timestamps = distributions.Constant(100)

    specieses = [Species("y", distributions.Constant(1.0))]

    args = {
        "sledge": kinetic_parameters_module.ConstantKineticParameter(
            distributions.Constant(0.0)
        )
    }

    def deriv(_t, _y, arguments):
        return arguments["sledge"]

    sm = system_model.SystemModel(
        name, specieses, args, deriv=deriv, timestamps=timestamps
    )
    return sm


def get_non_converging_system_model():
    name = "Non Converging System Model"
    timestamps = distributions.Constant(100)

    specieses = [Species("y", distributions.Constant(1.0))]

    args = {
        "sledge": kinetic_parameters_module.ConstantKineticParameter(
            distributions.Constant(1.0)
        )
    }

    def deriv(_t, _y, arguments):
        return arguments["sledge"]

    sm = system_model.SystemModel(
        name, specieses, args, deriv=deriv, timestamps=timestamps
    )
    return sm
