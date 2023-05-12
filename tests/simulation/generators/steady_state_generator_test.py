import os
import shutil

import pytest

import pandas as pd

from simba_ml.simulation import noisers
from simba_ml.simulation import distributions
from simba_ml.simulation import system_model
from simba_ml.simulation.species import Species
from simba_ml.simulation import generators
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module


def test_is_similar_thows_exception_if_series_have_different_length():
    sm = create_example_system_model()
    generator = generators.SteadyStateGenerator(sm)
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([1, 2, 3, 4])
    with pytest.raises(ValueError):
        generator._is_similar(s1, s2)  # pylint: disable=protected-access


def test_throws_value_error_if_no_steady_state_after_pertubation():
    sm = create_system_model_with_unstable_limit()
    generator = generators.SteadyStateGenerator(sm)
    with pytest.raises(ValueError):
        generator.generate_signals(1)


def test_returns_correct_table():
    sm = create_example_system_model()
    generator = generators.SteadyStateGenerator(sm)
    signals = generator.generate_signals(1)
    assert isinstance(signals, pd.DataFrame)
    expected = pd.DataFrame(
        {
            "y1": [0.0],
            "kinetic_parameter_sledge": [0.0],
            "y1_start_value": [0.0],
            "y2_start_value": [0.0],
        }
    )
    assert signals.equals(expected)


def test_generates_correct_csv_if_out_dir_not_exists():
    save_dir = "./steady_state_tmp/"
    sm = create_example_system_model()
    generator = generators.SteadyStateGenerator(sm)
    generator.generate_csvs(1, save_dir)
    signals = pd.read_csv(save_dir + sm.name + "_steady_states.csv")
    assert isinstance(signals, pd.DataFrame)
    expected = pd.DataFrame(
        {
            "y1": [0.0],
            "kinetic_parameter_sledge": [0.0],
            "y1_start_value": [0.0],
            "y2_start_value": [0.0],
        }
    )
    assert signals.equals(expected)
    shutil.rmtree(save_dir)


def test_generates_correct_csv_if_out_dir_exists():
    save_dir = "./steady_state_tmp/"
    os.mkdir(save_dir)
    sm = create_example_system_model()
    generator = generators.SteadyStateGenerator(sm)
    generator.generate_csvs(1, save_dir)
    signals = pd.read_csv(save_dir + sm.name + "_steady_states.csv")
    assert isinstance(signals, pd.DataFrame)
    expected = pd.DataFrame(
        {
            "y1": [0.0],
            "kinetic_parameter_sledge": [0.0],
            "y1_start_value": [0.0],
            "y2_start_value": [0.0],
        }
    )
    assert signals.equals(expected)
    shutil.rmtree(save_dir)


def test_raises_value_error_if_steady_state_not_exists():
    sm = create_example_system_model(sledge=1)
    generator = generators.SteadyStateGenerator(sm)
    with pytest.raises(ValueError):
        generator.generate_signals(1)


def create_example_system_model(number_of_timestamps=200, sledge=0.0):
    name = "Linear function"
    timestamps = distributions.Constant(number_of_timestamps)

    specieses = [
        Species("y1", distributions.Constant(0.0)),
        Species("y2", distributions.Constant(0.0), contained_in_output=False),
    ]

    kinetic_parameters = {
        "sledge": kinetic_parameters_module.ConstantKineticParameter(
            distributions.Constant(sledge)
        )
    }

    def deriv(_t, _y, arguments):
        """Derivative of the function at the point _t.

        Returns:
            List[float]
        """
        return [arguments["sledge"], 0.0]

    noiser = noisers.additive_noiser.AdditiveNoiser(
        distributions.NormalDistribution(1, 0)
    )
    sm = system_model.SystemModel(
        name,
        specieses,
        kinetic_parameters,
        deriv=deriv,
        noiser=noiser,
        timestamps=timestamps,
    )
    return sm


def create_system_model_with_unstable_limit():
    name = "Unstable limit"
    timestamps = distributions.Constant(3)

    specieses = [Species("y", distributions.Constant(1.0))]

    kinetic_parameters = {}

    def deriv(_t, y, _arguments):
        return y[0] - 1

    sm = system_model.SystemModel(
        name, specieses, kinetic_parameters, deriv=deriv, timestamps=timestamps
    )
    return sm
