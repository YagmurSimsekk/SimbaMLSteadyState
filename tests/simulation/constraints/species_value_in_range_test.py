import pytest

import pandas as pd

from simba_ml.simulation import noisers
from simba_ml.simulation import system_model
from simba_ml.simulation.generators.time_series_generator import TimeSeriesGenerator
from simba_ml.simulation.species import Species
from simba_ml.simulation import constraints
from simba_ml.simulation import distributions


def test_constraint_succeeds_if_correct_signal_is_created():
    days = 200
    sm = create_example_system_model(days)
    noiser = noisers.multiplicative_noiser.MultiplicativeNoiser(
        distributions.Constant(1)
    )
    sm.sm.noiser = noiser
    generator = TimeSeriesGenerator(sm)
    signal = generator.generate_signal()
    assert isinstance(signal, pd.DataFrame)
    assert signal.shape[0] == days
    assert signal["y1"].min() >= 1


def test_constraint_throws_error_if_correct_signal_is_created():
    timestamps = 2
    sm = create_example_system_model(timestamps)
    noiser = noisers.multiplicative_noiser.MultiplicativeNoiser(
        distributions.Constant(0)
    )
    sm.sm.noiser = noiser
    generator = TimeSeriesGenerator(sm)
    with pytest.raises(constraints.species_value_in_range.MaxRetriesReachedError):
        generator.generate_signal()


def create_example_system_model(number_of_timestamps=200):
    name = "Constant function"
    timestamps = distributions.Constant(number_of_timestamps)

    specieses = [
        Species("y1", distributions.Constant(500), min_value=1),
        Species("y2", distributions.Constant(500)),
    ]

    kinetic_parameters = {}

    def deriv(_t, _y, _arguments):
        """Derivative of the function at the point _t.

        Returns:
            List[float]
        """
        return [0, 0]

    noiser = noisers.additive_noiser.AdditiveNoiser(
        distributions.NormalDistribution(1, 100)
    )
    sm = constraints.species_value_in_range.KeepSpeciesRange(
        system_model.SystemModel(
            name,
            specieses,
            kinetic_parameters,
            deriv=deriv,
            noiser=noiser,
            timestamps=timestamps,
        ),
        max_retries=2,
    )
    return sm
