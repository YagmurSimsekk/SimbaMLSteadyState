from simba_ml.simulation import noisers
from simba_ml.simulation import system_model
from simba_ml.simulation.generators import time_series_generator
from simba_ml.simulation.species import Species
from simba_ml.simulation import constraints
from simba_ml.simulation import distributions


def test_constraint_truncates():
    timestamps = 200
    sm = create_example_system_model(timestamps)
    generator = time_series_generator.TimeSeriesGenerator(sm)
    signal = generator.generate_signal()
    assert signal["y1"].min() >= 1000
    assert signal["y2"].max() <= 0


def create_example_system_model(number_of_timestamps=200):
    name = "Constant function"
    timestamps = distributions.Constant(number_of_timestamps)

    specieses = [
        Species("y1", distributions.Constant(500), min_value=1000),
        Species("y2", distributions.Constant(500), max_value=0),
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
    sm = constraints.species_value_truncator.SpeciesValueTruncator(
        system_model.SystemModel(
            name,
            specieses,
            kinetic_parameters,
            deriv=deriv,
            noiser=noiser,
            timestamps=timestamps,
        )
    )
    return sm
