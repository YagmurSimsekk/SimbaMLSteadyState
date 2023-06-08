import pandas as pd

from simba_ml.simulation import distributions
from simba_ml.simulation import species
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module
from simba_ml.simulation import noisers
from simba_ml.simulation import system_model
from simba_ml.simulation import random_generator
from simba_ml.simulation import generators


def list_of_dataframes_similar(list1, list2, max_difference=0.00001):
    if not len(list1) == len(list2):
        return False
    for i in range(len(list1)):
        if not (
            list1[i].to_numpy().reshape(-1) - list2[i].to_numpy().reshape(-1)
            <= max_difference
        ).all():
            return False
    return True


def test_seeding():
    sm = create_example_system_model()
    generator = generators.TimeSeriesGenerator(sm)
    random_generator.set_seed(42)
    result = generator.generate_signals(2)
    expected_result = [
        pd.DataFrame(
            [
                [9.15250908],
                [7.37045458],
                [5.54378722],
                [8.88790029],
                [6.5422931],
                [7.54977531],
                [5.26495249],
                [5.22911248],
                [6.61273201],
                [14.05146558],
            ],
            columns=["y"],
        ),
        pd.DataFrame(
            [
                [4.981558],
                [16.183999],
                [3.526283],
                [4.634122],
                [7.221429],
                [7.757346],
                [14.852047],
                [9.979742],
                [6.312557],
                [16.227660],
            ],
            columns=["y"],
        ),
    ]
    assert list_of_dataframes_similar(result, expected_result)
    random_generator.set_seed(42)
    result2 = generator.generate_signals(2)
    assert list_of_dataframes_similar(result, result2, 0)
    random_generator.set_seed(43)
    result3 = generator.generate_signals(2)
    assert not list_of_dataframes_similar(result, result3, 1)


def create_example_system_model():
    name = "Constant function"
    timestamps = distributions.Constant(10)

    specieses = [species.Species("y", distributions.Constant(0))]

    kinetic_parameters = {
        "Useless": kinetic_parameters_module.ConstantKineticParameter(
            distributions.Constant(1)
        )
    }

    def deriv(_t, _y, _arguments):
        """Derivative of the function at the point _t.

        Returns:
            List[float]
        """
        return [0]

    noiser = noisers.SequentialNoiser(
        [
            noisers.additive_noiser.AdditiveNoiser(
                distributions.BetaDistribution(1, 1)
            ),
            noisers.AdditiveNoiser(distributions.ContinuousUniformDistribution(1, 2)),
            noisers.additive_noiser.AdditiveNoiser(
                distributions.LogNormalDistribution(1, 1)
            ),
            noisers.AdditiveNoiser(distributions.NormalDistribution(0, 1)),
            noisers.additive_noiser.AdditiveNoiser(
                distributions.VectorDistribution([1, 2, 3])
            ),
        ]
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
