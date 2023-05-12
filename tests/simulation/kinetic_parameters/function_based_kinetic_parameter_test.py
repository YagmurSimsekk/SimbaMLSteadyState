import itertools

from simba_ml.simulation import kinetic_parameters


def test_function_based_kinetic_parameter():
    def function(_t):
        return 1

    kinetic_parameter = kinetic_parameters.FunctionBasedKineticParameter(function)
    kinetic_parameter.prepare_samples(10)
    for run, t in itertools.product(range(10), range(5)):
        assert kinetic_parameter.get_at_timestamp(run, t) == 1


def test_function_based_kinetic_parameter_with_uniform_distribution():
    def function(_t):
        return t * 2

    kinetic_parameter = kinetic_parameters.FunctionBasedKineticParameter(function)
    kinetic_parameter.prepare_samples(10)
    for run, t in itertools.product(range(10), range(1, 5)):
        assert kinetic_parameter.get_at_timestamp(run, t) == t * 2
