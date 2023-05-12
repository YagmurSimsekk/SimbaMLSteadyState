import pytest

from simba_ml.simulation.kinetic_parameters import dict_based_kinetic_parameter


def test_dict_based_kinetic_parameter_raises_zero_not_set_error():
    known_values = {100: 20}
    with pytest.raises(dict_based_kinetic_parameter.ZeroNotSetError):
        dict_based_kinetic_parameter.DictBasedKineticParameter(known_values)


def test_dict_based_kinetic_parameter():
    known_values = {0: 10, 100: 20}
    kinetic_parameter = dict_based_kinetic_parameter.DictBasedKineticParameter(
        known_values
    )
    kinetic_parameter.prepare_samples(10)
    assert kinetic_parameter.get_at_timestamp(0, 0) == 10
    assert kinetic_parameter.get_at_timestamp(1, 0) == 10
    assert kinetic_parameter.get_at_timestamp(0, 1) == 10
    assert kinetic_parameter.get_at_timestamp(1, 1) == 10
    assert kinetic_parameter.get_at_timestamp(0, 100) == 20
    assert kinetic_parameter.get_at_timestamp(1, 100) == 20
    assert kinetic_parameter.get_at_timestamp(0, 101) == 20
    assert kinetic_parameter.get_at_timestamp(1, 101) == 20
