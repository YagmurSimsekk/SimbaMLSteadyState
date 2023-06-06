import pytest
from simba_ml.simulation import distributions
from simba_ml.simulation import random_generator


def test_continuous_uniform_distribution_one_value():
    """Check that the `ContinuousUniformDistribution` returns a scalar inside
    of the given range when calling `ContinuousUniformDistribution.get_random_value`."""
    my_min = 3
    my_max = 5

    initial_condition = distributions.ContinuousUniformDistribution(my_min, my_max)
    assert (
        my_min
        <= distributions.get_random_value_from_distribution(initial_condition)
        <= my_max
    )


def test_continuous_uniform_distribution_multiple_values():
    """Check that the `ContinuousUniformDistribution` returns an array containing
    values from the given range when calling `get_multiple_random_values`."""
    my_min = 3
    my_max = 5

    n = 5
    initial_condition = distributions.ContinuousUniformDistribution(my_min, my_max)
    result = initial_condition.get_random_values(n)
    assert result == [
        3.5395734275277406,
        3.081947047872389,
        3.033055271057058,
        4.626540478400544,
        4.825511154555444,
    ]


def test_continuous_uniform_distribution_throws_error_when_min_value_not_float_or_int():
    """Check that the `ContinuousUniformDistribution` throws a TypeError when min_value
    is not float or int."""
    with pytest.raises(TypeError):
        distributions.ContinuousUniformDistribution("1", -1)


def test_continuous_uniform_distribution_throws_error_when_max_value_not_float_or_int():
    """Check that the `ContinuousUniformDistribution` throws a TypeError when max_value
    is not float or int."""
    with pytest.raises(TypeError):
        distributions.ContinuousUniformDistribution(1, "3")


def test_continuous_get_hypercube_samples():
    initial_condition = distributions.ContinuousUniformDistribution(3, 6)
    result = sorted(initial_condition.get_samples_from_hypercube(3))
    assert len(result) == 3
    assert 3 <= result[0] <= 4
    assert 4 <= result[1] <= 5
    assert 5 <= result[2] <= 6
