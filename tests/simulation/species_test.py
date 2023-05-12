import pytest

from simba_ml.simulation import distributions
from simba_ml.simulation.species import Species


def test_species_throws_error_if_min_greater_max():
    with pytest.raises(ValueError):
        Species("Hallo", distributions.Constant(1), min_value=1000, max_value=1)


def test_to_string():
    s = Species("Hallo", distributions.Constant(1), min_value=0, max_value=1)
    assert str(s) == "Species(name=Hallo)"
