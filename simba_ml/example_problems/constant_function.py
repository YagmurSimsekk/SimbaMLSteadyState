"""Introduces the problem of deriving a constant function."""

from simba_ml.simulation import system_model
from simba_ml.simulation import species
from simba_ml.simulation import noisers
from simba_ml.simulation import distributions
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module


name = "Constant function"
timestamps = distributions.Constant(200)

specieses = [species.Species("y", distributions.ContinuousUniformDistribution(-10, 10))]

kinetic_parameters: dict[str, kinetic_parameters_module.KineticParameter[float]] = {}


def deriv(_t: float, _y: list[float], _arguments: dict[str, float]) -> tuple[float]:
    """Defines the derivative of the function at the point _.

    Returns:
        Tuple[float]
    """
    return (0,)


noiser = noisers.ElasticNoiser(
    distributions.Constant(100), invert=True, exponential=True
)
sm = system_model.SystemModel(
    name,
    specieses,
    kinetic_parameters,
    deriv=deriv,
    noiser=noiser,
    timestamps=timestamps,
)
