r"""Introduces the problem of deriving sinus and cosinus from each other.

Trigonometry models the \\(sin\\) and \\(cos\\) functions and the additive inversions.
"""

from simba_ml.simulation import system_model
from simba_ml.simulation import species
from simba_ml.simulation import noisers
from simba_ml.simulation import derivative_noiser
from simba_ml.simulation import distributions
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module

name = "Trigonometry"
timestamps = distributions.Constant(200)

specieses = [
    species.Species("sin", distributions.Constant(0)),
    species.Species("cos", distributions.Constant(1)),
    species.Species("-sin", distributions.Constant(0)),
    species.Species("-cos", distributions.Constant(-1)),
]

kinetic_parameters: dict[str, kinetic_parameters_module.KineticParameter[float]] = {}


def deriv(
    _t: float, y: list[float], _arguments: dict[str, float]
) -> tuple[float, float, float, float]:
    """Defines the derivative of the function at the point _.

    Args:
        y: Current y vector.

    Returns:
        Tuple[float, float, float, float]
    """
    sin, cos, minsin, mincos = y
    dsin_dt = cos / 100
    dcos_dt = minsin / 100
    dminsin_dt = mincos / 100
    dmincos_dt = sin / 100
    return dsin_dt, dcos_dt, dminsin_dt, dmincos_dt


deriv_noiser = derivative_noiser.AdditiveDerivNoiser(
    distributions.NormalDistribution(0, 0.05)
)

noiser1 = noisers.AdditiveNoiser(distributions.NormalDistribution(0, 0.05))
noiser2 = noisers.multiplicative_noiser.MultiplicativeNoiser(
    distributions.NormalDistribution(1, 0.05)
)
noiser = noisers.sequential_noiser.SequentialNoiser([noiser1, noiser2])
sm = system_model.SystemModel(
    name,
    specieses,
    kinetic_parameters,
    deriv=deriv,
    noiser=noiser,
    timestamps=timestamps,
    deriv_noiser=deriv_noiser,
)
