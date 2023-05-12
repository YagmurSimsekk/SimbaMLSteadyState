r"""Introduces the problem of salt and brine tanks.

Salt and brine tanks models the amount of salt and brine
in a tank with a volume of \\(V\\) and a flow rate of \\(r\\).
"""

from simba_ml.simulation import system_model
from simba_ml.simulation import species
from simba_ml.simulation import noisers
from simba_ml.simulation import distributions
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module

name = "Salt and Brine Tanks"
timestamps = distributions.Constant(5000)
specieses = [
    species.Species("x1", distributions.VectorDistribution([799, 999])),
    species.Species("x2", distributions.Constant(499), contained_in_output=False),
]

kinetic_parameters: dict[str, kinetic_parameters_module.KineticParameter[float]] = {
    "r": kinetic_parameters_module.ConstantKineticParameter(
        distributions.ContinuousUniformDistribution(0.1, 0.3)
    ),
    "V": kinetic_parameters_module.ConstantKineticParameter(
        distributions.VectorDistribution([1, 10, 100, 1000])
    ),
}


def deriv(
    _t: float, y: list[float], arguments: dict[str, float]
) -> tuple[float, float]:
    """Defines the derivative of the function at the point _.

    Args:
        y: Current y vector.
        arguments: Dictionary of arguments configuring the problem.

    Returns:
        Tuple[float, float]
    """
    x1, x2 = y
    dx1_dt = arguments["r"] / arguments["V"] * (x2 - x1)
    dx2_dt = arguments["r"] / arguments["V"] * (x1 - x2)
    return dx1_dt, dx2_dt


noiser = noisers.AdditiveNoiser(distributions.NormalDistribution(0, 20))
sm = system_model.SystemModel(
    name,
    specieses,
    kinetic_parameters,
    deriv=deriv,
    noiser=noiser,
    timestamps=timestamps,
)
