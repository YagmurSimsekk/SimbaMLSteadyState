# pylint: disable=line-too-long
r"""Introduces the epidemiological SIRD model.

SIRD is an extension of
[SIR](https://medium.com/@shaliniharkar/sir-model-for-spread-of-disease-the-differential-equation-model-7e441e8636ab)
where infected people will die with a probability of \\(\\delta\\) and
recovered people will become susceptible with a probability of \\(\\epsilon\\).
"""
# pylint: enable=line-too-long

from simba_ml.simulation import system_model
from simba_ml.simulation import species
from simba_ml.simulation import noisers
from simba_ml.simulation import distributions
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module


name = "SIRD"
specieses = [
    species.Species("Suspectible", distributions.Constant(999)),
    species.Species("Infected", distributions.VectorDistribution([10, 500, 1000])),
    species.Species("Recovered", distributions.LogNormalDistribution(2, 0.1)),
    species.Species("Dead", distributions.Constant(0)),
]

kinetic_parameters: dict[str, kinetic_parameters_module.KineticParameter[float]] = {
    "beta": kinetic_parameters_module.ConstantKineticParameter(
        distributions.ContinuousUniformDistribution(0.1, 0.3)
    ),
    "gamma": kinetic_parameters_module.ConstantKineticParameter(
        distributions.Constant(0.04)
    ),
    "delta": kinetic_parameters_module.ConstantKineticParameter(
        distributions.Constant(0.04)
    ),
    "epsilon": kinetic_parameters_module.ConstantKineticParameter(
        distributions.Constant(0.04)
    ),
}


def deriv(
    _t: float, y: list[float], arguments: dict[str, float]
) -> tuple[float, float, float, float]:
    """Defines the derivative of the function at the point _.

    Args:
        y: Current y vector.
        arguments: Dictionary of arguments configuring the problem.

    Returns:
        Tuple[float, float, float, float]
    """
    S, I, R, _ = y
    N = sum(y)
    dS_dt = -arguments["beta"] * S * I / N + arguments["epsilon"] * R
    dI_dt = (
        arguments["beta"] * S * I / N - (arguments["gamma"] + arguments["delta"]) * I
    )
    dR_dt = arguments["gamma"] * I - arguments["epsilon"] * R
    dD_dt = arguments["delta"] * I
    return dS_dt, dI_dt, dR_dt, dD_dt


noiser = noisers.AdditiveNoiser(distributions.LogNormalDistribution(0, 2))
sm = system_model.SystemModel(
    name, specieses, kinetic_parameters, deriv=deriv, noiser=noiser
)
