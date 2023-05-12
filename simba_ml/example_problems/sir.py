# pylint: disable=line-too-long
"""Introduces the epidemiological SIR model.

[SIR](https://medium.com/@shaliniharkar/sir-model-for-spread-of-disease-the-differential-equation-model-7e441e8636ab)
is a epidemiological model modeling the spread of deseases.
"""
# pylint: enable=line-too-long

from simba_ml.simulation import system_model
from simba_ml.simulation import species
from simba_ml.simulation import noisers

# pylint: disable=unused-import; Should be used before merging.
from simba_ml.simulation import constraints
from simba_ml.simulation import distributions
from simba_ml.simulation import sparsifier as sparsifier_module
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module

name = "SIR"
specieses = [
    species.Species(
        "Suspectible",
        distributions.Constant(100000),
        contained_in_output=False,
        min_value=0,
    ),
    species.Species("Infected", distributions.VectorDistribution([1]), min_value=0),
    species.Species(
        "Recovered",
        distributions.LogNormalDistribution(2, 1),
        contained_in_output=False,
        min_value=0,
    ),
]

kinetic_parameters: dict[str, kinetic_parameters_module.KineticParameter[float]] = {
    "beta": kinetic_parameters_module.DictBasedKineticParameter(
        {0: 0.1, 30: 0.2, 60: 0.15, 80: 0}
    ),
    "gamma": kinetic_parameters_module.ConstantKineticParameter(
        distributions.Constant(0.1)
    ),
}


def deriv(
    _t: float, y: list[float], arguments: dict[str, float]
) -> tuple[float, float, float]:
    """Defines the derivative of the function at the point _.

    Args:
        y: Current y vector.
        arguments: Dictionary of arguments configuring the problem.

    Returns:
        Tuple[float, float, float]
    """
    S, I, _ = y
    N = sum(y)
    dS_dt = -arguments["beta"] * S * I / N
    dI_dt = arguments["beta"] * S * I / N - (arguments["gamma"]) * I
    dR_dt = arguments["gamma"] * I
    return dS_dt, dI_dt, dR_dt


noiser = noisers.AdditiveNoiser(distributions.LogNormalDistribution(0, 2))
sparsifier1 = sparsifier_module.ConstantSuffixRemover(n=5, epsilon=1, mode="absolute")
sparsifier2 = sparsifier_module.ConstantSuffixRemover(n=5, epsilon=0.1, mode="relative")
sparsifier = sparsifier_module.SequentialSparsifier(
    sparsifiers=[sparsifier1, sparsifier2]
)

sm = system_model.SystemModel(
    name,
    specieses,
    kinetic_parameters,
    deriv=deriv,
    noiser=noiser,
    timestamps=distributions.Constant(100),
)
