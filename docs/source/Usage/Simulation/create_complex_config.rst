Create a more complex Config File
=================================

Here, we provide an example for a more complex problem: SIR.

Load packages
------------------

>>> from simba_ml.simulation import system_model
>>> from simba_ml.simulation import generators
>>> from simba_ml.simulation import species
>>> from simba_ml.simulation import distributions
>>> from simba_ml.simulation import noisers
>>> from simba_ml.simulation import derivative_noiser
>>> from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module
>>>
>>> name = "SIR"
>>>
>>> specieses = [
...     species.Species("Suspectible", distributions.Constant(999), contained_in_output=False),
...     species.Species("Infected", distributions.VectorDistribution([10, 500, 1000])),
...     species.Species("Recovered", distributions.LogNormalDistribution(2, 1), contained_in_output=False),
... ]
...
>>> kinetic_parameters = {
...     "beta": kinetic_parameters_module.ConstantKineticParameter(distributions.ContinuousUniformDistribution(0.1, 0.3)),
...     "gamma": kinetic_parameters_module.ConstantKineticParameter(distributions.Constant(0.04))
... }
...
>>> def deriv(t: float, y: list[float], arguments: dict) -> tuple[float, float, float]:
...     S, I, _ = y
...     N = sum(y)
...     dS_dt = -arguments["beta"] * S * I / N
...     dI_dt = arguments["beta"] * S * I / N - (arguments["gamma"]) * I
...     dR_dt = arguments["gamma"] * I
...     return dS_dt, dI_dt, dR_dt

Add Noisers
-----------
Noisers can noise a signal, while derivative noisers can noise the derivative.

>>> noiser = noisers.AdditiveNoiser(distributions.LogNormalDistribution(0, 2))
>>> derivative_noiser = derivative_noiser.AdditiveDerivNoiser(distributions.NormalDistribution(0, 0.05))

Create the SystemModel object
--------------------------------
Save it in a variable called ‘sm’.

>>> sm = system_model.SystemModel(name, specieses, kinetic_parameters, deriv=deriv, deriv_noiser=derivative_noiser, noiser=noiser)