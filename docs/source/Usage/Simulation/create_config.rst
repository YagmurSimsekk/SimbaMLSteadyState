Create Config File
==================


Config files are python files, which provide a variable `sm`, which is a `SystemModel`.
Some steps are needed, to create such a `SystemModel`.
We provide an example configuration for a simple problem: a linear function."

Load packages
-------------

>>> from simba_ml.simulation import system_model
>>> from simba_ml.simulation import generators
>>> from simba_ml.simulation import species
>>> from simba_ml.simulation import distributions
>>> from simba_ml.simulation import species
>>> from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module

Create name
-----------
A prediction task needs a name.

>>> name = "Linear Function"

Create a list of species
------------------------
Each species describes a variable of the model. In this linear function, we have only three species. The start value is either 10 or 500 or 1000.

>>> specieses = [species.Species("y", distributions.VectorDistribution([10, 500, 1000]))]

Define the arguments of the problem
-----------------------------------
Some problems need arguments. In linear function has a slope, which is parameterizable. Possible values fof those arguments are provided using an InitialCondition.

>>> kinetic_parameters = {
...     "slope": kinetic_parameters_module.ConstantKineticParameter(distributions.ContinuousUniformDistribution(0.1, 0.3)),
... }

Define Ordinary Differential Equations
---------------------------------------
Define a function which calculates the derivatives at a time t. It must have the exact arguments provided in the example. Arguments are concrete (float) values sampled from the kinetic_parameters given above. The first argument is a vector providing the values of the function at time t. The return of the function must be a tuple with the derivative for each species.

>>> def deriv(t: float, y: list[float], arguments) -> tuple[float, float, float]:
...     return (arguments["slope"],)


Create the SystemModel object
--------------------------------
Save it in a variable called ‘sm’.

>>> sm = system_model.SystemModel(name, specieses, kinetic_parameters, deriv=deriv)
