"""Provides the `SystemModel` and and it's interface."""

# pylint: disable=only-importing-modules-is-allowed
from simba_ml.simulation.system_model.system_model import SystemModel
from simba_ml.simulation.system_model.sbml_system_model import SBMLSystemModel

__all__ = ["SystemModel", "SBMLSystemModel"]
