from simba_ml.example_problems import salt_and_brine_tanks as problem_module
from simba_ml.simulation.system_model import system_model_interface


def test_problem_module_contains_sm():
    assert hasattr(problem_module, "sm")


def test_problem_module_sm_is_prediction_task():
    assert isinstance(problem_module.sm, system_model_interface.SystemModelInterface)


def test_derivate():
    kinetic_parameters = {"r": 1, "V": 100}
    y = [10, 20]
    assert problem_module.sm.deriv(0, y, kinetic_parameters) == (0.1, -0.1)
