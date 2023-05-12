from simba_ml.example_problems import constant_function as problem_module
from simba_ml.simulation.system_model import system_model_interface


def test_problem_module_contains_sm():
    assert hasattr(problem_module, "sm")


def test_problem_module_sm_is_prediction_task():
    assert isinstance(problem_module.sm, system_model_interface.SystemModelInterface)


def test_derivate():
    kinetic_parameters = {}
    y = [10]
    assert problem_module.sm.deriv(0, y, kinetic_parameters) == (0,)
