from simba_ml.example_problems import trigonometry as problem_module
from simba_ml.simulation.system_model import system_model_interface


def test_problem_module_contains_sm():
    assert hasattr(problem_module, "sm")


def test_problem_module_sm_is_prediction_task():
    assert isinstance(problem_module.sm, system_model_interface.SystemModelInterface)


def test_derivate():
    assert problem_module.sm.deriv(0, [1, 2, 3, 4], {}) == (0.02, 0.03, 0.04, 0.01)
