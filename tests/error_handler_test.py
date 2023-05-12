import pytest

from simba_ml import error_handler


def test_confirm_param_is_float_raises_type_error_when_value_is_not_float():
    with pytest.raises(TypeError):
        error_handler.confirm_param_is_float(4)


def test_confirm_param_is_float_runs_without_error_when_provided_with_float():
    error_handler.confirm_param_is_float(1.2)


def test_confirm_param_is_int_raises_type_type_error_when_value_is_not_int():
    with pytest.raises(TypeError):
        error_handler.confirm_param_is_int(4.0)


def test_confirm_param_is_int_runs_without_error_when_provided_with_int():
    error_handler.confirm_param_is_int(5)


def test_confirm_param_is_float_or_int_raises_type_error_when_value_is_float_or_int():
    with pytest.raises(TypeError):
        error_handler.confirm_param_is_float_or_int("1231")


def test_confirm_param_is_float_or_int_runs_when_provided_with_float_or_int():
    error_handler.confirm_param_is_float_or_int(4)
    error_handler.confirm_param_is_float_or_int(4.0)


def test_confirm_seq_raises_type_error_when_list_contains_not_only_floats_or_ints():
    with pytest.raises(TypeError):
        error_handler.confirm_sequence_contains_only_floats_or_ints([1, 2.0, "3"])


def test_confirm_seq_runs_without_error_when_provided_floats_or_ints():
    error_handler.confirm_sequence_contains_only_floats_or_ints([1, 2.0])


def test_confirm_raises_value_error_when_number_is_negative():
    with pytest.raises(ValueError):
        error_handler.confirm_number_is_greater_or_equal_to_0(-2)


def test_confirm_runs_without_error_when_number_is_greater_or_equal_to_0():
    error_handler.confirm_number_is_greater_or_equal_to_0(0)
    error_handler.confirm_number_is_greater_or_equal_to_0(1)


def test_confirm_raises_value_error_when_number_is_not_positive():
    with pytest.raises(ValueError):
        error_handler.confirm_number_is_greater_than_0(0)


def test_confirm_number_is_greater_than_0_runs_without_error_when_number_is_positive():
    error_handler.confirm_number_is_greater_than_0(1)


def test_confirm_when_number_is_not_in_interval():
    with pytest.raises(ValueError):
        error_handler.confirm_number_is_in_interval(
            number=50, start_value=1, end_value=10
        )


def test_confirm_number_is_in_interval_runs_without_error_when_number_is_in_interval():
    error_handler.confirm_number_is_in_interval(number=5, start_value=1, end_value=10)


def test_confirm_sequence_is_not_empty_raises_index_error_when_list_is_empty():
    with pytest.raises(IndexError):
        error_handler.confirm_sequence_is_not_empty([])


def test_confirm_sequence_is_not_empty_runs_without_error_when_list_is_not_empty():
    error_handler.confirm_sequence_is_not_empty([1, 2, 3])
