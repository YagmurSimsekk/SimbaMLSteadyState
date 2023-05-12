"""Implements methods for general error handling."""

import typing
import collections


def confirm_param_is_float(param: typing.Any, param_name: str = "param") -> None:
    """Checks that param is of type float.

    Args:
        param: The param that gets confirmed.
        param_name: Optional name to display in the error message.

    Raises:
        TypeError: If param is not a float.
    """
    if not isinstance(param, float):
        raise TypeError(f"{param_name} must be a float, not {str(type(param))}.")


def confirm_param_is_int(param: typing.Any, param_name: str = "param") -> None:
    """Checks that param is of type int.

    Args:
        param: The param that gets confirmed.
        param_name: Optional name to display in the error message.

    Raises:
        TypeError: If param is not an int.
    """
    if not isinstance(param, int):
        raise TypeError(f"{param_name} must be an int, not {str(type(param))}.")


def confirm_param_is_float_or_int(param: typing.Any, param_name: str = "param") -> None:
    """Checks that param is of type float or int.

    Args:
        param: The param that gets confirmed.
        param_name: Optional name to display in the error message.

    Raises:
        TypeError: If param is not a float or int.
    """
    if not isinstance(param, (float, int)):
        raise TypeError(f"{param_name} must be float or int, not {str(type(param))}.")


def confirm_sequence_contains_only_floats_or_ints(
    sequence_: collections.abc.Sequence[typing.Any], param_name: str = "param"
) -> None:
    """Checks that an iterable contains only floats and ints.

    Args:
        sequence_: The sequence that gets confirmed.
        param_name: Optional name to display in the error message.

    Raises:
        TypeError: If sequence contains other data types than float and int.
    """
    if not all(isinstance(v, (float, int)) for v in sequence_):
        raise TypeError(f"All values of {param_name} must be float or int.")


def confirm_number_is_greater_or_equal_to_0(
    number: typing.Union[int, float], param_name: str = "param"
) -> None:
    """Checks that a number is greater or equal to 0.

    Args:
        number: The number that gets confirmed.
        param_name: Optional name to display in the error message.

    Raises:
        ValueError: If param is < 0.
    """
    if number < 0:
        raise ValueError(f"{param_name} must be >= 0, not {number}.")


def confirm_number_is_greater_than_0(
    number: typing.Union[int, float], param_name: str = "param"
) -> None:
    """Checks that a number is greater than 0.

    Args:
        number: The number that gets confirmed.
        param_name: Optional name to display in the error message.

    Raises:
        ValueError: If param <= 0.
    """
    if number <= 0:
        raise ValueError(f"{param_name} must be > 0, not {number}.")


def confirm_number_is_in_interval(
    number: typing.Union[int, float],
    start_value: typing.Union[int, float],
    end_value: typing.Union[int, float],
    include_left: bool = True,
    include_right: bool = True,
    param_name: str = "param",
) -> None:
    """Checks that a number lays in an intervall.

    Args:
        number: The number that gets confirmed.
        start_value: Start value of the intervall.
        end_value: End value of the intervall.
        include_left: Whether or not the start value is included in the intervall.
        include_right: Whether or not the end value is included in the intervall.
        param_name: Optional name to display in the error message.

    Raises:
        ValueError: If number is not in the provided intervall.
    """
    left = number >= start_value if include_left else number > start_value
    right = number <= end_value if include_right else number < end_value

    if not (left and right):
        interval_type_left = "inclusive" if include_left else "exclusive"
        interval_type_right = "inclusive" if include_right else "exclusive"
        raise ValueError(
            f"{param_name} must be a value between {start_value} ({interval_type_left})"
            f"and {end_value} ({interval_type_right}), not {str(number)}."
        )


def confirm_sequence_is_not_empty(
    sequence_: collections.abc.Sequence[typing.Any], param_name: str = "param"
) -> None:
    """Checks that a sequence in not empty.

    Args:
        sequence_: The sequence that gets confirmed.
        param_name: Optional name to display in the error message.

    Raises:
        IndexError: If sequence is empty.
    """
    if len(sequence_) == 0:
        raise IndexError(f"{param_name} can't be an empty.")
