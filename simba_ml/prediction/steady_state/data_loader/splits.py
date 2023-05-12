"""Module with splitting functions for steady state data."""
import random

import pandas as pd


def train_test_split(
    data: list[pd.DataFrame], test_split: float
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Splits a given dataframe in train, test and validations split.

    Args:
        data: List of time series.
        test_split: percentage of data that will be used for the test split.

    Returns:
        Tuple of train and test set.
    """
    test = []
    train = []
    random.shuffle(data)
    total_data_len = len(data)
    test_train_split = int(total_data_len * test_split)
    test = data[:test_train_split]
    train = data[test_train_split:]
    return train, test


def train_validation_split(
    data: list[pd.DataFrame], k_cross_validation: int = 5
) -> list[dict[str, list[pd.DataFrame]]]:
    """Splits a given dataframe horizontally in train and validations split.

    Args:
        data: List of time series.
        k_cross_validation: Number of cross validation splits. Defaults to 5.

    Returns:
        A set consisting of a List with k cross validation sets,
            with a train set and a validation set, where each set is a list
            of dataframe time series.
    """
    random.shuffle(data)
    train_validation = [
        {"train": [pd.DataFrame], "validation": [pd.DataFrame]}
        for _ in range(k_cross_validation)
    ]
    train_validation_split_point = len(data) // k_cross_validation
    for i in range(k_cross_validation):
        train = (
            data[: i * train_validation_split_point]
            + data[(i + 1) * train_validation_split_point :]
        )
        validation = data[
            i * train_validation_split_point : (i + 1) * train_validation_split_point
        ]
        train_validation[i]["train"] = train
        train_validation[i]["validation"] = validation
    return train_validation
