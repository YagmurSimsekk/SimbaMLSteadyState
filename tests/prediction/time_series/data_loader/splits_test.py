import numpy as np
import pandas as pd
import pandas.testing as pdtest
import pytest

from simba_ml.prediction.time_series.data_loader import splits
from tests.prediction.help_functions import create_data


def test_train_test_split_horizontal_format():
    input_length = 5
    data = create_data(num_columns=5)
    test_split = 0.2
    split_axis = "horizontal"
    train, test = splits.train_test_split(
        data=data,
        test_split=test_split,
        input_length=input_length,
        split_axis=split_axis,
    )
    assert len(train) == len(data) * (1 - test_split)
    assert train[0].shape[1] == 5
    assert len(test) == len(data) * test_split
    assert test[0].shape[1] == 5


def test_train_test_split_vertical_format():
    input_length = 5
    data = create_data()
    test_split = 0.2
    split_axis = "vertical"
    train, test = splits.train_test_split(
        data=data,
        test_split=test_split,
        input_length=input_length,
        split_axis=split_axis,
    )
    assert len(train[0]) == int(data[0].shape[0] * (1 - test_split))
    assert len(test[0]) == int(data[0].shape[0] * test_split) + input_length


def test_train_test_split_vertical_test_not_in_train():
    input_length = 3
    data = [pd.DataFrame(np.array([1, 2, 3, 4, 5]))]
    test_split = 0.2
    split_axis = "vertical"
    train, test = splits.train_test_split(
        data=data,
        test_split=test_split,
        input_length=input_length,
        split_axis=split_axis,
    )
    pdtest.assert_frame_equal(train[0], pd.DataFrame(np.array([1, 2, 3, 4])))
    pdtest.assert_frame_equal(test[0], pd.DataFrame(np.array([2, 3, 4, 5])))


def test_train_test_split_raises_value_error():
    input_length = 3
    data = create_data(num_dfs=1, num_columns=2, num_rows=10)
    test_split = 0.2
    split_axis = "test"
    with pytest.raises(ValueError):
        splits.train_test_split(
            data=data,
            test_split=test_split,
            input_length=input_length,
            split_axis=split_axis,
        )
