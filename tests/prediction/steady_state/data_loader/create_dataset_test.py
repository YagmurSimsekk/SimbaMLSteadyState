import numpy as np
import pandas as pd

from simba_ml.prediction.steady_state.data_loader.dataset_generator import (
    create_dataset,
)


def test_create_dataset_returns_correct_result():
    data = [
        pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            }
        ),
        pd.DataFrame(
            {
                "a": [10.0, 11.0, 12.0],
                "b": [13.0, 14.0, 15.0],
                "c": [16.0, 17.0, 18.0],
            }
        ),
    ]
    prediction_params = ["a", "b"]
    start_value_params = ["c"]
    expected_X = np.array([[7.0], [8.0], [9.0], [16.0], [17.0], [18.0]])
    expected_y = np.array(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [10.0, 13.0], [11.0, 14.0], [12.0, 15.0]]
    )
    X, y = create_dataset(data, start_value_params, prediction_params)
    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)
