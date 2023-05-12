"""Help functions for the tests."""
import os

import pandas as pd
import numpy as np


def create_data(num_dfs=100, num_columns=5, num_rows=250):
    return [
        pd.DataFrame(
            np.random.normal(0, 1, (num_rows, num_columns)),
            columns=[f"Column {str(i)}" for i in range(num_columns)],
        )
        for _ in range(num_dfs)
    ]


def create_dummy_csvs(path, num_csvs=10, num_rows=10, num_columns=5):
    if not os.path.exists(path):
        os.makedirs(path)
    if len(os.listdir(path)) == 0:
        for i in range(num_csvs):
            df = pd.DataFrame(
                np.arange(0, num_rows * num_columns).reshape(num_rows, num_columns)
            )
            df.to_csv(path + str(i) + ".csv", index=False)
