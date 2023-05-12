"""Provides a normalizer, which can normalize train and test data."""
import numpy as np
from numpy import typing as npt
import pandas as pd

from simba_ml.prediction.time_series.config import (
    time_series_config,
)


class NotInitializedError(Exception):
    """Raised when `normalize_test_data` is called on uninitialized Normalizer."""


class Normalizer:
    """Normalizes train and test data."""

    train_std: npt.NDArray[np.float64]
    train_mean: npt.NDArray[np.float64]
    initialized: bool = False
    x_features_indices: list[int]
    y_features_indices: list[int]

    def normalize_train_data(
        self,
        train: list[pd.DataFrame],
        time_series_params: time_series_config.TimeSeriesConfig,
    ) -> list[pd.DataFrame]:
        """Normalizes the train data.

        Normalizes the train data by subtracting the mean and dividing
        through the standard deviation of the train set.

        Args:
            train: Train data.
            time_series_params: Time series parameters.

        Returns:
            The normalized train data.
        """
        self.initialized = True
        # select only the features that are used for training
        self.x_features_indices = [
            i
            for i, column in enumerate(train[0].columns)
            if column in time_series_params.input_features
        ]
        # get the index of output features in input feature list
        self.y_features_indices = [
            i
            for i, column in enumerate(train[0].columns)
            if column in time_series_params.output_features
        ]
        self.train_mean = np.concatenate(train).mean(axis=0)
        self.train_std = np.concatenate(train).std(axis=0)
        self.train_std[self.train_std == 0.0] = 1.0
        return [
            pd.DataFrame((train[i] - self.train_mean) / self.train_std)
            for i in range(len(train))
        ]

    def normalize_test_data(
        self, test: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Normalizes the test data.

        Subtracts the mean and divides through the standard deviation of the train set.

        Args:
            test: Test data.

        Returns:
            The normalized test data.

        Raises:
            NotInitializedError: If the normalizer is not initialized.
        """
        if not self.initialized:
            raise NotInitializedError()
        return (test - self.train_mean[self.x_features_indices]) / self.train_std[
            self.x_features_indices
        ]

    def denormalize_prediction_data(
        self, data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Denormalizes the data.

        Denormalizes the data, by multiplying with the standard deviation
        and adding the mean of the train set.

        Args:
            data: Data to denormalize.

        Returns:
            The denormalized data.

        Raises:
            NotInitializedError: If the normalizer is not initialized.
        """
        if not self.initialized:
            raise NotInitializedError()
        return (
            data * self.train_std[self.y_features_indices]
            + self.train_mean[self.y_features_indices]
        )
