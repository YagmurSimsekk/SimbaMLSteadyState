"""This module provides the dataloader."""
import os
from typing import Tuple

import pandas as pd
import numpy as np
from numpy import typing as npt

from simba_ml.prediction import preprocessing
from simba_ml.prediction.steady_state.config import steady_state_data_config
from simba_ml.prediction.steady_state.data_loader import splits, dataset_generator


class MixedDataLoader:
    """Loads and preprocesses the data.

    Attributes:
        X_test: the input of the test data
        y_test: the labels for the test data
        train_validation_sets: list of validations sets,
            one for each ratio of synthethic to observed data.
    """

    config: steady_state_data_config.DataConfig
    __X_test: npt.NDArray[np.float64] | None = None
    __y_test: npt.NDArray[np.float64] | None = None
    __list_of_train_validation_sets: list[
        list[dict[str, list[npt.NDArray[np.float64]]]]
    ] = []

    def __init__(self, config: steady_state_data_config.DataConfig) -> None:
        """Inits the DataLoader.

        Args:
            config: the data configuration.
        """
        self.config = config
        self.__list_of_train_validation_sets = [
            [] for _ in range(len(self.config.mixing_ratios))
        ]

    def load_data(self) -> Tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Loads the data.

        Returns:
            A list of dataframes.
        """
        synthetic = (
            []
            if self.config.synthethic is None
            else preprocessing.read_dataframes_from_csvs(
                os.getcwd() + self.config.synthethic
            )
        )
        observed = (
            []
            if self.config.observed is None
            else preprocessing.read_dataframes_from_csvs(
                os.getcwd() + self.config.observed
            )
        )
        return synthetic, observed

    def prepare_data(self) -> None:
        """This function preprocesses the data."""
        if self.__X_test is not None:  # pragma: no cover
            return  # pragma: no cover

        synthethic_data, observed_data = self.load_data()

        synthetic_train, _ = splits.train_test_split(
            data=synthethic_data, test_split=self.config.test_split
        )
        observed_train, observed_test = splits.train_test_split(
            data=observed_data, test_split=self.config.test_split
        )

        # compute train validation sets for each
        # of the data ratios defined in the data config
        for ratio_idx, ratio in enumerate(self.config.mixing_ratios):
            train = preprocessing.mix_data(
                synthetic_data=synthetic_train,
                observed_data=observed_train,
                ratio=ratio,
            )
            train_validation_sets = splits.train_validation_split(
                train, k_cross_validation=self.config.k_cross_validation
            )
            for train_validation_set in train_validation_sets:
                train_validation_set[
                    "train"
                ] = preprocessing.convert_dataframe_to_numpy(
                    train_validation_set["train"]
                )
                train_validation_set[
                    "validation"
                ] = preprocessing.convert_dataframe_to_numpy(
                    train_validation_set["validation"]
                )
            self.__list_of_train_validation_sets[ratio_idx] = train_validation_sets

        self.__X_test, self.__y_test = dataset_generator.create_dataset(
            observed_test, self.config.start_value_params, self.config.prediction_params
        )

    # sourcery skip: snake-case-functions
    @property
    def X_test(self) -> npt.NDArray[np.float64]:
        """The input of the test dataset.

        Returns:
            The input of the test dataset.
        """
        if self.__X_test is None:
            self.prepare_data()
            return self.X_test
        return self.__X_test

    @property
    def y_test(self) -> npt.NDArray[np.float64]:
        """The output of the test dataset.

        Returns:
            The output of the test dataset.
        """
        if self.__y_test is None:
            self.prepare_data()
            return self.y_test
        return self.__y_test

    @property
    def list_of_train_validation_sets(
        self,
    ) -> list[list[dict[str, list[npt.NDArray[np.float64]]]]]:
        """Lists of train validation sets.

        One set for each ratio of synthethic to observed data.

        Returns:
            A list of list of dicts containing train and validation sets.
        """
        if all(li == [] for li in self.__list_of_train_validation_sets):
            self.prepare_data()
            return self.__list_of_train_validation_sets
        return self.__list_of_train_validation_sets
