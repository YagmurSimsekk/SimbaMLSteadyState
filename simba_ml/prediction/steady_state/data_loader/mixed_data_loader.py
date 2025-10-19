"""This module provides the dataloader."""
import os
from typing import Tuple, Union, Optional, List
from pathlib import Path

import pandas as pd
import numpy as np
from numpy import typing as npt

from simba_ml.prediction import preprocessing
from simba_ml.prediction.steady_state.config import steady_state_data_config
from simba_ml.prediction.steady_state.config.pattern_based_config import create_config_from_csv
from simba_ml.prediction.steady_state.data_loader import splits, dataset_generator


class MixedDataLoader:
    """Loads and preprocesses the data with automatic or manual configuration.

    Can be initialized in three ways:
    1. Manual configuration: MixedDataLoader(config)
    2. Automatic from CSV: MixedDataLoader.from_sbml_csv('data.csv')
    3. Automatic from DataFrame: MixedDataLoader.from_sbml_dataframes(df)

    Attributes:
        X_test: the input of the test data
        y_test: the labels for the test data
        train_validation_sets: list of validations sets,
            one for each ratio of synthethic to observed data.

    Examples:
        >>> # Manual configuration
        >>> config = DataConfig(
        ...     start_value_params=['kinetic_parameter_k1', 'M_start_value'],
        ...     prediction_params=['M', 'MpY'],
        ...     mixing_ratios=[0.0, 0.5, 1.0]
        ... )
        >>> loader = MixedDataLoader(config)
        >>>
        >>> # Automatic from CSV
        >>> loader = MixedDataLoader.from_sbml_csv(
        ...     'steady_state_data.csv',
        ...     max_input_features=None,  # Use all features
        ...     verbose=True
        ... )
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
        # For caching DataFrames when using from_sbml_dataframes()
        self._cached_synthetic_data = None
        self._cached_observed_data = None

    @classmethod
    def from_sbml_csv(cls,
                      synthetic_csv: Union[str, Path],
                      observed_csv: Optional[Union[str, Path]] = None,
                      max_input_features: Optional[int] = None,
                      max_output_features: Optional[int] = None,
                      mixing_ratios: Optional[List[float]] = None,
                      test_split: float = 0.2,
                      k_cross_validation: int = 5,
                      verbose: bool = False) -> 'MixedDataLoader':
        """
        Create MixedDataLoader by automatically analyzing SBML CSV file.

        Args:
            synthetic_csv: Path to synthetic SBML steady-state data CSV
            observed_csv: Path to observed/experimental data CSV (optional)
            max_input_features: Limit number of input features (None = use all)
            max_output_features: Limit number of output features (None = use all)
            mixing_ratios: Ratios for synthetic/observed data mixing
            test_split: Test set proportion
            k_cross_validation: Number of CV folds
            verbose: Whether to print dataset analysis

        Returns:
            Configured MixedDataLoader instance

        Example:
            >>> # Automatic feature detection
            >>> loader = MixedDataLoader.from_sbml_csv(
            ...     'steady_state_data.csv',
            ...     max_input_features=10,
            ...     max_output_features=5
            ... )
            >>> loader.prepare_data()
            >>> X_test, y_test = loader.X_test, loader.y_test
        """
        # Auto-generate config from synthetic data using pattern-based detection
        config = create_config_from_csv(
            csv_file=str(synthetic_csv),
            method='automatic',
            max_input_features=max_input_features,
            max_output_features=max_output_features,
            mixing_ratios=mixing_ratios,
            test_split=test_split,
            k_cross_validation=k_cross_validation,
            verbose=verbose
        )

        # Update config paths
        config.synthethic = str(synthetic_csv)
        if observed_csv:
            config.observed = str(observed_csv)

        return cls(config)

    @classmethod
    def from_sbml_dataframes(cls,
                             synthetic_df: pd.DataFrame,
                             observed_df: Optional[pd.DataFrame] = None,
                             max_input_features: Optional[int] = None,
                             max_output_features: Optional[int] = None,
                             mixing_ratios: Optional[List[float]] = None,
                             test_split: float = 0.2,
                             k_cross_validation: int = 5,
                             verbose: bool = False) -> 'MixedDataLoader':
        """
        Create MixedDataLoader directly from DataFrames.

        Args:
            synthetic_df: Synthetic SBML steady-state DataFrame
            observed_df: Observed/experimental DataFrame (optional)
            max_input_features: Limit number of input features (None = use all)
            max_output_features: Limit number of output features (None = use all)
            mixing_ratios: Ratios for synthetic/observed data mixing
            test_split: Test set proportion
            k_cross_validation: Number of CV folds
            verbose: Whether to print dataset analysis

        Returns:
            Configured MixedDataLoader instance with DataFrames cached
        """
        # Save DataFrame to temp CSV for pattern-based config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            synthetic_df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            # Generate config using pattern-based detection
            config = create_config_from_csv(
                csv_file=tmp_path,
                method='automatic',
                max_input_features=max_input_features,
                max_output_features=max_output_features,
                mixing_ratios=mixing_ratios,
                test_split=test_split,
                k_cross_validation=k_cross_validation,
                verbose=verbose
            )
        finally:
            os.unlink(tmp_path)

        # Create instance
        instance = cls(config)

        # Cache the DataFrames to avoid file I/O
        instance._cached_synthetic_data = [synthetic_df]
        if observed_df is not None:
            instance._cached_observed_data = [observed_df]

        return instance

    def load_data(self) -> Tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Loads the data.

        Returns:
            A list of dataframes.
        """
        # Use cached data if available (from from_sbml_dataframes())
        if self._cached_synthetic_data is not None:
            synthetic_data = self._cached_synthetic_data
        else:
            synthetic_data = self._load_synthetic_data()

        if self._cached_observed_data is not None:
            observed_data = self._cached_observed_data
        else:
            observed_data = self._load_observed_data()

        return synthetic_data, observed_data

    def _load_synthetic_data(self) -> list[pd.DataFrame]:
        """Load synthetic data from file."""
        if self.config.synthethic:
            path = Path(self.config.synthethic)
            if path.is_file():
                return [pd.read_csv(path)]
            else:
                return preprocessing.read_dataframes_from_csvs(
                    os.getcwd() + self.config.synthethic
                )
        else:
            return []

    def _load_observed_data(self) -> list[pd.DataFrame]:
        """Load observed data from file."""
        if self.config.observed:
            path = Path(self.config.observed)
            if path.is_file():
                return [pd.read_csv(path)]
            else:
                return preprocessing.read_dataframes_from_csvs(
                    os.getcwd() + self.config.observed
                )
        else:
            return []

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

    def print_config_summary(self) -> None:
        """Print summary of the current configuration."""
        print("📋 MixedDataLoader Configuration")
        print("=" * 45)
        print(f"Input features ({len(self.config.start_value_params)}): {self.config.start_value_params[:3]}..."
              if len(self.config.start_value_params) > 3 else f"Input features: {self.config.start_value_params}")
        print(f"Output features ({len(self.config.prediction_params)}): {self.config.prediction_params[:3]}..."
              if len(self.config.prediction_params) > 3 else f"Output features: {self.config.prediction_params}")
        print(f"Mixing ratios: {self.config.mixing_ratios}")
        print(f"Test split: {self.config.test_split}")
        print(f"K-fold CV: {self.config.k_cross_validation}")

        if self.config.synthethic:
            print(f"Synthetic data: {self.config.synthethic}")
        if self.config.observed:
            print(f"Observed data: {self.config.observed}")
