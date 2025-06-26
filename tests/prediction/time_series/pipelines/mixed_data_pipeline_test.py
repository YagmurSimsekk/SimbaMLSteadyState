"""Test function for the pipeline."""
import logging
import typing
import os
import shutil
import csv

import pytest

import pandas as pd
import tensorflow as tf
import numpy as np
from simba_ml.prediction.time_series.pipelines import mixed_data_pipeline
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_model_config_factory_raises_type_error():
    """Test function for the model_config_factory function."""
    # pylint: disable=protected-access,too-many-function-args
    with pytest.raises(TypeError):
        ts_config = time_series_config.TimeSeriesConfig(
            input_features=[], output_features=[]
        )
        mixed_data_pipeline._model_config_factory({"id": 1}, ts_config)


def test_mixed_data_pipeline_correct_results() -> None:
    actual_results = mixed_data_pipeline.main(
        "tests/prediction/time_series/conf/time_series_pipeline_test_conf.toml"
    )
    actual = [
        pd.DataFrame(data=actual_results[str(ratio)]).T
        for ratio in actual_results.keys()
    ]
    expected = [
        pd.DataFrame(
            [
                {
                    "mean_absolute_error": 125.0,
                    "mean_squared_error": 15625.0,
                    "mean_absolute_percentage_error": 0.3125,
                },
                {
                    "mean_absolute_error": 10.0,
                    "mean_squared_error": 1000.0,
                    "mean_absolute_percentage_error": 0.025,
                },
                {
                    "mean_absolute_error": 400.0,
                    "mean_squared_error": 160000.0,
                    "mean_absolute_percentage_error": 1.0,
                },
            ],
            index=["Average Predictor", "My Last Value Predictor", "Zero Predictor"],
        ),
        pd.DataFrame(
            [
                {
                    "mean_absolute_error": 118.75,
                    "mean_squared_error": 14101.5625,
                    "mean_absolute_percentage_error": 0.296875,
                },
                {
                    "mean_absolute_error": 10.0,
                    "mean_squared_error": 1000.0,
                    "mean_absolute_percentage_error": 0.025,
                },
                {
                    "mean_absolute_error": 400.0,
                    "mean_squared_error": 160000.0,
                    "mean_absolute_percentage_error": 1.0,
                },
            ],
            index=["Average Predictor", "My Last Value Predictor", "Zero Predictor"],
        ),
    ]
    logging.info("Actual: %s", actual)
    logging.info("Expected: %s", expected)
    assert list(actual_results.keys()) == ["1.0", "0.5"]
    assert isinstance(actual, typing.List)
    assert actual[0].equals(expected[0])
    assert actual[1].equals(expected[1])


def test_mixed_data_pipeline_correct_results_with_plugins_normalized() -> None:
    np.random.seed(42)
    tf.random.set_seed(42)
    pipeline_results = mixed_data_pipeline.main(
        "tests/prediction/time_series/conf/pipeline_test_conf_plugins_normalize.toml"
    )
    obtained = [
        pd.DataFrame(data=pipeline_results[str(ratio)]).T
        for ratio in pipeline_results.keys()
    ]
    assert list(pipeline_results.keys()) == ["1.0", "0.5"]
    assert isinstance(obtained, typing.List)

    # Keras
    # ratio=1.0 run
    assert obtained[0]["mean_absolute_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(80.0, rel=0.01)
    assert obtained[0]["mean_squared_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(9047.0, rel=0.001)
    assert obtained[0]["mean_absolute_percentage_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(0.217, rel=0.01)
    # ratio=0.5 run
    assert obtained[1]["mean_absolute_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(24.43, rel=0.01)
    assert obtained[1]["mean_squared_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(1021.97, rel=0.01)
    assert obtained[1]["mean_absolute_percentage_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(0.056, rel=0.01)

    # PyTorch Lightning
    # ratio=1.0 run
    assert obtained[0]["mean_absolute_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(108.0, rel=0.01)
    assert obtained[0]["mean_squared_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(14966.1, rel=0.01)
    assert obtained[0]["mean_absolute_percentage_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(0.535, rel=0.01)
    # ratio=0.5 run
    assert obtained[1]["mean_absolute_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(104.63, rel=0.01)
    assert obtained[1]["mean_squared_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(14549.14, rel=0.01)
    assert obtained[1]["mean_absolute_percentage_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(0.527, rel=0.01)


def test_mixed_data_pipeline_correct_results_with_plugins() -> None:
    pipeline_results = mixed_data_pipeline.main(
        "tests/prediction/time_series/conf/pipeline_test_conf_plugins.toml"
    )
    obtained = [
        pd.DataFrame(data=pipeline_results[str(ratio)]).T
        for ratio in pipeline_results.keys()
    ]
    assert list(pipeline_results.keys()) == ["1.0"]
    assert isinstance(obtained, typing.List)

    # Keras
    # ratio=1.0 run
    assert obtained[0]["mean_absolute_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(275.0, rel=0.01)
    assert obtained[0]["mean_squared_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(80105.0, rel=0.01)
    assert obtained[0]["mean_absolute_percentage_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(0.95, rel=0.01)

    # PyTorch Lightning
    # ratio=1.0 run
    assert obtained[0]["mean_absolute_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(357.0, rel=0.01)
    assert obtained[0]["mean_squared_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(134100.90, rel=0.01)
    assert obtained[0]["mean_absolute_percentage_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(1.25, rel=0.01)


def test_mixed_data_pipeline_returns_results_correct_type_and_format() -> None:
    """Test function for the pipeline."""
    results = mixed_data_pipeline.main(
        "tests/prediction/time_series/conf/time_series_pipeline_test_conf.toml"
    )

    assert isinstance(results, typing.Dict)
    assert len(results) == 2
    assert isinstance(results["1.0"], typing.Dict)
    assert len(results["1.0"]) == 3
    assert isinstance(results["1.0"]["Average Predictor"], typing.Dict)
    assert len(results["1.0"]["Average Predictor"]) == 3


def test_mixed_data_pipeline_export() -> None:
    export_path = "tests/prediction/time_series/test_data/export"
    if os.path.exists(os.path.join(os.getcwd(), export_path)):
        shutil.rmtree(os.path.join(os.getcwd(), export_path))
    mixed_data_pipeline.main(
        "tests/prediction/time_series/conf/mixed_data_pipeline_export.toml"
    )
    assert (
        len(os.listdir(os.path.join(os.getcwd(), export_path))) == 5
    )
    assert np.load(
        os.path.join(
            os.getcwd(), export_path, "Keras Dense Neural Network-1.0-y_pred.npy"
        )
    ).shape == (50, 1, 2)
    assert np.load(
        os.path.join(os.getcwd(), export_path, "y_true.npy")
    ).shape == (50, 1, 2)
    with open(
        os.path.join(os.getcwd(), export_path, "features.csv"),
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.reader(f)
        data = list(reader)
        assert data == [["Infected", "Recovered"]]
    shutil.rmtree(os.path.join(os.getcwd(), export_path))
