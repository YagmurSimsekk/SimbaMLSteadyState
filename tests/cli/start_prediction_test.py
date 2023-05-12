import os

from click.testing import CliRunner
import pandas as pd

from simba_ml.cli.__main__ import main


def test_start_prediction() -> None:
    """Tests start_prediction."""
    output_path = "tmp.csv"
    runner = CliRunner()
    runner.invoke(
        main,
        [
            "start-prediction",
            "synthetic_data",
            "--output-path",
            output_path,
            "--config-path",
            "tests/prediction/time_series/conf/synthetic_data_pipeline_test_conf.toml",
        ],
    )
    results = pd.read_csv(output_path, index_col=0)
    assert isinstance(results, pd.DataFrame)
    assert results.shape == (7, 3)
    os.remove(output_path)
