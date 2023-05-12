import os
import shutil

from click.testing import CliRunner
import pandas as pd

from simba_ml.cli.__main__ import main


def test_generate_data() -> None:
    """Tests start_prediction."""
    output_path = "./tmp/"
    runner = CliRunner()
    runner.invoke(
        main,
        [
            "generate-data",
            "--output-dir",
            output_path,
            "--config-module",
            "simba_ml.example_problems.sir",
            "-n",
            "1",
        ],
    )
    assert os.path.exists(output_path)
    assert len(os.listdir(output_path)) == 1
    results = pd.read_csv(output_path + os.listdir(output_path)[0], index_col=0)
    assert isinstance(results, pd.DataFrame)
    assert results.shape == (100, 0)
    shutil.rmtree(output_path)
