from unittest.mock import patch

from click.testing import CliRunner

from simba_ml.cli.__main__ import main


@patch("streamlit.web.bootstrap.run")
def test_main_does_not_fail_when_called_appropriatly(mock_bootstrap):
    runner = CliRunner()
    runner.invoke(
        main,
        [
            "run-problem-viewer",
            "--module",
            "simba_ml.example_problems.trigonometry",
        ],
    )
    assert mock_bootstrap.called
