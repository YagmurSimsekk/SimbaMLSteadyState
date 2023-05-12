from unittest.mock import patch

from simba_ml.prediction.logging import wandb_logger
from simba_ml.prediction.logging import logging_config


class Mock:
    """Magic mock class for testing purposes."""

    def __init__(self, *args, **kwargs):
        self.calls = [("__init__", args, kwargs)]

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return method


@patch("simba_ml.prediction.logging.wandb_logger.wandb", Mock())
def test_wandb_logger():
    with patch(
        "simba_ml.prediction.logging.wandb_logger.wandb",
        Mock(),
    ):
        # pylint: disable=import-outside-toplevel
        from simba_ml.prediction.logging.wandb_logger import wandb

        config = logging_config.LoggingConfig(project="simba-ml", entity="tests")
        logger = wandb_logger.WandbLogger(config)
        logger.login()
        logger.init(12, 34, name="NAME")
        logger.log({"loss": 0.5})
        logger.log({"ratio": 1.0})

        actual = wandb.calls  # pylint: disable=no-member
        assert actual == [
            ("__init__", (), {}),
            ("login", (), {}),
            (
                "init",
                (12, 34),
                {"name": "NAME", "project": "simba-ml", "entity": "tests"},
            ),
            ("log", ({"loss": 0.5},), {}),
            ("log", ({"ratio": 1.0},), {}),
        ]
