import pytest

from simba_ml.prediction.time_series.models import check_params
from simba_ml.prediction.time_series.models.pytorch_lightning import (
    pytorch_lightning_model,
)


def test_check_params_with_only_finetuning_set():
    cfg = pytorch_lightning_model.PytorchLightningModelConfig(
        finetuning=True,
    )
    with pytest.raises(ValueError):
        check_params.check_funetuning_params(cfg)


def test_check_params_finetuning_and_epochs_set():
    cfg = pytorch_lightning_model.PytorchLightningModelConfig(
        finetuning=True,
    )
    cfg.training_params.finetuning_epochs = 1
    with pytest.raises(ValueError):
        check_params.check_funetuning_params(cfg)


def test_check_params_finetuning_and_lr_set():
    cfg = pytorch_lightning_model.PytorchLightningModelConfig(
        finetuning=True,
    )
    cfg.training_params.finetuning_learning_rate = 0.0001
    with pytest.raises(ValueError):
        check_params.check_funetuning_params(cfg)


def test_check_params_all_set_for_finetuning():
    cfg = pytorch_lightning_model.PytorchLightningModelConfig(
        finetuning=True,
    )
    cfg.training_params.finetuning_learning_rate = 0.0001
    cfg.training_params.finetuning_epochs = 1
    check_params.check_funetuning_params(cfg)
