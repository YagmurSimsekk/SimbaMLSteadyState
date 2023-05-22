"""Checks whether all the required parameters are set."""

from simba_ml.prediction.time_series.models.pytorch_lightning import (
    pytorch_lightning_model,
)


def check_funetuning_params(
    model_params: pytorch_lightning_model.PytorchLightningModelConfig,
):
    """Checks whether all the required arguments for finetuning the model are set.

    Args:
        model_params: the model parameters to check.

    Raises:
        ValueError: if the model is not set to finetuning.
    """
    if model_params.finetuning and not (
        model_params.training_params.finetuning_learning_rate
        and model_params.training_params.finetuning_epochs
    ):
        raise ValueError(
            "The model is set to finetuning but the finetuning learning rate or the finetuning epochs are not set."  # pylint: disable=line-too-long
        )
