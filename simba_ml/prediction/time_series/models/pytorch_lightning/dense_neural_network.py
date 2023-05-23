"""Uses a Dense Neural Network to predict the next timestamps."""

import typing
import dataclasses

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from simba_ml.prediction.time_series.models.pytorch_lightning import (
    pytorch_lightning_model,
)
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


@dataclasses.dataclass
class DenseNeuralNetworkConfig(pytorch_lightning_model.PytorchLightningModelConfig):
    """Defines the configuration for the DenseNeuralNetwork."""

    name: str = "PyTorch Lightning Dense Neural Network"


class DenseNeuralNetwork(pytorch_lightning_model.PytorchLightningModel):
    """Defines a model, which uses a dense neural network for prediction.

    Args:
        history: History documenting the training process of the model.
    """

    def get_model(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: pytorch_lightning_model.PytorchLightningModelConfig,
    ) -> pl.LightningModule:
        """Returns the model.

        Args:
            time_series_params: parameters of time series that affects
                training and architecture of the model
            model_params: configuration for the model.

        Returns:
            The model.
        """
        return _DenseNeuralNetwork(time_series_params, model_params)


class _DenseNeuralNetwork(pl.LightningModule):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: pytorch_lightning_model.PytorchLightningModelConfig,
    ) -> None:
        super().__init__()
        self.time_series_params = time_series_params
        self.model_params = model_params
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=time_series_params.input_length
                * len(time_series_params.input_features),
                out_features=model_params.architecture_params.units,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=model_params.architecture_params.units,
                out_features=time_series_params.output_length
                * len(time_series_params.output_features),
            ),
        )

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self.model(x).reshape(
            (
                x.shape[0],
                self.time_series_params.output_length,
                len(self.time_series_params.output_features),
            )
        )

    def configure_optimizers(self) -> torch.optim.Adam:
        if self.model_params.finetuning:
            if self.model_params.training_params.finetuning_learning_rate is None:
                raise ValueError("finetuning_learning_rate must be set.")
            return torch.optim.Adam(
                self.parameters(),
                lr=self.model_params.training_params.finetuning_learning_rate,
            )
        return torch.optim.Adam(
            self.parameters(),
            lr=self.model_params.training_params.learning_rate,
        )

    def training_step(  # pylint: disable=arguments-differ
        self,
        train_batch: typing.List[typing.Tuple[typing.List[float], typing.List[int]]],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        x, y = train_batch
        out = self(x)
        return F.mse_loss(y, out)  # type: ignore[arg-type]
