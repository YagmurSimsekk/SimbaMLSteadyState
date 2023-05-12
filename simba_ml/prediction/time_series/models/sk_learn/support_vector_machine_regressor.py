"""Provides a support vector machine regressor, which predicts next timesteps with."""
import enum
import dataclasses
import sklearn as sk

from sklearn import multioutput
from simba_ml.prediction.time_series.models.sk_learn import sk_learn_model
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


class Kernel(enum.Enum):
    """Specifies the kernel type to be used in the algorithm."""

    linear: str = "linear"
    poly: str = "poly"
    rbf: str = "rbf"
    sigmoid: str = "sigmoid"
    precomputed: str = "precomputed"


@dataclasses.dataclass
class SVMRegressorConfig(sk_learn_model.SkLearnModelConfig):
    """Defines the configuration for the SVMRegressor.

    Attributes:
        name: name of the model.
        kernel: kernel type to be used in the algorithm.
    """

    name: str = "SVM Regressor"
    kernel: Kernel = Kernel.linear


class SVMRegressorModel(sk_learn_model.SkLearnModel):
    """Defines a model, which uses a SVM regressor."""

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: SVMRegressorConfig,
    ) -> None:
        """Initializes the configuration for the SVMRegressor.

        Args:
            time_series_params: Time-series parameters that affect
                the training and architecture of models
            model_params: configuration for the model.
        """
        super().__init__(time_series_params, model_params)

    def get_model(  # type: ignore
        self, model_params: SVMRegressorConfig
    ) -> sk.base.BaseEstimator:
        """Returns the model.

        Args:
            model_params: configuration for the model.

        Returns:
            The model.
        """
        if model_params.kernel == Kernel.linear:
            return multioutput.MultiOutputRegressor(sk.svm.LinearSVR())
        return multioutput.MultiOutputRegressor(
            sk.svm.SVR(kernel=model_params.kernel.value)
        )
