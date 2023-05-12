"""Provides the sklearn models."""
# pylint: disable=only-importing-modules-is-allowed
from simba_ml.prediction.time_series.models.sk_learn.decision_tree_regressor import (
    DecisionTreeRegressorModel,
)
from simba_ml.prediction.time_series.models.sk_learn.linear_regressor import (
    LinearRegressorModel,
)

from simba_ml.prediction.time_series.models.sk_learn.random_forest_regressor import (
    RandomForestRegressorModel,
)

# pylint: disable=line-too-long
from simba_ml.prediction.time_series.models.sk_learn.support_vector_machine_regressor import (
    SVMRegressorModel,
)

from simba_ml.prediction.time_series.models.sk_learn.nearest_neighbors_regressor import (
    NearestNeighborsRegressorModel,
)

# pylint: enable=line-too-long

__all__ = [
    "DecisionTreeRegressorModel",
    "LinearRegressorModel",
    "NearestNeighborsRegressorModel",
    "RandomForestRegressorModel",
    "SVMRegressorModel",
]
