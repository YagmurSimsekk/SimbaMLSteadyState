Time-Series Models
==================

SimbaML provides a variety of models for time-series prediction. These models can be used for the synthetic-data pipeline and mixed-data pipeline.

.. note::
    Time-Series models in **SimbaML** are models that are trained once on either synthetic or a mix of synthetic and real-world data. In constrast, transfer learning time-series models are first trained on synthetic and then fine-tuned on real-world data. Here, the model's train() function is called twice.

Provided Models for Time-Series Prediction
------------------------------------------

The following models can be used out-of-the-box for the synthetic-data pipeline and mixed-data pipeline. Note that the strings indicate the model's id that is used in the config file to select the model for training.

* Scikit-learn
    * Linear Regressor: "LinearRegressor"
    * Random Forest: "RandomForestRegressor"
    * Decision Tree: "DecisionTreeRegressor"
    * Support Vector Machine: "SVMRegressor"
    * Nearest Neighbor: "NearestNeighborRegressor"
* Pytorch Lightning
    * Deep Neural Network: "PytorchLightningDenseNeuralNetwork"
* Keras
    * Deep Neural Network: "KerasDenseNeuralNetwork"

.. note::
    To successfully run the Pytorch Lightning and Keras deep neural network, you have to install the corresponding packages first. Both are not installed by default when installing **SimbaML**.

Besides, we provide two benchmark models:

* Last Value Predictor: "LastValuePredictor"
* Average Value Predictor "AverageValuePredictor"


Include Own Model for Time-Series Prediction
--------------------------------------------

SimbaML allows for the effortless integration of any other machine learning models for time-series prediction.
This includes PyTorch Lightning, Keras, and Scikit-learn models.
The following code show how to integrate a model, which always predicts zero:

>>> import dataclasses
>>> import numpy as np
>>> import numpy.typing as npt
>>> 
>>> from simba_ml.prediction.time_series.models import model
>>> from simba_ml.prediction.time_series.models import factory
>>> 
>>> 
>>> @dataclasses.dataclass
... class ZeroPredictorConfig(model.ModelConfig):
...     """Defines the configuration for the DenseNeuralNetwork."""
...     name: str = "Zero Predictor"
... 
... 
>>> class ZeroPredictor(model.Model):
...     """Defines a model, which predicts the average of the train data."""
... 
...     def __init__(self, input_length: int, output_length: int, config: ZeroPredictorConfig):
...         """Inits the `AveragePredictor`.
... 
...         Args:
...             input_length: the length of the input data.
...             output_length: the length of the output data.
...             config: the config for the model
...         """
...         super().__init__(input_length, output_length, config)
...     
...     def set_seed(self, seed: int) -> None:
...         """Sets the seed for the model. For this model, this is not required."""  
...         pass
... 
...     def train(self, train: list[npt.NDArray[np.float64]], val: list[npt.NDArray[np.float64]]) -> None:
...         pass
... 
...     def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
...         self.validate_prediction_input(data)
...         return np.full((data.shape[0], self.output_length, data.shape[2]), 0.0)
... 
... 
>>> def register() -> None:
...     factory.register(
...         "ZeroPredictor",
...         ZeroPredictorConfig,
...         ZeroPredictor
...     )