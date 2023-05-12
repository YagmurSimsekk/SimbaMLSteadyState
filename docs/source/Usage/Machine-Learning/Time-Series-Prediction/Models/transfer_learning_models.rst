Transfer Learning Time-Series Models
====================================

SimbaML provides multiple machine learning models out-of-the-box for transfer learning. These models can be used with with the transfer learning pipeline.

.. note::
    Transfer learning time-series models in **SimbaML** are models that are first trained on synthetic and then fine-tuned on real-world data. The models train() function is called twice. In constrast, plain time-series models are trained once on either synthetic or a mix of synthetic and real-world data.

Provided Models for Transfer Learning
-------------------------------------

The following model can be used out-of-the-boxed for the transfer learning pipeline:

* Keras
    * Deep Neural Network: "KerasDenseNeuralNetworkTransferLearning"

Further, we provide one benchmark model:

* LastValuePredictor: "LastValuePredictor"

Include Own Model for Transfer Learning
---------------------------------------

Besides the provided models, SimbaML allows for the effortless integration of any other machine learning models, for example, PyTorch Lightning and Keras

.. note::
    Before applying your own model to the transfer learning pipeline, make sure that the model's weights are not reset when the train() function is called the second time. This is, for example, the case for Scikit-learn models.