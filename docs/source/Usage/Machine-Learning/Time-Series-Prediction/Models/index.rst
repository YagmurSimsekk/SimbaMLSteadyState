Time-Series Prediction Models
=============================

SimbaML provides multiple machine learning models out-of-the-box for time-series-prediction. Besides that, SimbaML allows for the effortless integration of any other machine learning models, for example, PyTorch Lightning, Keras, and Scikit-learn.

In SimbaML, there are two types of time-series prediction models: First, we run pipelines on common time-series prediction models. Those models are trained once either on synthetic data only (see synthetic data pipeline) or on a mix of synthetic and real-world data (see mixed data pipeline). Besides, we allow users to train transfer learning models, that are first trained on synthetic data and then (fine-tuned) on real-world data (see transfer learning pipeline).

.. toctree::
    :maxdepth: 2

    time_series_models
    transfer_learning_models
