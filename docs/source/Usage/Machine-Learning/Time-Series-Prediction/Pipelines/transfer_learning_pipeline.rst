Transfer Learning Pipeline
==========================

To enable scalable and easy-to-run machine learning experiments on time-series data, SimbaML offers multiple pipelines covering data pre-processing, training, and evaluation of ML models.
The transfer learning pipeline can be run by passing it the corresponding config file.

Configure Pipeline
------------------

All provided machine learning pipelines of SimbaML can be configured based on config files.
This way, users can change the models that are going to be trained, their hyperparameters, specify details for preprocessing and much more.

.. literalinclude:: transfer_learning_pipeline.toml
   :language: toml
   :linenos:

Start Pipeline
--------------

   $ simba_ml start-prediction transfer_learning --config-path transfer_learning_pipeline.toml
