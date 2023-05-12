Quickstart
============

Visualization of Problems
-------------------------

You can visualize problems.

.. code-block:: bash
    :caption: Start the Problem problem_viewer
    :linenos:

    simba_ml run-problem_viewer --module simba_ml.example_problems.trigonometry
    # or
    simba_ml run-problem_viewer --module my_own_costum_module

You can find available example_problems in the `example_problems` package.

.. note::
    The module you want to vizualize should create a `SystemModel` named `sm`.

Generation of CSV Files
-----------------------

In order to generate CSV files of a problem, you need to create a config file, which has a `SystemModel` called `sm`.
You can then generate CSV files by calling 
`simba_ml generate [GENERATOR] --config-file [CONFIG_FILE containing the SystemModel] -n [Number of samples to generate] --output-path [OUTPUT DIR]`.