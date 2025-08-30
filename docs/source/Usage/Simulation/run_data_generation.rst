Run Data Generation
======================

To run the data generation, you need execute the following command:

    $ simba_ml generate-data [generator] --config-module [config_module] --output-dir [output_dir]

The generator is the name of the generator to use.
Run `simba_ml generate-data --help` to see the list of available generators.
The config_module is the path of the module that contains the `SystemModel` for the generator.
The output_dir is the directory where the generated data will be stored.
