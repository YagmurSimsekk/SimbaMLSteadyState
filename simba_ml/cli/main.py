#!/usr/bin/env python3
"""
SimbaML - Machine Learning for Systems Biology

A framework for integrating prior knowledge of ODE models into machine learning
workflows through synthetic data augmentation.
"""

import click
import sys
from pathlib import Path

# Import subcommands
from simba_ml.cli.commands import sbml, biomodels, steady_state, generate, predict


@click.group()
@click.version_option()
@click.pass_context
def simba_ml(ctx):
    """
    SimbaML - Machine Learning for Systems Biology

    A framework for integrating biological models with machine learning workflows.
    Generate synthetic data from SBML models and train ML models for biological predictions.

    Examples:

        # Parse an SBML model
        simba-ml sbml parse model.xml

        # Download from BioModels
        simba-ml biomodels download BIOMD0000000505

        # Generate steady-state data
        simba-ml steady-state generate model.xml --samples 1000

        # Generate time-series data
        simba-ml generate data config.toml --samples 500
    """
    # Ensure commands are found
    ctx.ensure_object(dict)


# Add command groups
simba_ml.add_command(sbml.sbml)
simba_ml.add_command(biomodels.biomodels)
simba_ml.add_command(steady_state.steady_state)
simba_ml.add_command(generate.generate)
simba_ml.add_command(predict.predict)


def main():
    """Entry point for the simba-ml command."""
    simba_ml()


if __name__ == '__main__':
    main()
