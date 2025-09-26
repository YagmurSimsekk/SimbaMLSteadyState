"""Steady-state data generation commands."""

import click
import logging
import sys
from pathlib import Path

from simba_ml.simulation.generators.enhanced_steady_state_generator import SBMLSteadyStateGenerator
from simba_ml.simulation.system_model.sbml_system_model import SBMLSystemModel
from simba_ml.simulation.generators.enhanced_steady_state_generator import EnhancedSteadyStateGenerator
from simba_ml.cli.utils import is_ode_ready


@click.group()
def steady_state():
    """Generate steady-state data from biological models."""
    pass


@steady_state.command()
@click.argument('sbml_file', type=click.Path(exists=True, path_type=Path))
@click.option('--samples', '-n', default=100, help='Number of steady-state samples (default: 100)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), default=Path('./steady_state_data'),
              help='Output directory (default: ./steady_state_data)')
@click.option('--solver', '-s', default='scipy', type=click.Choice(['scipy', 'newton', 'fsolve']),
              help='Numerical solver (default: scipy)')
@click.option('--tolerance', '-t', default=1e-8, type=float, help='Solver tolerance (default: 1e-8)')
@click.option('--format', '-f', 'output_format', default='csv', type=click.Choice(['csv', 'json', 'npz']),
              help='Output format (default: csv)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed progress')
def generate(sbml_file, samples, output_dir, solver, tolerance, output_format, verbose):
    """Generate steady-state data from an SBML model."""

    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)

    try:
        click.echo(click.style(f"🧮 Generating {samples} steady-state samples", fg='green', bold=True))
        click.echo(f"📄 Model: {sbml_file.name}")
        click.echo(f"🔧 Solver: {solver} (tolerance: {tolerance})")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize generator
        solver_kwargs = {
            'tolerance': tolerance,
            'max_iter': 1000
        }

        generator = SBMLSteadyStateGenerator(
            str(sbml_file),
            solver_type=solver,
            solver_kwargs=solver_kwargs,
            perturbation_std=0.1
        )

        # Show model info
        click.echo(f"\n📊 Model Analysis:")
        click.echo(f"  • Total species: {len(generator.species_data)}")

        # Count dynamic vs boundary species
        dynamic_species = [
            sp_id for sp_id, sp_data in generator.species_data.items()
            if not sp_data['boundary_condition'] and not sp_data['constant']
        ]
        boundary_species = [
            sp_id for sp_id, sp_data in generator.species_data.items()
            if sp_data['boundary_condition'] or sp_data['constant']
        ]

        click.echo(f"  • Dynamic species: {len(dynamic_species)}")
        click.echo(f"  • Boundary species: {len(boundary_species)}")
        click.echo(f"  • Reactions: {len(generator.reactions_data)}")
        click.echo(f"  • Parameters: {len(generator.parameters_data)}")

        if len(dynamic_species) == 0:
            click.echo(click.style("⚠️  Warning: No dynamic species found!", fg='yellow'))
            click.echo("This model may not be suitable for steady-state generation.")
            return

        # Generate data with progress bar
        click.echo(f"\n🔄 Generating steady-state samples...")

        with click.progressbar(length=samples, label='Solving') as bar:
            def progress_callback(completed):
                bar.update(completed - bar.pos)

            # Note: The actual generator doesn't support progress callbacks yet,
            # so we'll update the bar manually
            data = generator.generate_steady_states(n_samples=samples, max_attempts_per_sample=3)
            bar.update(samples)

        if data.empty:
            click.echo(click.style("❌ Failed to generate any steady-state solutions", fg='red'))
            click.echo("Try adjusting solver parameters or check model validity.")
            sys.exit(1)

        # Save data
        model_name = sbml_file.stem
        if output_format == 'csv':
            output_file = output_dir / f"{model_name}_steady_states.csv"
            data.to_csv(output_file, index=False)
        elif output_format == 'json':
            output_file = output_dir / f"{model_name}_steady_states.json"
            data.to_json(output_file, orient='records', indent=2)
        elif output_format == 'npz':
            output_file = output_dir / f"{model_name}_steady_states.npz"
            import numpy as np
            np.savez_compressed(output_file, **{col: data[col].values for col in data.columns})

        # Success summary
        click.echo(click.style(f"\n✅ Generation completed!", fg='green', bold=True))
        click.echo(f"📊 Generated: {len(data)} successful samples")
        click.echo(f"📁 Saved to: {output_file}")

        # Data summary
        if verbose:
            species_cols = [c for c in data.columns if c.startswith('species_')]
            param_cols = [c for c in data.columns if c.startswith('param_')]

            click.echo(f"\n📈 Dataset Summary:")
            click.echo(f"  • Species concentrations: {len(species_cols)} columns")
            click.echo(f"  • Parameters: {len(param_cols)} columns")

            if species_cols:
                # Show concentration ranges
                click.echo(f"  • Concentration ranges:")
                for col in species_cols[:5]:  # Show first 5
                    min_val = data[col].min()
                    max_val = data[col].max()
                    click.echo(f"    - {col.replace('species_', '')}: {min_val:.2e} - {max_val:.2e}")
                if len(species_cols) > 5:
                    click.echo(f"    ... and {len(species_cols) - 5} more")

    except KeyboardInterrupt:
        click.echo(click.style("\n⏹️  Generation cancelled by user", fg='yellow'))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"❌ Generation failed: {e}", fg='red'), err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@steady_state.command()
@click.argument('sbml_file', type=click.Path(exists=True, path_type=Path))
def analyze(sbml_file):
    """Analyze an SBML model for steady-state generation suitability."""
    try:
        click.echo(click.style(f"🔍 Analyzing model: {sbml_file.name}", fg='blue', bold=True))

        # Parse model
        from simba_ml.sbml_parser.main_parser import MainSBMLParser

        parser = MainSBMLParser(str(sbml_file))
        sbml_data = parser.process()

        # Analyze species
        total_species = len(sbml_data['species'])
        boundary_species = sum(1 for sp in sbml_data['species'] if sp.get('boundary_condition', False))
        constant_species = sum(1 for sp in sbml_data['species'] if sp.get('constant', False))
        dynamic_species = total_species - boundary_species - constant_species

        click.echo(f"\n📊 Species Analysis:")
        click.echo(f"  • Total species: {total_species}")
        click.echo(f"  • Dynamic species: {dynamic_species}")
        click.echo(f"  • Boundary species: {boundary_species}")
        click.echo(f"  • Constant species: {constant_species}")

        # Analyze reactions
        total_reactions = len(sbml_data['reactions'])
        reactions_with_kinetics = sum(1 for r in sbml_data['reactions'] if r.get('kinetic_law'))

        click.echo(f"\n⚗️  Reaction Analysis:")
        click.echo(f"  • Total reactions: {total_reactions}")
        click.echo(f"  • With kinetic laws: {reactions_with_kinetics}")
        click.echo(f"  • Without kinetic laws: {total_reactions - reactions_with_kinetics}")

        # Suitability assessment
        click.echo(f"\n🎯 Steady-State Suitability:")

        issues = []
        warnings = []

        if dynamic_species == 0:
            issues.append("No dynamic species found")
        elif dynamic_species < 3:
            warnings.append(f"Only {dynamic_species} dynamic species (consider >3 for interesting dynamics)")
        else:
            click.echo(f"  ✅ {dynamic_species} dynamic species - suitable for steady-state analysis")

        # Check for ODE readiness (reactions with kinetic laws OR rate rules)
        rate_rules = sum(1 for rule in sbml_data.get('rules', []) if rule.get('type') == 'rate')

        if not is_ode_ready(sbml_data):
            issues.append("No kinetic laws or rate rules found")
        elif rate_rules > 0 and total_reactions == 0:
            click.echo(f"  ✅ Rule-based ODE model with {rate_rules} rate rules")
        elif reactions_with_kinetics == 0 and rate_rules > 0:
            warnings.append(f"Model uses {rate_rules} rate rules but reactions lack kinetic laws")
        elif reactions_with_kinetics < total_reactions and rate_rules == 0:
            warnings.append(f"{total_reactions - reactions_with_kinetics} reactions missing kinetic laws")
        else:
            click.echo(f"  ✅ All reactions have kinetic laws")

        # Check initial values
        species_with_initial = sum(1 for sp in sbml_data['species']
                                 if sp.get('initial_concentration') is not None or sp.get('initial_amount') is not None)
        if species_with_initial < total_species:
            warnings.append(f"{total_species - species_with_initial} species missing initial values")
        else:
            click.echo(f"  ✅ All species have initial values")

        # Overall assessment
        if issues:
            click.echo(f"\n❌ Issues found ({len(issues)}):")
            for issue in issues:
                click.echo(f"  • {issue}")
            click.echo(f"  → Model is NOT suitable for steady-state generation")
        elif warnings:
            click.echo(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                click.echo(f"  • {warning}")
            click.echo(f"  → Model is suitable but may have limitations")
        else:
            click.echo(f"\n✅ Model is well-suited for steady-state generation!")

        # Recommendations
        click.echo(f"\n💡 Recommendations:")
        if dynamic_species > 0:
            click.echo(f"  • Try generating 100-1000 samples initially")
            click.echo(f"  • Use 'scipy' solver for best reliability")
            click.echo(f"  • Consider tolerance 1e-8 for high accuracy")

        if boundary_species > 0:
            click.echo(f"  • {boundary_species} boundary species will remain constant during simulation")

    except Exception as e:
        click.echo(click.style(f"Analysis failed: {e}", fg='red'), err=True)
        raise click.Abort()


@steady_state.command()
@click.argument('data_file', type=click.Path(exists=True, path_type=Path))
@click.option('--plot-species', '-p', multiple=True, help='Species to plot (can be used multiple times)')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Save plot to file')
def visualize(data_file, plot_species, output):
    """Visualize steady-state data distributions."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np

        click.echo(f"📊 Loading data from: {data_file.name}")

        # Load data
        if data_file.suffix == '.csv':
            data = pd.read_csv(data_file)
        elif data_file.suffix == '.json':
            data = pd.read_json(data_file)
        else:
            click.echo("Unsupported file format. Use CSV or JSON files.")
            return

        # Get species columns
        species_cols = [c for c in data.columns if c.startswith('species_')]

        if not species_cols:
            click.echo("No species data found in file.")
            return

        # Filter species to plot
        if plot_species:
            species_to_plot = [f"species_{sp}" if not sp.startswith('species_') else sp
                             for sp in plot_species]
            species_to_plot = [sp for sp in species_to_plot if sp in species_cols]
        else:
            species_to_plot = species_cols[:6]  # Plot first 6 species

        if not species_to_plot:
            click.echo("No valid species found to plot.")
            return

        click.echo(f"📈 Creating plots for {len(species_to_plot)} species...")

        # Create subplots
        n_species = len(species_to_plot)
        n_cols = min(3, n_species)
        n_rows = (n_species + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_species == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        # Plot distributions
        for i, species in enumerate(species_to_plot):
            ax = axes[i]
            values = data[species]

            # Create histogram
            ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(species.replace('species_', ''), fontsize=10)
            ax.set_xlabel('Concentration')
            ax.set_ylabel('Frequency')

            # Add statistics
            mean_val = values.mean()
            std_val = values.std()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2e}')
            ax.legend(fontsize=8)

        # Hide empty subplots
        for i in range(len(species_to_plot), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if output:
            plt.savefig(output, dpi=300, bbox_inches='tight')
            click.echo(f"📁 Plot saved to: {output}")
        else:
            plt.show()

        click.echo("✅ Visualization completed!")

    except ImportError:
        click.echo(click.style("Error: matplotlib is required for visualization", fg='red'))
        click.echo("Install with: pip install matplotlib")
    except Exception as e:
        click.echo(click.style(f"Visualization failed: {e}", fg='red'), err=True)
        raise click.Abort()
