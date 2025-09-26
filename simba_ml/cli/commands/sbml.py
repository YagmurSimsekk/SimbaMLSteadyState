"""SBML parsing and analysis commands."""

import click
import json
import os
from pathlib import Path

from simba_ml.sbml_parser.main_parser import MainSBMLParser, UnsupportedSBMLVersionError, SBMLParsingError
from simba_ml.sbml_parser.ml_exporter import SBMLExporter
from simba_ml.cli.utils import is_ode_ready


@click.group()
def sbml():
    """Parse and analyze SBML models."""
    pass


@sbml.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed parsing information")
@click.option("--species-limit", "-s", default=5, help="Number of species to display (default: 5)")
@click.option("--reactions-limit", "-r", default=5, help="Number of reactions to display (default: 5)")
@click.option("--export", "-e", type=click.Choice(['csv']), help="Export data in CSV format (only format supported)")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=Path("./sbml_exports"),
              help="Output directory for exports (default: ./sbml_exports)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress visual output")
def parse(file, verbose, species_limit, reactions_limit, export, output_dir, quiet):
    """Parse an SBML file and display model information."""
    try:
        sbml_parser = MainSBMLParser(str(file))
        result = sbml_parser.process()

        if quiet:
            # JSON output for programmatic use
            info = result['sbml_info']
            summary = {
                'model_id': info['model_id'],
                'model_name': info['model_name'],
                'sbml_level': info['level'],
                'sbml_version': info['version'],
                'num_species': info['num_species'],
                'num_reactions': info['num_reactions'],
                'num_parameters': info['num_parameters'],
                'num_compartments': info['num_compartments'],
                'ode_ready': is_ode_ready(result)
            }
            click.echo(json.dumps(summary, indent=2))
            return

        _display_model_info(result, file, species_limit, reactions_limit, verbose)

        # Handle export if requested
        if export:
            if not quiet:
                click.echo(f"\n🔬 Exporting data...")

            try:
                exporter = SBMLExporter(result)
                exported_files = exporter.export_to_files(str(output_dir), format=export)

                if not quiet:
                    click.echo(click.style(f"📁 Data exported to: {output_dir}", fg='green'))
                    for data_type, file_path in exported_files.items():
                        click.echo(f"  • {data_type}: {Path(file_path).name}")

                    # Show ML statistics (dataset summary)
                    ml_dataset = exporter.get_ml_dataset()
                    click.echo(f"\n📊 Export Summary:")
                    if 'matrices' in ml_dataset:
                        S = ml_dataset['matrices']['stoichiometry']
                        A = ml_dataset['matrices']['adjacency']
                        click.echo(f"  • Stoichiometry matrix: {S.shape}")
                        click.echo(f"  • Adjacency matrix: {A.shape}")
                        click.echo(f"  • Network density: {(A.sum() / (A.shape[0] * A.shape[1]) * 100):.1f}%")
                else:
                    # Quiet mode: just print the export location
                    main_file = list(exported_files.values())[0] if exported_files else str(output_dir)
                    click.echo(f"Exported to: {main_file}")

            except Exception as e:
                if not quiet:
                    click.echo(click.style(f"Export failed: {e}", fg='red'), err=True)
                else:
                    click.echo(f"Export error: {e}", err=True)

    except (SBMLParsingError, UnsupportedSBMLVersionError) as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"Unexpected error: {e}", fg='red'), err=True)
        raise click.Abort()


@sbml.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "-f", type=click.Choice(['csv']), default='csv',
              help="Export format (only CSV supported)")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=Path("./sbml_exports"),
              help="Output directory (default: ./sbml_exports)")
@click.option("--verbose", "-v", is_flag=True, help="Show export details")
def export(file, format, output_dir, verbose):
    """Export SBML model matrices in CSV format for steady-state workflows."""
    try:
        if verbose:
            click.echo(f"Parsing SBML file: {file}")

        sbml_parser = MainSBMLParser(str(file))
        result = sbml_parser.process()

        if verbose:
            click.echo(f"Creating {format} export...")

        exporter = SBMLExporter(result)
        exported_files = exporter.export_to_files(str(output_dir), format=format)

        click.echo(click.style("✅ Export completed!", fg='green', bold=True))
        click.echo(f"📁 Output directory: {output_dir}")

        for data_type, file_path in exported_files.items():
            click.echo(f"  • {data_type}: {Path(file_path).name}")

        if verbose:
            # Show dataset statistics
            ml_dataset = exporter.get_ml_dataset()
            click.echo("\n📊 Dataset Summary:")
            if 'matrices' in ml_dataset:
                S = ml_dataset['matrices']['stoichiometry']
                A = ml_dataset['matrices']['adjacency']
                click.echo(f"  • Stoichiometry matrix: {S.shape}")
                click.echo(f"  • Adjacency matrix: {A.shape}")
                click.echo(f"  • Network density: {(A.sum() / (A.shape[0] * A.shape[1]) * 100):.1f}%")

    except Exception as e:
        click.echo(click.style(f"Export failed: {e}", fg='red'), err=True)
        raise click.Abort()




@sbml.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def validate(file):
    """Validate SBML file and check for common issues."""
    try:
        sbml_parser = MainSBMLParser(str(file))
        result = sbml_parser.process()

        click.echo(click.style("🔍 SBML Validation Results", fg='blue', bold=True))
        click.echo("=" * 50)

        # Basic validation
        click.echo("✅ File parsed successfully")
        click.echo("✅ SBML structure is valid")

        # Check for common issues
        issues = []
        warnings = []

        # Check for missing initial values
        species_without_initial = [
            sp for sp in result['species']
            if sp.get('initial_concentration') is None and sp.get('initial_amount') is None
        ]
        if species_without_initial:
            warnings.append(f"{len(species_without_initial)} species missing initial values")

        # Check for reactions without kinetic laws
        reactions_without_kinetics = [
            rxn for rxn in result['reactions']
            if not rxn.get('kinetic_law')
        ]
        if reactions_without_kinetics:
            warnings.append(f"{len(reactions_without_kinetics)} reactions missing kinetic laws")

        # Check for empty reactions
        empty_reactions = [
            rxn for rxn in result['reactions']
            if not rxn.get('reactants') and not rxn.get('products')
        ]
        if empty_reactions:
            issues.append(f"{len(empty_reactions)} reactions have no reactants or products")

        # Report results
        if not issues and not warnings:
            click.echo("✅ No issues found")
        else:
            if warnings:
                click.echo(f"\n⚠️  Warnings ({len(warnings)}):")
                for warning in warnings:
                    click.echo(f"  • {warning}")

            if issues:
                click.echo(f"\n❌ Issues ({len(issues)}):")
                for issue in issues:
                    click.echo(f"  • {issue}")

        # Recommendations
        info = result['sbml_info']
        if info['level'] == 3:
            click.echo(f"\n💡 Recommendations:")
            click.echo(f"  • SBML Level 3 detected - ensure all units are explicitly defined")
            if not any([info.get('substance_unit'), info.get('time_unit'), info.get('volume_unit')]):
                click.echo(f"  • Consider adding unit definitions for better model clarity")

    except Exception as e:
        click.echo(click.style(f"Validation failed: {e}", fg='red'), err=True)
        raise click.Abort()


def _display_model_info(result, file, species_limit, reactions_limit, verbose):
    """Display formatted model information."""
    info = result['sbml_info']

    # Header
    click.echo(click.style("=" * 60, fg='green'))
    click.echo(click.style(f"SBML Model Analysis", fg='green', bold=True))
    click.echo(click.style("=" * 60, fg='green'))
    click.echo()

    # Basic info
    click.echo(click.style(f"📄 File:", fg='blue', bold=True) + f" {file.name}")
    click.echo(click.style(f"📋 Model:", fg='blue', bold=True) + f" {info['model_name']}")
    click.echo(click.style(f"🆔 ID:", fg='blue', bold=True) + f" {info['model_id']}")
    click.echo(click.style(f"🔢 SBML:", fg='blue', bold=True) + f" Level {info['level']}, Version {info['version']}")
    click.echo()

    # Statistics
    click.echo(click.style("📊 Model Statistics:", fg='cyan', bold=True))
    click.echo(f"  • Species: {info['num_species']}")
    click.echo(f"  • Reactions: {info['num_reactions']}")
    click.echo(f"  • Parameters: {info['num_parameters']}")
    click.echo(f"  • Compartments: {info['num_compartments']}")

    # Boundary species analysis
    boundary_count = sum(1 for sp in result['species'] if sp.get('boundary_condition', False))
    dynamic_count = info['num_species'] - boundary_count

    click.echo(f"\n🧬 Species Analysis:")
    click.echo(f"  • Dynamic species: {dynamic_count}")
    click.echo(f"  • Boundary species: {boundary_count}")
    click.echo()

    # ODE readiness
    is_ode_ready_result = is_ode_ready(result)
    if is_ode_ready_result:
        # Determine the type of ODE model
        has_kinetic_laws = any(r.get('kinetic_law') for r in result['reactions'])
        has_rate_rules = any(rule.get('type') == 'rate' for rule in result.get('rules', []))

        if has_kinetic_laws and has_rate_rules:
            model_type = " (reaction + rule-based)"
        elif has_rate_rules:
            model_type = " (rule-based)"
        else:
            model_type = " (reaction-based)"

        click.echo(click.style("✅ ODE Ready:", fg='green', bold=True) + f" Suitable for simulation{model_type}")
    else:
        click.echo(click.style("⚠️  Warning:", fg='yellow', bold=True) + " No kinetic laws or rate rules found")
    click.echo()

    # Sample species
    if result['species']:
        click.echo(click.style(f"🧬 Sample Species (showing {min(species_limit, len(result['species']))}):", fg='magenta', bold=True))
        for i, species in enumerate(result['species'][:species_limit]):
            boundary = " (boundary)" if species.get('boundary_condition') else ""
            initial = ""
            if species.get('initial_concentration') is not None:
                initial = f" [C₀={species['initial_concentration']}]"
            elif species.get('initial_amount') is not None:
                initial = f" [A₀={species['initial_amount']}]"
            click.echo(f"  {i+1:2d}. {species['id']} in {species['compartment']}{boundary}{initial}")

        if len(result['species']) > species_limit:
            remaining = len(result['species']) - species_limit
            click.echo(f"      ... and {remaining} more")
        click.echo()

    # Sample reactions
    if result['reactions']:
        click.echo(click.style(f"⚗️  Sample Reactions (showing {min(reactions_limit, len(result['reactions']))}):", fg='red', bold=True))
        for i, reaction in enumerate(result['reactions'][:reactions_limit]):
            reactants = " + ".join([r['species'] for r in reaction.get('reactants', [])])
            products = " + ".join([p['species'] for p in reaction.get('products', [])])
            arrow = " ⇌ " if reaction.get('reversible', False) else " → "
            kinetic = " (kinetic ✓)" if reaction.get('kinetic_law') else " (kinetic ✗)"
            click.echo(f"  {i+1:2d}. {reaction['id']}: {reactants}{arrow}{products}{kinetic}")

        if len(result['reactions']) > reactions_limit:
            remaining = len(result['reactions']) - reactions_limit
            click.echo(f"      ... and {remaining} more")
        click.echo()

    # Verbose info
    if verbose and info.get('notes'):
        click.echo(click.style("📝 Description:", fg='white', bold=True))
        notes_preview = info['notes'][:300] + "..." if len(info['notes']) > 300 else info['notes']
        click.echo(f"  {notes_preview}")
        click.echo()

    # Always show description (not just in verbose mode)
    if not verbose and info.get('notes'):
        click.echo(click.style("📝 Description:", fg='white', bold=True))
        notes_preview = info['notes'][:200] + "..." if len(info['notes']) > 200 else info['notes']
        click.echo(f"  {notes_preview}")
        click.echo()
