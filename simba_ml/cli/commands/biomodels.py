"""BioModels Database integration commands."""

import click
import json
from pathlib import Path

from simba_ml.sbml_parser.biomodels_api import BioModelsAPI
from simba_ml.cli.utils import is_ode_ready


@click.group()
def biomodels():
    """Search and download models from BioModels Database."""
    pass


@biomodels.command()
@click.argument("query")
@click.option("--limit", "-l", default=10, help="Maximum number of results (default: 10)")
@click.option("--format", "-f", type=click.Choice(['table', 'json']), default='table',
              help="Output format (default: table)")
def search(query, limit, format):
    """Search BioModels Database for models."""
    try:
        api = BioModelsAPI()

        if format == 'json':
            # For programmatic use
            click.echo("Searching BioModels...")
        else:
            click.echo(click.style(f"🔍 Searching BioModels for: '{query}'", fg='blue', bold=True))

        results = api.search_models(query, limit=limit)

        if not results:
            click.echo("No models found matching your query.")
            return

        if format == 'json':
            click.echo(json.dumps(results, indent=2))
        else:
            _display_search_results(results)

    except Exception as e:
        click.echo(click.style(f"Search failed: {e}", fg='red'), err=True)
        raise click.Abort()


@biomodels.command()
@click.argument("model_id")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=Path("./biomodels"),
              help="Download directory (default: ./biomodels)")
@click.option("--parse", "-p", is_flag=True, help="Parse the model after downloading")
@click.option("--verbose", "-v", is_flag=True, help="Show download progress")
def download(model_id, output_dir, parse, verbose):
    """Download a model from BioModels Database."""
    try:
        api = BioModelsAPI()

        if verbose:
            click.echo(f"📥 Downloading model: {model_id}")

        # Download model
        local_file = api.download_model(model_id, str(output_dir))

        click.echo(click.style("✅ Download completed!", fg='green', bold=True))
        click.echo(f"📁 Saved to: {local_file}")

        if parse:
            if verbose:
                click.echo("🔍 Parsing downloaded model...")

            # Parse the model
            from simba_ml.sbml_parser.main_parser import MainSBMLParser

            parser = MainSBMLParser(local_file)
            result = parser.process()
            info = result['sbml_info']

            click.echo(f"\n📋 Model Summary:")
            click.echo(f"  • Name: {info['model_name']}")
            click.echo(f"  • Species: {info['num_species']}")
            click.echo(f"  • Reactions: {info['num_reactions']}")
            click.echo(f"  • Parameters: {info['num_parameters']}")

    except Exception as e:
        click.echo(click.style(f"Download failed: {e}", fg='red'), err=True)
        raise click.Abort()


@biomodels.command()
@click.argument("model_id")
def info(model_id):
    """Get detailed information about a BioModels entry."""
    try:
        api = BioModelsAPI()

        click.echo(f"🔍 Fetching information for: {model_id}")

        model_info = api.get_model_info(model_id)

        if not model_info:
            click.echo("Model not found.")
            return

        _display_model_info(model_info)

    except Exception as e:
        click.echo(click.style(f"Failed to get model info: {e}", fg='red'), err=True)
        raise click.Abort()


@biomodels.command()
@click.argument("query")
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--limit", "-l", default=10, help="Maximum models to download (default: 10)")
@click.option("--parse", "-p", is_flag=True, help="Parse models after downloading")
def batch_download(query, output_dir, limit, parse):
    """Search and download multiple models matching a query."""
    try:
        api = BioModelsAPI()

        click.echo(f"🔍 Searching for models: '{query}'")
        results = api.search_models(query, limit=limit)

        if not results:
            click.echo("No models found.")
            return

        click.echo(f"📥 Downloading {len(results)} models to {output_dir}")

        successful_downloads = []
        failed_downloads = []

        with click.progressbar(results, label='Downloading models') as bar:
            for model in bar:
                try:
                    model_id = model['model_id']
                    local_file = api.download_model(model_id, str(output_dir))
                    successful_downloads.append((model_id, local_file))
                except Exception as e:
                    failed_downloads.append((model_id, str(e)))

        # Summary
        click.echo(f"\n✅ Downloaded: {len(successful_downloads)} models")
        if failed_downloads:
            click.echo(f"❌ Failed: {len(failed_downloads)} models")
            for model_id, error in failed_downloads:
                click.echo(f"  • {model_id}: {error}")

        # Parse if requested
        if parse and successful_downloads:
            click.echo(f"\n🔍 Parsing downloaded models...")

            from simba_ml.sbml_parser.main_parser import MainSBMLParser

            parsed_summary = []
            for model_id, local_file in successful_downloads:
                try:
                    parser = MainSBMLParser(local_file)
                    result = parser.process()
                    info = result['sbml_info']
                    parsed_summary.append({
                        'model_id': model_id,
                        'species': info['num_species'],
                        'reactions': info['num_reactions'],
                        'ode_ready': is_ode_ready(result)
                    })
                except Exception:
                    parsed_summary.append({
                        'model_id': model_id,
                        'species': 'Error',
                        'reactions': 'Error',
                        'ode_ready': False
                    })

            # Display parsing summary
            click.echo(f"\n📊 Parsing Summary:")
            click.echo(f"{'Model ID':<20} {'Species':<8} {'Reactions':<10} {'ODE Ready'}")
            click.echo("-" * 50)
            for summary in parsed_summary:
                ode_status = "✅" if summary['ode_ready'] else "❌"
                click.echo(f"{summary['model_id']:<20} {summary['species']:<8} {summary['reactions']:<10} {ode_status}")

    except Exception as e:
        click.echo(click.style(f"Batch download failed: {e}", fg='red'), err=True)
        raise click.Abort()


def _display_search_results(results):
    """Display search results in a formatted table."""
    click.echo(f"\n📊 Found {len(results)} models:")
    click.echo("-" * 80)
    click.echo(f"{'Model ID':<15} {'Name':<50} {'Format':<10}")
    click.echo("-" * 80)

    for model in results:
        model_id = model.get('model_id', 'Unknown')
        name = model.get('name', 'Unknown')[:47] + "..." if len(model.get('name', '')) > 50 else model.get('name', 'Unknown')
        format_info = model.get('format', 'SBML')

        click.echo(f"{model_id:<15} {name:<50} {format_info:<10}")

    click.echo("-" * 80)
    click.echo(f"\n💡 Use 'simba-ml biomodels download MODEL_ID' to download a model")


def _display_model_info(model_info):
    """Display detailed model information."""
    click.echo(f"\n📋 BioModels Entry Information")
    click.echo("=" * 50)

    # Basic info
    click.echo(f"Model ID: {model_info.get('model_id', 'Unknown')}")
    click.echo(f"Name: {model_info.get('name', 'Unknown')}")

    if model_info.get('description'):
        click.echo(f"Description: {model_info['description'][:200]}...")

    if model_info.get('publication'):
        pub = model_info['publication']
        click.echo(f"\n📚 Publication:")
        if pub.get('title'):
            click.echo(f"  Title: {pub['title']}")
        if pub.get('authors'):
            click.echo(f"  Authors: {pub['authors']}")
        if pub.get('journal'):
            click.echo(f"  Journal: {pub['journal']}")

    if model_info.get('curation_status'):
        click.echo(f"\n✅ Curation: {model_info['curation_status']}")

    if model_info.get('submitter'):
        click.echo(f"👤 Submitter: {model_info['submitter']}")

    if model_info.get('submission_date'):
        click.echo(f"📅 Submitted: {model_info['submission_date']}")

    if model_info.get('file_size'):
        click.echo(f"📁 Size: {model_info['file_size']}")

    click.echo(f"\n🔗 BioModels URL: https://www.ebi.ac.uk/biomodels/{model_info.get('model_id', '')}")
