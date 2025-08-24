import click
from simba_ml.sbml_parser.biomodels_api import BioModelsAPI, download_biomodel, search_biomodels


@click.group()
def biomodels():
    """BioModels Database commands."""
    pass


@biomodels.command()
@click.argument("model_id", type=str)
@click.option("--output-dir", "-o", default="./biomodels_downloads", help="Output directory for downloaded model")
def download(model_id, output_dir):
    """Download an SBML model from BioModels Database."""
    try:
        file_path = download_biomodel(model_id, output_dir)
        click.echo(click.style(f"✅ Downloaded: {file_path}", fg='green'))
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'), err=True)
        raise click.Abort()


@biomodels.command()
@click.argument("query", type=str)
@click.option("--limit", "-l", default=10, help="Maximum number of results")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed information")
def search(query, limit, detailed):
    """Search for models in BioModels Database."""
    try:
        models = search_biomodels(query, limit)
        
        if not models:
            click.echo(f"No models found for query: {query}")
            return
        
        click.echo(click.style(f"Found {len(models)} models for '{query}':", fg='cyan', bold=True))
        click.echo()
        
        for i, model in enumerate(models, 1):
            model_id = model.get('id', 'unknown')
            name = model.get('name', 'No name available')
            
            click.echo(f"{i}. {click.style(model_id, fg='blue', bold=True)}")
            click.echo(f"   {name}")
            
            if detailed:
                authors = model.get('submitter', 'Unknown authors')
                publication = model.get('publication', {})
                pub_year = publication.get('year', 'Unknown year')
                
                click.echo(f"   Authors: {authors}")
                click.echo(f"   Year: {pub_year}")
                
                if publication.get('title'):
                    title = publication['title'][:100] + ('...' if len(publication['title']) > 100 else '')
                    click.echo(f"   Publication: {title}")
            
            click.echo()
            
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'), err=True)
        raise click.Abort()


@biomodels.command()
@click.argument("model_id", type=str)
def info(model_id):
    """Get information about a specific model."""
    try:
        api = BioModelsAPI()
        
        # Get model info
        model_info = api.get_model_info(model_id)
        files_info = api.get_model_files(model_id)
        
        click.echo(click.style(f"Model Information: {model_id}", fg='cyan', bold=True))
        click.echo("=" * 50)
        
        name = model_info.get('name', 'No name available')
        click.echo(f"Name: {name}")
        
        publication = model_info.get('publication', {})
        if publication:
            click.echo(f"Publication: {publication.get('title', 'No title')}")
            click.echo(f"Authors: {publication.get('authors', 'Unknown')}")
            click.echo(f"Year: {publication.get('year', 'Unknown')}")
        
        # Show available files
        click.echo()
        click.echo(click.style("Available Files:", fg='yellow', bold=True))
        
        main_files = files_info.get('main', [])
        if main_files:
            click.echo("Main files:")
            for f in main_files:
                size = f.get('fileSize', 'unknown size')
                click.echo(f"  • {f['name']} ({size} bytes)")
        
        additional_files = files_info.get('additional', [])
        if additional_files:
            click.echo("Additional files:")
            for f in additional_files[:5]:  # Show first 5
                size = f.get('fileSize', 'unknown size')
                desc = f.get('description', 'No description')
                click.echo(f"  • {f['name']} ({size} bytes) - {desc}")
            if len(additional_files) > 5:
                click.echo(f"  ... and {len(additional_files) - 5} more files")
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'), err=True)
        raise click.Abort()


@biomodels.command()
@click.argument("model_id", type=str)
@click.option("--output-dir", "-o", default="./biomodels_downloads", help="Output directory")
def download_and_parse(model_id, output_dir):
    """Download a model and immediately parse it."""
    try:
        # Download model
        file_path = download_biomodel(model_id, output_dir)
        click.echo(click.style(f"✅ Downloaded: {file_path}", fg='green'))
        
        # Parse model
        click.echo()
        click.echo("Parsing model...")
        
        from simba_ml.sbml_parser.main_parser import MainSBMLParser
        from simba_ml.sbml_parser.ml_exporter import SBMLMLExporter
        
        parser = MainSBMLParser(file_path)
        result = parser.process()
        exporter = SBMLMLExporter(result)
        
        # Show basic info
        info = result['sbml_info']
        click.echo(click.style(f"Model: {info['model_name']}", fg='blue', bold=True))
        click.echo(f"SBML Level: {info['level']}, Version: {info['version']}")
        click.echo(f"Species: {info['num_species']}")
        click.echo(f"Reactions: {info['num_reactions']}")
        click.echo(f"Parameters: {info['num_parameters']}")
        
        # Check if suitable for ODE
        has_kinetic_laws = any(r.get('kinetic_law') is not None for r in result['reactions'])
        if has_kinetic_laws:
            click.echo(click.style("✅ ODE Ready: Model contains kinetic laws", fg='green'))
        else:
            click.echo(click.style("⚠️  No kinetic laws found", fg='yellow'))
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'), err=True)
        raise click.Abort()
