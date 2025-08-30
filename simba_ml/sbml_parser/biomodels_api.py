"""
BioModels Database API integration for downloading SBML models.

Based on BioModels REST API documentation at:
https://www.ebi.ac.uk/biomodels/docs/
"""

import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


class BioModelsAPI:
    """Client for BioModels Database REST API."""

    BASE_URL = "https://www.ebi.ac.uk/biomodels"

    def search_models(self, query: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Search for models in BioModels Database.

        Args:
            query: Search query (model name, author, keywords)
            limit: Maximum number of results to return (API minimum is 10)
            offset: Number of results to skip

        Returns:
            List of model dictionaries
        """
        url = f"{self.BASE_URL}/search"
        params = {
            'query': query,
            'numResults': max(limit, 10),  # BioModels API minimum is 10
            'offset': offset,
            'format': 'json'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract models from the response
        if isinstance(data, dict) and 'models' in data:
            models = data['models'][:limit]  # Limit to requested number
            # Standardize the format
            return [
                {
                    'model_id': model.get('id', ''),
                    'name': model.get('name', ''),
                    'format': model.get('format', 'SBML'),
                    'submitter': model.get('submitter', ''),
                    'submission_date': model.get('submissionDate', ''),
                    'last_modified': model.get('lastModified', ''),
                    'url': model.get('url', '')
                }
                for model in models
            ]
        else:
            return []

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_id: Model identifier (e.g., "BIOMD0000000012", "MODEL1312040000")

        Returns:
            Dictionary with model information
        """
        # Try to get model information from the model endpoint
        url = f"{self.BASE_URL}/{model_id}?format=json"

        try:
            response = requests.get(url)
            if response.status_code == 404:
                raise ValueError(f"Model {model_id} not found in BioModels Database")
            response.raise_for_status()
            return response.json()
        except Exception:
            # Fallback: basic info from model ID
            return {
                'model_id': model_id,
                'name': f'Model {model_id}',
                'format': 'SBML'
            }

    def download_model(self, model_id: str, output_dir: Optional[str] = None) -> str:
        """
        Download SBML model file.

        Args:
            model_id: Model identifier (e.g., "BIOMD0000000505")
            output_dir: Directory to save the model. If None, saves to current directory.

        Returns:
            Path to downloaded file
        """
        # Set output directory
        output_path = Path(output_dir) if output_dir else Path(".")
        output_path.mkdir(parents=True, exist_ok=True)

        # Get model info to find the actual filename
        try:
            model_info = self.get_model_info(model_id)

            # Extract main SBML file name from files section
            files = model_info.get('files', {})
            main_files = files.get('main', [])

            if main_files and len(main_files) > 0:
                filename = main_files[0].get('name')
                if not filename:
                    # Fallback to standard pattern
                    filename = f"{model_id}_url.xml"
            else:
                # Fallback to standard pattern
                filename = f"{model_id}_url.xml"

        except Exception:
            # If we can't get model info, use standard pattern
            filename = f"{model_id}_url.xml"

        # Download the file using BioModels standard download URL
        download_url = f"{self.BASE_URL}/model/download/{model_id}"
        params = {'filename': filename}

        try:
            response = requests.get(download_url, params=params, allow_redirects=True)
            response.raise_for_status()

            # Save file with original filename
            output_file = output_path / filename
            with open(output_file, 'wb') as f:
                f.write(response.content)

            return str(output_file)

        except requests.RequestException as e:
            raise ValueError(f"Failed to download model {model_id}: {e}")
