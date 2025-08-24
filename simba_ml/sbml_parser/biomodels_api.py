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
    
    def search_models(self, query: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """
        Search for models in BioModels Database.
        
        Args:
            query: Search query (model name, author, keywords)
            limit: Maximum number of results to return (API minimum is 10)
            offset: Number of results to skip
            
        Returns:
            Dictionary containing search results
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
        return response.json()
    
    def get_model_files(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about files available for a model.
        
        Args:
            model_id: Model identifier (e.g., "BIOMD0000000012", "Malkov2020")
            
        Returns:
            Dictionary with file information
        """
        url = f"{self.BASE_URL}/model/files/{model_id}"
        params = {'format': 'json'}
        
        response = requests.get(url, params=params)
        if response.status_code == 404:
            raise ValueError(f"Model {model_id} not found in BioModels Database")
        response.raise_for_status()
        
        return response.json()
    
    def download_model(self, model_id: str, output_dir: Optional[str] = None, 
                      filename: Optional[str] = None) -> str:
        """
        Download SBML model file.
        
        Args:
            model_id: Model identifier
            output_dir: Directory to save the model. If None, saves to current directory.
            filename: Specific filename to download. If None, downloads the main SBML file.
            
        Returns:
            Path to downloaded file
        """
        # Set output directory
        output_path = Path(output_dir) if output_dir else Path(".")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get model file information if filename not specified
        if not filename:
            model_info = self.get_model_files(model_id)
            
            # Find SBML file in main files
            sbml_files = [f for f in model_info.get('main', []) 
                         if f['name'].endswith(('.xml', '.sbml'))]
            
            if not sbml_files:
                raise ValueError(f"No SBML file found for model {model_id}")
            
            filename = sbml_files[0]['name']
        
        # Download the file
        download_url = f"{self.BASE_URL}/model/download/{model_id}"
        params = {'filename': filename}
        
        print(f"Downloading {model_id}/{filename} from BioModels Database...")
        response = requests.get(download_url, params=params)
        response.raise_for_status()
        
        # Save file
        output_file = output_path / filename
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded: {output_file}")
        return str(output_file)
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with model information
        """
        url = f"{self.BASE_URL}/model/{model_id}"
        params = {'format': 'json'}
        
        response = requests.get(url, params=params)
        if response.status_code == 404:
            raise ValueError(f"Model {model_id} not found in BioModels Database")
        response.raise_for_status()
        
        return response.json()


def download_biomodel(model_id: str, output_dir: Optional[str] = None) -> str:
    """
    Convenience function to download a BioModel.
    
    Args:
        model_id: Model identifier (e.g., "BIOMD0000000012", "Malkov2020")
        output_dir: Directory to save the model
        
    Returns:
        Path to downloaded SBML file
    """
    api = BioModelsAPI()
    return api.download_model(model_id, output_dir)


def search_biomodels(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience function to search BioModels.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of model information
    """
    api = BioModelsAPI()
    results = api.search_models(query, max(limit, 10))  # API minimum is 10
    models = results.get('models', [])
    return models[:limit]  # Trim to requested limit
