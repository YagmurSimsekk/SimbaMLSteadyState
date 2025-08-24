"""
ML Data Exporter for SBML models.

Converts parsed SBML data into machine learning-ready formats including:
- Structured DataFrames for species, reactions, parameters
- Network matrices (stoichiometry, adjacency)
- Feature vectors for ML training
- Export to various formats (CSV, JSON, NumPy, etc.)
"""

import pandas as pd
import numpy as np
import json
import libsbml
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class SBMLMLExporter:
    """Export SBML parsed data in machine learning-ready formats."""
    
    def __init__(self, parsed_data: Dict[str, Any]):
        """
        Initialize with parsed SBML data.
        
        Args:
            parsed_data: Output from MainSBMLParser.process()
        """
        self.data = parsed_data
        self.sbml_info = parsed_data['sbml_info']
        self.species = parsed_data['species']
        self.reactions = parsed_data['reactions']
        self.parameters = parsed_data['parameters']
        self.compartments = parsed_data['compartments']
        
        # Process species to separate dynamic from boundary
        self._process_species_types()
        self._parse_units_system()
        self._normalize_species_units()
    
    def _process_species_types(self):
        """Separate dynamic species from boundary/constant species."""
        self.dynamic_species = []
        self.boundary_species = []
        
        for sp in self.species:
            is_boundary = sp.get('boundary_condition', False)
            is_constant = sp.get('constant', False)
            
            if is_boundary or is_constant:
                self.boundary_species.append(sp)
            else:
                self.dynamic_species.append(sp)
    
    def _parse_units_system(self):
        """Parse SBML units system using libSBML."""
        # Initialize units info based on SBML Level
        level = self.sbml_info['level']
        
        if level == 2:
            # Level 2: Use SBML specification defaults
            self.units_info = {
                'substance_unit': 'mole',      # Official SBML Level 2 default
                'time_unit': 'second',         # Official SBML Level 2 default
                'volume_unit': 'litre',        # Official SBML Level 2 default
                'substance_multiplier': 1.0,
                'time_multiplier': 1.0
            }
        elif level == 3:
            # Level 3: No defaults, all units must be explicitly defined
            self.units_info = {
                'substance_unit': None,        # Must be explicitly defined
                'time_unit': None,             # Must be explicitly defined
                'volume_unit': None,           # Must be explicitly defined
                'substance_multiplier': 1.0,
                'time_multiplier': 1.0
            }
        else:
            # Fallback for other levels
            self.units_info = {
                'substance_unit': None,
                'time_unit': None,
                'volume_unit': None,
                'substance_multiplier': 1.0,
                'time_multiplier': 1.0
            }
        
        # We need to re-parse with libSBML to get units info
        # This is necessary because the main parser doesn't extract unit definitions
        if 'sbml_file_path' in self.data.get('metadata', {}):
            file_path = self.data['metadata']['sbml_file_path']
        else:
            # If file path not available, we'll work with defaults
            return
            
        try:
            reader = libsbml.SBMLReader()
            doc = reader.readSBML(file_path)
            model = doc.getModel()
            
            # Parse unit definitions
            self._extract_unit_definitions(model)
            
        except Exception as e:
            # If units parsing fails, use defaults
            pass
    
    def _extract_unit_definitions(self, model):
        """Extract unit definitions from libSBML model."""
        # Check for custom unit definitions
        for i in range(model.getNumUnitDefinitions()):
            unit_def = model.getUnitDefinition(i)
            unit_id = unit_def.getId()
            
            if unit_id in ['substance', 'time', 'volume']:
                # Parse the unit definition
                if unit_def.getNumUnits() > 0:
                    unit = unit_def.getUnit(0)  # Take first unit
                    kind = libsbml.UnitKind_toString(unit.getKind())
                    scale = unit.getScale()
                    multiplier = unit.getMultiplier()
                    
                    # Calculate actual multiplier: multiplier * 10^scale
                    actual_multiplier = multiplier * (10 ** scale)
                    
                    if unit_id == 'substance':
                        self.units_info['substance_unit'] = kind
                        self.units_info['substance_multiplier'] = actual_multiplier
                    elif unit_id == 'time':
                        self.units_info['time_unit'] = kind
                        self.units_info['time_multiplier'] = actual_multiplier
                    elif unit_id == 'volume':
                        self.units_info['volume_unit'] = kind
        
        # Set model-level units if specified
        level = self.sbml_info['level']
        
        if model.isSetSubstanceUnits():
            substance_unit_ref = model.getSubstanceUnits()
            # For Level 3, this should reference a unit definition
            if level == 3 and substance_unit_ref in ['substance'] and self.units_info['substance_unit']:
                # Keep the parsed unit definition
                pass
            else:
                self.units_info['substance_unit'] = substance_unit_ref
                
        if model.isSetTimeUnits():
            time_unit_ref = model.getTimeUnits()
            # For Level 3, this should reference a unit definition
            if level == 3 and time_unit_ref in ['time'] and self.units_info['time_unit']:
                # Keep the parsed unit definition
                pass
            else:
                self.units_info['time_unit'] = time_unit_ref
                
        if model.isSetVolumeUnits():
            volume_unit_ref = model.getVolumeUnits()
            # For Level 3, this should reference a unit definition
            if level == 3 and volume_unit_ref in ['volume'] and self.units_info['volume_unit']:
                # Keep the parsed unit definition
                pass
            else:
                self.units_info['volume_unit'] = volume_unit_ref
        
        # For Level 3, validate that all required units are explicitly defined
        if level == 3:
            self._validate_level3_units()
    
    def _validate_level3_units(self):
        """Validate that Level 3 models have required units explicitly defined."""
        import logging
        logger = logging.getLogger(__name__)
        
        missing_units = []
        
        # Check if model actually needs these units
        needs_substance = self._model_uses_concentrations_or_amounts()
        needs_time = self._model_has_kinetic_laws()
        needs_volume = self._model_uses_concentrations()
        
        if needs_substance and self.units_info['substance_unit'] is None:
            missing_units.append('substance')
        if needs_time and self.units_info['time_unit'] is None:
            missing_units.append('time') 
        if needs_volume and self.units_info['volume_unit'] is None:
            missing_units.append('volume')
        
        if missing_units:
            error_msg = (f"SBML Level 3 model missing required unit definitions: {', '.join(missing_units)}. "
                        f"Level 3 specification requires all used units to be explicitly defined.")
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _model_uses_concentrations_or_amounts(self) -> bool:
        """Check if model uses species amounts or concentrations."""
        return any(sp.get('initial_concentration') is not None or 
                  sp.get('initial_amount') is not None 
                  for sp in self.species)
    
    def _model_has_kinetic_laws(self) -> bool:
        """Check if model has kinetic laws (needs time units)."""
        return any(rxn.get('kinetic_law') is not None for rxn in self.reactions)
    
    def _model_uses_concentrations(self) -> bool:
        """Check if model uses concentrations (needs volume units)."""
        return any(sp.get('initial_concentration') is not None for sp in self.species)
    
    
    def _get_compartment_size(self, compartment_id: str) -> float:
        """Get size of a compartment by ID."""
        for comp in self.compartments:
            if comp['id'] == compartment_id:
                return comp.get('size', 1.0)
        return 1.0  # Default size if not found
    
    def _normalize_species_units(self):
        """Convert all species to concentration units for consistent ODE formulation."""
        for sp in self.species:
            compartment_size = self._get_compartment_size(sp['compartment'])
            
            # Convert to concentration if needed
            if sp.get('initial_concentration') is not None:
                # Already in concentration units
                sp['normalized_concentration'] = sp['initial_concentration']
                sp['units_type'] = 'concentration'
                
            elif sp.get('initial_amount') is not None:
                # Convert amount to concentration: [X] = amount / volume
                sp['normalized_concentration'] = sp['initial_amount'] / compartment_size
                sp['units_type'] = 'amount_converted'
                
            else:
                # No initial condition specified
                sp['normalized_concentration'] = 0.0
                sp['units_type'] = 'default'
            
            # Add units information from parsed SBML
            sp['substance_unit'] = self.units_info['substance_unit']
            sp['substance_multiplier'] = self.units_info['substance_multiplier']
            sp['time_unit'] = self.units_info['time_unit']
            sp['time_multiplier'] = self.units_info['time_multiplier']
            sp['volume_unit'] = self.units_info['volume_unit']
    
    def get_dynamic_species_concentrations(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get normalized initial concentrations for dynamic species only.
        
        Returns:
            tuple: (concentrations_array, species_ids)
                - concentrations_array: Initial concentrations for ODE system
                - species_ids: Corresponding species identifiers
        """
        concentrations = []
        species_ids = []
        
        for sp in self.dynamic_species:
            concentrations.append(sp['normalized_concentration'])
            species_ids.append(sp['id'])
        
        return np.array(concentrations), species_ids
    
    def get_boundary_species_info(self) -> List[Dict[str, Any]]:
        """
        Get information about boundary/constant species.
        
        Returns:
            list: Information about boundary species that remain constant
        """
        boundary_info = []
        for sp in self.boundary_species:
            boundary_info.append({
                'id': sp['id'],
                'concentration': sp['normalized_concentration'],
                'boundary_condition': sp.get('boundary_condition', False),
                'constant': sp.get('constant', False)
            })
        return boundary_info
        
    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Convert parsed data to pandas DataFrames.
        
        Returns:
            dict: DataFrames for different components
        """
        dataframes = {}
        
        # Species DataFrame
        if self.species:
            species_data = []
            for sp in self.species:
                species_data.append({
                    'species_id': sp['id'],
                    'name': sp['name'],
                    'compartment': sp['compartment'],
                    'initial_concentration': sp.get('initial_concentration'),
                    'initial_amount': sp.get('initial_amount'),
                    'normalized_concentration': sp.get('normalized_concentration'),
                    'units_type': sp.get('units_type'),
                    'substance_unit': sp.get('substance_unit'),
                    'substance_multiplier': sp.get('substance_multiplier'),
                    'time_unit': sp.get('time_unit'),
                    'time_multiplier': sp.get('time_multiplier'),
                    'volume_unit': sp.get('volume_unit'),
                    'boundary_condition': sp.get('boundary_condition', False),
                    'constant': sp.get('constant', False),
                    'has_only_substance_units': sp.get('has_only_substance_units', False),
                    'is_dynamic': not (sp.get('boundary_condition', False) or sp.get('constant', False))
                })
            dataframes['species'] = pd.DataFrame(species_data)
        
        # Reactions DataFrame
        if self.reactions:
            reaction_data = []
            for rxn in self.reactions:
                # Basic reaction info
                rxn_row = {
                    'reaction_id': rxn['id'],
                    'name': rxn['name'],
                    'reversible': rxn.get('reversible', False),
                    'fast': rxn.get('fast', False),
                    'has_kinetic_law': rxn.get('kinetic_law') is not None,
                    'num_reactants': len(rxn.get('reactants', [])),
                    'num_products': len(rxn.get('products', [])),
                    'num_modifiers': len(rxn.get('modifiers', []))
                }
                
                # Add kinetic law info if available
                if rxn.get('kinetic_law'):
                    kl = rxn['kinetic_law']
                    rxn_row.update({
                        'kinetic_formula': kl.get('formula'),
                        'kinetic_math': kl.get('math'),
                        'num_local_parameters': len(kl.get('parameters', []))
                    })
                
                reaction_data.append(rxn_row)
            dataframes['reactions'] = pd.DataFrame(reaction_data)
        
        # Parameters DataFrame
        if self.parameters:
            param_data = []
            for param in self.parameters:
                param_data.append({
                    'parameter_id': param['id'],
                    'name': param['name'],
                    'value': param.get('value'),
                    'units': param.get('units'),
                    'constant': param.get('constant', True)
                })
            dataframes['parameters'] = pd.DataFrame(param_data)
        
        # Compartments DataFrame
        if self.compartments:
            comp_data = []
            for comp in self.compartments:
                comp_data.append({
                    'compartment_id': comp['id'],
                    'name': comp['name'],
                    'size': comp.get('size'),
                    'spatial_dimensions': comp.get('spatial_dimensions'),
                    'constant': comp.get('constant', True)
                })
            dataframes['compartments'] = pd.DataFrame(comp_data)
        
        return dataframes
    
    def get_stoichiometry_matrix(self, dynamic_only: bool = True) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Create stoichiometry matrix for the reaction network.
        
        Args:
            dynamic_only: If True, only include non-boundary, non-constant species
        
        Returns:
            tuple: (matrix, species_ids, reaction_ids)
                - matrix: shape (n_dynamic_species, n_reactions) or (n_species, n_reactions)
                - species_ids: list of species identifiers  
                - reaction_ids: list of reaction identifiers
        """
        if dynamic_only:
            species_list = self.dynamic_species
            species_ids = [sp['id'] for sp in species_list]
        else:
            species_list = self.species
            species_ids = [sp['id'] for sp in species_list]
            
        reaction_ids = [rxn['id'] for rxn in self.reactions]
        
        # Create species index mapping
        species_idx = {sp_id: i for i, sp_id in enumerate(species_ids)}
        
        # Initialize stoichiometry matrix
        S = np.zeros((len(species_ids), len(reaction_ids)))
        
        for j, reaction in enumerate(self.reactions):
            # Add reactants (negative stoichiometry)
            for reactant in reaction.get('reactants', []):
                sp_id = reactant['species']
                if sp_id in species_idx:  # Only include if in our species list
                    i = species_idx[sp_id]
                    stoich = reactant.get('stoichiometry', 1.0)
                    S[i, j] -= stoich
            
            # Add products (positive stoichiometry)
            for product in reaction.get('products', []):
                sp_id = product['species']
                if sp_id in species_idx:  # Only include if in our species list
                    i = species_idx[sp_id]
                    stoich = product.get('stoichiometry', 1.0)
                    S[i, j] += stoich
        
        return S, species_ids, reaction_ids
    
    def get_adjacency_matrix(self, include_modifiers: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Create adjacency matrix representing species-species interactions.
        
        Args:
            include_modifiers: Whether to include modifier relationships
            
        Returns:
            tuple: (adjacency_matrix, species_ids)
                - adjacency_matrix: shape (n_species, n_species)
                - species_ids: list of species identifiers
        """
        species_ids = [sp['id'] for sp in self.species]
        species_idx = {sp_id: i for i, sp_id in enumerate(species_ids)}
        
        # Initialize adjacency matrix
        A = np.zeros((len(species_ids), len(species_ids)))
        
        for reaction in self.reactions:
            reactant_ids = [r['species'] for r in reaction.get('reactants', [])]
            product_ids = [p['species'] for p in reaction.get('products', [])]
            modifier_ids = [m['species'] for m in reaction.get('modifiers', [])] if include_modifiers else []
            
            # Reactants to products
            for reactant_id in reactant_ids:
                for product_id in product_ids:
                    if reactant_id in species_idx and product_id in species_idx:
                        i, j = species_idx[reactant_id], species_idx[product_id]
                        A[i, j] = 1
            
            # Modifiers to products (regulatory interactions)
            if include_modifiers:
                for modifier_id in modifier_ids:
                    for product_id in product_ids:
                        if modifier_id in species_idx and product_id in species_idx:
                            i, j = species_idx[modifier_id], species_idx[product_id]
                            A[i, j] = 1
        
        return A, species_ids
    
    def get_feature_vectors(self) -> Dict[str, np.ndarray]:
        """
        Extract feature vectors for ML training.
        
        Returns:
            dict: Feature vectors for different components
        """
        features = {}
        
        # Species features
        if self.species:
            species_features = []
            for sp in self.species:
                feat = [
                    float(sp.get('initial_concentration') or sp.get('initial_amount') or 0),
                    float(sp.get('boundary_condition', False)),
                    float(sp.get('constant', False)),
                    float(sp.get('has_only_substance_units', False))
                ]
                species_features.append(feat)
            features['species'] = np.array(species_features)
        
        # Reaction features
        if self.reactions:
            reaction_features = []
            for rxn in self.reactions:
                feat = [
                    float(rxn.get('reversible', False)),
                    float(rxn.get('fast', False)),
                    float(rxn.get('kinetic_law') is not None),
                    float(len(rxn.get('reactants', []))),
                    float(len(rxn.get('products', []))),
                    float(len(rxn.get('modifiers', [])))
                ]
                reaction_features.append(feat)
            features['reactions'] = np.array(reaction_features)
        
        # Network topology features
        if self.species and self.reactions:
            S, _, _ = self.get_stoichiometry_matrix()
            A, _ = self.get_adjacency_matrix()
            
            # Network-level features
            network_features = [
                len(self.species),  # Number of species
                len(self.reactions),  # Number of reactions
                np.count_nonzero(S),  # Number of non-zero stoichiometry entries
                np.count_nonzero(A),  # Number of edges in adjacency graph
                np.mean(np.sum(np.abs(S), axis=1)),  # Average species degree
                np.mean(np.sum(A, axis=1)),  # Average adjacency degree
            ]
            features['network'] = np.array(network_features)
        
        return features
    
    def get_ml_dataset(self) -> Dict[str, Any]:
        """
        Get comprehensive ML-ready dataset.
        
        Returns:
            dict: Complete dataset with matrices, features, and metadata
        """
        dataset = {
            'metadata': {
                'sbml_level': self.sbml_info['level'],
                'sbml_version': self.sbml_info['version'],
                'model_id': self.sbml_info['model_id'],
                'model_name': self.sbml_info['model_name'],
                'num_species': len(self.species),
                'num_reactions': len(self.reactions),
                'num_parameters': len(self.parameters),
                'has_kinetic_laws': any(r.get('kinetic_law') for r in self.reactions)
            }
        }
        
        # Add matrices
        if self.species and self.reactions:
            S, species_ids, reaction_ids = self.get_stoichiometry_matrix()
            A, _ = self.get_adjacency_matrix()
            
            dataset['matrices'] = {
                'stoichiometry': S,
                'adjacency': A,
                'species_ids': species_ids,
                'reaction_ids': reaction_ids
            }
        
        # Add feature vectors
        dataset['features'] = self.get_feature_vectors()
        
        # Add DataFrames
        dataset['dataframes'] = self.to_dataframes()
        
        return dataset
    
    def export_to_files(self, output_dir: str, format: str = 'csv') -> Dict[str, str]:
        """
        Export data to files for ML workflows.
        
        Args:
            output_dir: Directory to save files
            format: Export format ('csv', 'json', 'npz', 'pickle')
            
        Returns:
            dict: Mapping of data type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_name = self.sbml_info.get('model_id', 'sbml_model')
        exported_files = {}
        
        if format == 'csv':
            # Export DataFrames as CSV
            dataframes = self.to_dataframes()
            for name, df in dataframes.items():
                file_path = output_path / f"{model_name}_{name}.csv"
                df.to_csv(file_path, index=False)
                exported_files[name] = str(file_path)
            
            # Export matrices as CSV
            if self.species and self.reactions:
                S, species_ids, reaction_ids = self.get_stoichiometry_matrix()
                A, _ = self.get_adjacency_matrix()
                
                # Stoichiometry matrix with labels
                S_df = pd.DataFrame(S, index=species_ids, columns=reaction_ids)
                S_file = output_path / f"{model_name}_stoichiometry.csv"
                S_df.to_csv(S_file)
                exported_files['stoichiometry'] = str(S_file)
                
                # Adjacency matrix with labels
                A_df = pd.DataFrame(A, index=species_ids, columns=species_ids)
                A_file = output_path / f"{model_name}_adjacency.csv"
                A_df.to_csv(A_file)
                exported_files['adjacency'] = str(A_file)
        
        elif format == 'json':
            # Export as JSON
            ml_dataset = self.get_ml_dataset()
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                return obj
            
            json_data = {}
            for key, value in ml_dataset.items():
                if key == 'dataframes':
                    json_data[key] = {k: v.to_dict('records') for k, v in value.items()}
                elif key == 'matrices':
                    json_data[key] = {k: convert_numpy(v) for k, v in value.items()}
                elif key == 'features':
                    json_data[key] = {k: convert_numpy(v) for k, v in value.items()}
                else:
                    json_data[key] = value
            
            json_file = output_path / f"{model_name}_ml_data.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            exported_files['ml_data'] = str(json_file)
        
        elif format == 'npz':
            # Export as NumPy compressed format
            ml_dataset = self.get_ml_dataset()
            arrays_to_save = {}
            
            # Flatten all numpy arrays with descriptive names
            if 'matrices' in ml_dataset:
                arrays_to_save['stoichiometry_matrix'] = ml_dataset['matrices']['stoichiometry']
                arrays_to_save['adjacency_matrix'] = ml_dataset['matrices']['adjacency']
            
            if 'features' in ml_dataset:
                for feat_name, feat_array in ml_dataset['features'].items():
                    arrays_to_save[f'{feat_name}_features'] = feat_array
            
            npz_file = output_path / f"{model_name}_ml_data.npz"
            np.savez_compressed(npz_file, **arrays_to_save)
            exported_files['ml_data'] = str(npz_file)
        
        return exported_files
