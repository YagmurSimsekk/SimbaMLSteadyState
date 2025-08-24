from libsbml import SBMLReader, formulaToString
import logging
from ..main_parser import SBMLParsingError

logger = logging.getLogger(__name__)

class Parser:
    """
    Parser for SBML Level 2 models (versions 4 and 5).
    Focuses on ODE model extraction and conversion.
    """
    
    def __init__(self, file_path, level=2, version=None):
        self.file_path = file_path
        self.level = level
        self.version = version
        self.model = None
        self.document = None

    def parse(self):
        """
        Parse SBML Level 2 file and extract ODE model components.
        
        Returns:
            dict: Parsed model data with species, reactions, parameters, compartments
        """
        try:
            reader = SBMLReader()
            self.document = reader.readSBML(self.file_path)
            self.model = self.document.getModel()
            
            if self.model is None:
                raise SBMLParsingError("No model found in SBML file")
                
            logger.info(f"Parsing SBML Level 2 Version {self.version or 'unknown'} file: {self.file_path}")
            
            parsed_data = {
                'sbml_info': self._get_sbml_info(),
                'species': self._parse_species(),
                'reactions': self._parse_reactions(),
                'parameters': self._parse_parameters(),
                'compartments': self._parse_compartments(),
                'rules': self._parse_rules(),
                'initial_assignments': self._parse_initial_assignments()
            }
            
            return parsed_data
            
        except Exception as e:
            if isinstance(e, SBMLParsingError):
                raise
            raise SBMLParsingError(f"Failed to parse Level 2 SBML file: {str(e)}")

    def _get_sbml_info(self):
        """Extract general SBML document information."""
        return {
            'level': self.document.getLevel(),
            'version': self.document.getVersion(),
            'model_id': self.model.getId(),
            'model_name': self.model.getName(),
            'notes': self._get_notes(self.model),
            'num_species': self.model.getNumSpecies(),
            'num_reactions': self.model.getNumReactions(),
            'num_parameters': self.model.getNumParameters(),
            'num_compartments': self.model.getNumCompartments()
        }

    def _parse_species(self):
        """Parse species information for ODE variables."""
        species_list = []
        
        for i in range(self.model.getNumSpecies()):
            species = self.model.getSpecies(i)
            species_data = {
                'id': species.getId(),
                'name': species.getName() if species.isSetName() else species.getId(),
                'compartment': species.getCompartment(),
                'initial_amount': species.getInitialAmount() if species.isSetInitialAmount() else None,
                'initial_concentration': species.getInitialConcentration() if species.isSetInitialConcentration() else None,
                'substance_units': species.getSubstanceUnits() if species.isSetSubstanceUnits() else None,
                'has_only_substance_units': species.getHasOnlySubstanceUnits(),
                'boundary_condition': species.getBoundaryCondition(),
                'constant': species.getConstant(),
                'notes': self._get_notes(species),
                'sbo_term': species.getSBOTermID() if species.isSetSBOTerm() else None
            }
            species_list.append(species_data)
            
        return species_list

    def _parse_reactions(self):
        """Parse reactions and kinetic laws for ODE system."""
        reactions_list = []
        
        for i in range(self.model.getNumReactions()):
            reaction = self.model.getReaction(i)
            
            # Parse reactants
            reactants = []
            for j in range(reaction.getNumReactants()):
                reactant = reaction.getReactant(j)
                reactants.append({
                    'species': reactant.getSpecies(),
                    'stoichiometry': reactant.getStoichiometry(),
                    'constant': reactant.getConstant() if hasattr(reactant, 'getConstant') else True
                })
            
            # Parse products  
            products = []
            for j in range(reaction.getNumProducts()):
                product = reaction.getProduct(j)
                products.append({
                    'species': product.getSpecies(),
                    'stoichiometry': product.getStoichiometry(),
                    'constant': product.getConstant() if hasattr(product, 'getConstant') else True
                })
            
            # Parse modifiers
            modifiers = []
            for j in range(reaction.getNumModifiers()):
                modifier = reaction.getModifier(j)
                modifiers.append({
                    'species': modifier.getSpecies()
                })
            
            # Parse kinetic law
            kinetic_law = None
            if reaction.isSetKineticLaw():
                kl = reaction.getKineticLaw()
                kinetic_law = {
                    'formula': kl.getFormula() if kl.isSetFormula() else None,
                    'math': formulaToString(kl.getMath()) if kl.isSetMath() else None,
                    'parameters': self._parse_local_parameters(kl),
                    'substance_units': kl.getSubstanceUnits() if kl.isSetSubstanceUnits() else None,
                    'time_units': kl.getTimeUnits() if kl.isSetTimeUnits() else None
                }
            
            reaction_data = {
                'id': reaction.getId(),
                'name': reaction.getName() if reaction.isSetName() else reaction.getId(),
                'reversible': reaction.getReversible(),
                'fast': reaction.getFast() if hasattr(reaction, 'getFast') else False,
                'reactants': reactants,
                'products': products,
                'modifiers': modifiers,
                'kinetic_law': kinetic_law,
                'notes': self._get_notes(reaction),
                'sbo_term': reaction.getSBOTermID() if reaction.isSetSBOTerm() else None
            }
            reactions_list.append(reaction_data)
            
        return reactions_list

    def _parse_parameters(self):
        """Parse global parameters."""
        parameters_list = []
        
        for i in range(self.model.getNumParameters()):
            param = self.model.getParameter(i)
            param_data = {
                'id': param.getId(),
                'name': param.getName() if param.isSetName() else param.getId(),
                'value': param.getValue() if param.isSetValue() else None,
                'units': param.getUnits() if param.isSetUnits() else None,
                'constant': param.getConstant(),
                'notes': self._get_notes(param),
                'sbo_term': param.getSBOTermID() if param.isSetSBOTerm() else None
            }
            parameters_list.append(param_data)
            
        return parameters_list

    def _parse_compartments(self):
        """Parse compartment information."""
        compartments_list = []
        
        for i in range(self.model.getNumCompartments()):
            comp = self.model.getCompartment(i)
            comp_data = {
                'id': comp.getId(),
                'name': comp.getName() if comp.isSetName() else comp.getId(),
                'spatial_dimensions': comp.getSpatialDimensions(),
                'size': comp.getSize() if comp.isSetSize() else None,
                'units': comp.getUnits() if comp.isSetUnits() else None,
                'constant': comp.getConstant(),
                'notes': self._get_notes(comp),
                'sbo_term': comp.getSBOTermID() if comp.isSetSBOTerm() else None
            }
            compartments_list.append(comp_data)
            
        return compartments_list

    def _parse_rules(self):
        """Parse assignment, rate, and algebraic rules."""
        rules_list = []
        
        for i in range(self.model.getNumRules()):
            rule = self.model.getRule(i)
            rule_type = rule.getTypeCode()
            
            rule_data = {
                'type': self._get_rule_type_name(rule_type),
                'variable': rule.getVariable() if hasattr(rule, 'getVariable') else None,
                'formula': rule.getFormula() if rule.isSetFormula() else None,
                'math': formulaToString(rule.getMath()) if rule.isSetMath() else None,
                'notes': self._get_notes(rule),
                'sbo_term': rule.getSBOTermID() if rule.isSetSBOTerm() else None
            }
            rules_list.append(rule_data)
            
        return rules_list

    def _parse_initial_assignments(self):
        """Parse initial assignments (Level 2 Version 2+)."""
        assignments_list = []
        
        if hasattr(self.model, 'getNumInitialAssignments'):
            for i in range(self.model.getNumInitialAssignments()):
                assignment = self.model.getInitialAssignment(i)
                assign_data = {
                    'symbol': assignment.getSymbol(),
                    'formula': assignment.getFormula() if assignment.isSetFormula() else None,
                    'math': formulaToString(assignment.getMath()) if assignment.isSetMath() else None,
                    'notes': self._get_notes(assignment),
                    'sbo_term': assignment.getSBOTermID() if assignment.isSetSBOTerm() else None
                }
                assignments_list.append(assign_data)
                
        return assignments_list

    def _parse_local_parameters(self, kinetic_law):
        """Parse local parameters within kinetic laws."""
        local_params = []
        
        for i in range(kinetic_law.getNumParameters()):
            param = kinetic_law.getParameter(i)
            param_data = {
                'id': param.getId(),
                'name': param.getName() if param.isSetName() else param.getId(),
                'value': param.getValue() if param.isSetValue() else None,
                'units': param.getUnits() if param.isSetUnits() else None,
                'notes': self._get_notes(param),
                'sbo_term': param.getSBOTermID() if param.isSetSBOTerm() else None
            }
            local_params.append(param_data)
            
        return local_params

    def _get_notes(self, element):
        """Extract notes/annotations from SBML element."""
        if element.isSetNotes():
            return element.getNotesString()
        return None

    def _get_rule_type_name(self, type_code):
        """Convert rule type code to readable name."""
        type_names = {
            1: 'assignment',  # SBML_ASSIGNMENT_RULE
            2: 'rate',        # SBML_RATE_RULE  
            3: 'algebraic'    # SBML_ALGEBRAIC_RULE
        }
        return type_names.get(type_code, 'unknown')
