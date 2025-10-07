from libsbml import SBMLReader, formulaToString
import logging
from ..main_parser import SBMLParsingError

logger = logging.getLogger(__name__)

class Parser:
    """
    Parser for SBML Level 3 models (versions 1 and 2).
    Enhanced parser supporting Level 3 features like conversionFactors and extensions.
    """

    def __init__(self, file_path, level=3, version=None):
        self.file_path = file_path
        self.level = level
        self.version = version
        self.model = None
        self.document = None

    def parse(self):
        """
        Parse SBML Level 3 file and extract ODE model components.

        Returns:
            dict: Parsed model data with species, reactions, parameters, compartments
        """
        try:
            reader = SBMLReader()
            self.document = reader.readSBML(self.file_path)
            self.model = self.document.getModel()

            if self.model is None:
                raise SBMLParsingError("No model found in SBML file")

            logger.info(f"Parsing SBML Level 3 Version {self.version or 'unknown'} file: {self.file_path}")

            parsed_data = {
                'sbml_info': self._get_sbml_info(),
                'species': self._parse_species(),
                'reactions': self._parse_reactions(),
                'parameters': self._parse_parameters(),
                'compartments': self._parse_compartments(),
                'rules': self._parse_rules(),
                'initial_assignments': self._parse_initial_assignments(),
                'events': self._parse_events(),
                'constraints': self._parse_constraints(),
                'unit_definitions': self._parse_unit_definitions(),
                'function_definitions': self._parse_function_definitions()
            }

            return parsed_data

        except Exception as e:
            if isinstance(e, SBMLParsingError):
                raise
            raise SBMLParsingError(f"Failed to parse Level 3 SBML file: {str(e)}")

    def _get_sbml_info(self):
        """Extract general SBML document information."""
        return {
            'level': self.document.getLevel(),
            'version': self.document.getVersion(),
            'model_id': self.model.getId(),
            'model_name': self.model.getName(),
            'substance_units': self.model.getSubstanceUnits() if self.model.isSetSubstanceUnits() else None,
            'time_units': self.model.getTimeUnits() if self.model.isSetTimeUnits() else None,
            'volume_units': self.model.getVolumeUnits() if self.model.isSetVolumeUnits() else None,
            'area_units': self.model.getAreaUnits() if self.model.isSetAreaUnits() else None,
            'length_units': self.model.getLengthUnits() if self.model.isSetLengthUnits() else None,
            'extent_units': self.model.getExtentUnits() if self.model.isSetExtentUnits() else None,
            'conversion_factor': self.model.getConversionFactor() if self.model.isSetConversionFactor() else None,
            'notes': self._get_notes(self.model),
            'num_species': self.model.getNumSpecies(),
            'num_reactions': self.model.getNumReactions(),
            'num_parameters': self.model.getNumParameters(),
            'num_compartments': self.model.getNumCompartments(),
            'num_events': self.model.getNumEvents() if hasattr(self.model, 'getNumEvents') else 0,
            'num_constraints': self.model.getNumConstraints() if hasattr(self.model, 'getNumConstraints') else 0,
            'num_functions': self.model.getNumFunctionDefinitions()
        }

    def _parse_species(self):
        """Parse species information with Level 3 enhancements."""
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
                'conversion_factor': species.getConversionFactor() if species.isSetConversionFactor() else None,
                'notes': self._get_notes(species),
                'sbo_term': species.getSBOTermID() if species.isSetSBOTerm() else None,
                'metaid': species.getMetaId() if species.isSetMetaId() else None
            }
            species_list.append(species_data)

        return species_list

    def _parse_reactions(self):
        """Parse reactions with Level 3 features."""
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
                    'constant': reactant.getConstant()
                })

            # Parse products
            products = []
            for j in range(reaction.getNumProducts()):
                product = reaction.getProduct(j)
                products.append({
                    'species': product.getSpecies(),
                    'stoichiometry': product.getStoichiometry(),
                    'constant': product.getConstant()
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
                    'parameters': self._parse_local_parameters(kl)
                }

            reaction_data = {
                'id': reaction.getId(),
                'name': reaction.getName() if reaction.isSetName() else reaction.getId(),
                'reversible': reaction.getReversible(),
                'compartment': reaction.getCompartment() if reaction.isSetCompartment() else None,
                'reactants': reactants,
                'products': products,
                'modifiers': modifiers,
                'kinetic_law': kinetic_law,
                'notes': self._get_notes(reaction),
                'sbo_term': reaction.getSBOTermID() if reaction.isSetSBOTerm() else None,
                'metaid': reaction.getMetaId() if reaction.isSetMetaId() else None
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
                'sbo_term': param.getSBOTermID() if param.isSetSBOTerm() else None,
                'metaid': param.getMetaId() if param.isSetMetaId() else None
            }
            parameters_list.append(param_data)

        return parameters_list

    def _parse_compartments(self):
        """Parse compartment information with Level 3 features."""
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
                'sbo_term': comp.getSBOTermID() if comp.isSetSBOTerm() else None,
                'metaid': comp.getMetaId() if comp.isSetMetaId() else None
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
                'sbo_term': rule.getSBOTermID() if rule.isSetSBOTerm() else None,
                'metaid': rule.getMetaId() if rule.isSetMetaId() else None
            }
            rules_list.append(rule_data)

        return rules_list

    def _parse_initial_assignments(self):
        """Parse initial assignments."""
        assignments_list = []

        for i in range(self.model.getNumInitialAssignments()):
            assignment = self.model.getInitialAssignment(i)
            assign_data = {
                'symbol': assignment.getSymbol(),
                'formula': formulaToString(assignment.getMath()) if assignment.isSetMath() else None,
                'math': formulaToString(assignment.getMath()) if assignment.isSetMath() else None,
                'notes': self._get_notes(assignment),
                'sbo_term': assignment.getSBOTermID() if assignment.isSetSBOTerm() else None,
                'metaid': assignment.getMetaId() if assignment.isSetMetaId() else None
            }
            assignments_list.append(assign_data)

        return assignments_list

    def _parse_events(self):
        """Parse events (Level 2 Version 2+, Level 3)."""
        events_list = []

        if hasattr(self.model, 'getNumEvents'):
            for i in range(self.model.getNumEvents()):
                event = self.model.getEvent(i)

                # Parse trigger
                trigger_data = None
                if event.isSetTrigger():
                    trigger = event.getTrigger()
                    trigger_data = {
                        'formula': trigger.getFormula() if trigger.isSetFormula() else None,
                        'math': formulaToString(trigger.getMath()) if trigger.isSetMath() else None,
                        'initial_value': trigger.getInitialValue() if hasattr(trigger, 'getInitialValue') else None,
                        'persistent': trigger.getPersistent() if hasattr(trigger, 'getPersistent') else None
                    }

                # Parse delay
                delay_data = None
                if event.isSetDelay():
                    delay = event.getDelay()
                    delay_data = {
                        'formula': delay.getFormula() if delay.isSetFormula() else None,
                        'math': formulaToString(delay.getMath()) if delay.isSetMath() else None
                    }

                # Parse event assignments
                assignments = []
                for j in range(event.getNumEventAssignments()):
                    ea = event.getEventAssignment(j)
                    assignments.append({
                        'variable': ea.getVariable(),
                        'formula': ea.getFormula() if ea.isSetFormula() else None,
                        'math': formulaToString(ea.getMath()) if ea.isSetMath() else None
                    })

                event_data = {
                    'id': event.getId() if event.isSetId() else None,
                    'name': event.getName() if event.isSetName() else None,
                    'use_values_from_trigger_time': event.getUseValuesFromTriggerTime() if hasattr(event, 'getUseValuesFromTriggerTime') else None,
                    'trigger': trigger_data,
                    'delay': delay_data,
                    'event_assignments': assignments,
                    'notes': self._get_notes(event),
                    'sbo_term': event.getSBOTermID() if event.isSetSBOTerm() else None
                }
                events_list.append(event_data)

        return events_list

    def _parse_constraints(self):
        """Parse constraints (Level 2 Version 2+, Level 3)."""
        constraints_list = []

        if hasattr(self.model, 'getNumConstraints'):
            for i in range(self.model.getNumConstraints()):
                constraint = self.model.getConstraint(i)
                constraint_data = {
                    'formula': constraint.getFormula() if constraint.isSetFormula() else None,
                    'math': formulaToString(constraint.getMath()) if constraint.isSetMath() else None,
                    'message': constraint.getMessageString() if constraint.isSetMessage() else None,
                    'notes': self._get_notes(constraint),
                    'sbo_term': constraint.getSBOTermID() if constraint.isSetSBOTerm() else None
                }
                constraints_list.append(constraint_data)

        return constraints_list

    def _parse_unit_definitions(self):
        """Parse unit definitions."""
        unit_defs = []

        for i in range(self.model.getNumUnitDefinitions()):
            unit_def = self.model.getUnitDefinition(i)

            units = []
            for j in range(unit_def.getNumUnits()):
                unit = unit_def.getUnit(j)
                units.append({
                    'kind': unit.getKind(),
                    'exponent': unit.getExponent(),
                    'scale': unit.getScale(),
                    'multiplier': unit.getMultiplier()
                })

            unit_def_data = {
                'id': unit_def.getId(),
                'name': unit_def.getName() if unit_def.isSetName() else unit_def.getId(),
                'units': units,
                'notes': self._get_notes(unit_def),
                'sbo_term': unit_def.getSBOTermID() if unit_def.isSetSBOTerm() else None
            }
            unit_defs.append(unit_def_data)

        return unit_defs

    def _parse_function_definitions(self):
        """Parse function definitions."""
        function_defs = []

        for i in range(self.model.getNumFunctionDefinitions()):
            func_def = self.model.getFunctionDefinition(i)
            func_data = {
                'id': func_def.getId(),
                'name': func_def.getName() if func_def.isSetName() else func_def.getId(),
                'formula': formulaToString(func_def.getMath()) if func_def.isSetMath() else None,
                'math': formulaToString(func_def.getMath()) if func_def.isSetMath() else None,
                'notes': self._get_notes(func_def),
                'sbo_term': func_def.getSBOTermID() if func_def.isSetSBOTerm() else None
            }
            function_defs.append(func_data)

        return function_defs

    def _parse_local_parameters(self, kinetic_law):
        """Parse local parameters within kinetic laws."""
        local_params = []

        for i in range(kinetic_law.getNumLocalParameters()):
            param = kinetic_law.getLocalParameter(i)
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
            notes_xml = element.getNotesString()
            return self._clean_notes_text(notes_xml)
        return None

    def _clean_notes_text(self, notes_xml):
        """Extract clean text from SBML notes XML."""
        if not notes_xml:
            return None

        try:
            import re
            from html import unescape

            # Remove XML/HTML tags
            clean_text = re.sub(r'<[^>]+>', ' ', notes_xml)

            # Decode HTML entities
            clean_text = unescape(clean_text)

            # Clean up whitespace
            clean_text = ' '.join(clean_text.split())

            # Remove common SBML boilerplate
            clean_text = re.sub(r'This model is hosted on.*?BioModels Database.*?\.', '', clean_text, flags=re.DOTALL)
            clean_text = re.sub(r'To cite BioModels Database.*?models\.', '', clean_text, flags=re.DOTALL)
            clean_text = re.sub(r'To the extent possible under law.*?Dedication.*?\.', '', clean_text, flags=re.DOTALL)

            # Clean up extra whitespace again
            clean_text = ' '.join(clean_text.split())

            return clean_text.strip() if clean_text.strip() else None

        except Exception:
            # Fallback to original if cleaning fails
            return notes_xml

    def _get_rule_type_name(self, type_code):
        """Convert rule type code to readable name."""
        type_names = {
            21: 'algebraic',   # SBML_ALGEBRAIC_RULE
            22: 'assignment',  # SBML_ASSIGNMENT_RULE
            23: 'rate'         # SBML_RATE_RULE
        }
        return type_names.get(type_code, 'unknown')
