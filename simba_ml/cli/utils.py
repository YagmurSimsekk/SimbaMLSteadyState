"""Shared CLI utilities for SimbaML command-line interface."""


def is_ode_ready(sbml_data):
    """Check if SBML model is ready for ODE simulation.

    Args:
        sbml_data: Parsed SBML model data dictionary containing reactions and rules.

    Returns:
        bool: True if model has kinetic laws or rate rules for ODE simulation.
    """
    # Check for reactions with kinetic laws
    has_kinetic_laws = any(r.get('kinetic_law') for r in sbml_data.get('reactions', []))

    # Check for rate rules (direct ODE specification)
    has_rate_rules = False
    rules = sbml_data.get('rules', [])
    if rules:
        # Rate rules have a 'type' field with value 'rate'
        has_rate_rules = any(rule.get('type') == 'rate' for rule in rules)

    return has_kinetic_laws or has_rate_rules
