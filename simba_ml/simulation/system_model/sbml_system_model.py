"""SBML-based system model that implements the SystemModelInterface.

This module provides a bridge between parsed SBML models and SimbaML's
system model framework, allowing SBML models to be used directly with
existing simulation and ML pipelines.
"""

import typing
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from simba_ml.simulation.system_model import system_model_interface
from simba_ml.simulation.system_model import system_model
from simba_ml.simulation import species
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module
from simba_ml.simulation import distributions
from simba_ml.simulation.sparsifier import no_sparsifier
from simba_ml.simulation import noisers
from simba_ml.sbml_parser.main_parser import MainSBMLParser
from simba_ml.sbml_parser.ml_exporter import SBMLExporter

logger = logging.getLogger(__name__)


class SBMLSystemModel(system_model.SystemModel):
    """SystemModel implementation based on parsed SBML models."""

    def __init__(
        self,
        sbml_file_path: typing.Optional[str] = None,
        name: typing.Optional[str] = None,
        sparsifier: typing.Optional[typing.Any] = None,
        noiser: typing.Optional[noisers.Noiser] = None,
        parameter_distributions: typing.Optional[dict] = None,
        species_distributions: typing.Optional[dict] = None,
        parsed_data: typing.Optional[dict] = None,
        ml_exporter: typing.Optional[typing.Any] = None
    ):
        """Initialize SBML system model.

        Args:
            sbml_file_path: Path to SBML model file (required if parsed_data is None)
            name: Model name (defaults to SBML model name)
            sparsifier: Sparsifier for output processing
            noiser: Noiser for output processing
            parameter_distributions: Custom parameter distributions
            species_distributions: Custom species initial value distributions
            parsed_data: Parsed SBML data
            ml_exporter: SBMLExporter class
        """
        # Use already-parsed data if provided, otherwise parse file
        if parsed_data is not None and ml_exporter is not None:
            self.sbml_data = parsed_data
            self.ml_exporter = ml_exporter
            self.sbml_file_path = parsed_data.get('metadata', {}).get('sbml_file_path', 'parsed_data')
        else:
            # Legacy workflow: parse from file
            if sbml_file_path is None:
                raise ValueError("Either sbml_file_path or (parsed_data + ml_exporter) must be provided")
            self.sbml_file_path = sbml_file_path
            self.parser = MainSBMLParser(sbml_file_path)
            self.sbml_data = self.parser.process()
            self.ml_exporter = SBMLExporter(self.sbml_data)

        # Set model name
        fallback_name = Path(self.sbml_file_path).stem if self.sbml_file_path else 'SBML_Model'
        model_name = name or self.sbml_data['sbml_info'].get('model_name', fallback_name)

        # Build species and parameters from parsed data
        built_species = self._build_species(species_distributions)
        built_parameters = self._build_kinetic_parameters(parameter_distributions)
        built_deriv = self._build_derivative_function()

        # Initialize parent SystemModel with required parameters
        super().__init__(
            name=model_name,
            specieses=list(built_species.values()),
            kinetic_parameters=built_parameters,
            deriv=built_deriv,
            sparsifier=sparsifier or no_sparsifier.NoSparsifier(),
            noiser=noiser or noisers.NoNoiser()
        )

        # Keep SBML-specific references for internal use
        self._specieses = built_species
        self._kinetic_parameters = built_parameters
        self._deriv = built_deriv

        logger.info(f"Created SBML system model '{model_name}' with {len(built_species)} species")

    def _build_species(self, custom_distributions: typing.Optional[dict] = None) -> dict[str, species.Species]:
        """Build species from SBML data."""
        species_dict = {}
        custom_distributions = custom_distributions or {}

        for sp_data in self.sbml_data['species']:
            sp_id = sp_data['id']
            sp_name = sp_data.get('name', sp_id)

            # Get initial value
            initial_value = 1.0
            if sp_data.get('initial_concentration') is not None:
                initial_value = sp_data['initial_concentration']
            elif sp_data.get('initial_amount') is not None:
                initial_value = sp_data['initial_amount']

            # Create distribution for initial values
            if sp_id in custom_distributions:
                initial_distribution = custom_distributions[sp_id]
            else:
                # Default: lognormal around initial value
                if initial_value > 0:
                    initial_distribution = distributions.LogNormalDistribution(
                        mu=np.log(initial_value),
                        sigma=0.3  # 30% variability
                    )
                else:
                    initial_distribution = distributions.Constant(1e-6)

            # Determine if species should be in output
            is_boundary = sp_data.get('boundary_condition', False)
            is_constant = sp_data.get('constant', False)
            contained_in_output = not (is_boundary or is_constant)

            species_obj = species.Species(
                name=sp_id,
                distribution=initial_distribution,
                contained_in_output=contained_in_output
            )

            species_dict[sp_id] = species_obj

        return species_dict

    def _build_kinetic_parameters(
        self,
        custom_distributions: typing.Optional[dict] = None
    ) -> dict[str, kinetic_parameters_module.KineticParameter]:
        """Build kinetic parameters from SBML data."""
        params_dict = {}
        custom_distributions = custom_distributions or {}

        # Global parameters
        for param_data in self.sbml_data['parameters']:
            param_id = param_data['id']
            param_value = param_data.get('value', 1.0)

            if param_id in custom_distributions:
                distribution = custom_distributions[param_id]
            else:
                # Default: lognormal around parameter value
                if param_value > 0:
                    distribution = distributions.LogNormalDistribution(
                        mu=np.log(param_value),
                        sigma=0.2  # 20% variability
                    )
                else:
                    distribution = distributions.Constant(param_value)

            params_dict[param_id] = kinetic_parameters_module.ConstantKineticParameter(
                distribution=distribution
            )

        # Local parameters from kinetic laws
        for reaction in self.sbml_data['reactions']:
            if reaction.get('kinetic_law') and reaction['kinetic_law'].get('parameters'):
                for local_param in reaction['kinetic_law']['parameters']:
                    param_id = f"{reaction['id']}_{local_param['id']}"
                    param_value = local_param.get('value', 1.0)

                    if param_id in custom_distributions:
                        distribution = custom_distributions[param_id]
                    else:
                        if param_value > 0:
                            distribution = distributions.LogNormalDistribution(
                                mu=np.log(param_value),
                                sigma=0.2
                            )
                        else:
                            distribution = distributions.Constant(param_value)

                    params_dict[param_id] = kinetic_parameters_module.ConstantKineticParameter(
                        distribution=distribution
                    )

        return params_dict

    def _build_derivative_function(self) -> typing.Callable:
        """Build derivative function from SBML reaction network."""

        # Get dynamic species (non-boundary, non-constant)
        dynamic_species = []
        for sp_data in self.sbml_data['species']:
            is_boundary = sp_data.get('boundary_condition', False)
            is_constant = sp_data.get('constant', False)
            if not (is_boundary or is_constant):
                dynamic_species.append(sp_data['id'])

        species_to_index = {sp_id: i for i, sp_id in enumerate(dynamic_species)}

        def derivative_func(t: float, y: list[float], kinetic_params: dict) -> tuple[float, ...]:
            """Compute derivatives for dynamic species."""

            # Create species concentration dictionary
            species_conc = {}
            for i, sp_id in enumerate(dynamic_species):
                species_conc[sp_id] = max(y[i], 0.0)  # Ensure non-negative

            # Add boundary species (from initial values)
            for sp_data in self.sbml_data['species']:
                if sp_data.get('boundary_condition', False):
                    sp_id = sp_data['id']
                    initial_value = 1.0
                    if sp_data.get('initial_concentration') is not None:
                        initial_value = sp_data['initial_concentration']
                    elif sp_data.get('initial_amount') is not None:
                        initial_value = sp_data['initial_amount']
                    species_conc[sp_id] = initial_value

            # Compute derivatives
            dydt = [0.0] * len(dynamic_species)

            for reaction in self.sbml_data['reactions']:
                if not reaction.get('kinetic_law'):
                    continue

                # Get rate formula
                kinetic_law = reaction['kinetic_law']
                rate_formula = None
                if kinetic_law.get('math'):
                    rate_formula = kinetic_law['math']
                elif kinetic_law.get('formula'):
                    rate_formula = kinetic_law['formula']

                if not rate_formula:
                    continue

                # Evaluate reaction rate
                try:
                    # Prepare local parameters
                    local_params = {}
                    if kinetic_law.get('parameters'):
                        for local_param in kinetic_law['parameters']:
                            local_key = f"{reaction['id']}_{local_param['id']}"
                            if local_key in kinetic_params:
                                local_params[local_param['id']] = kinetic_params[local_key]
                            else:
                                local_params[local_param['id']] = local_param.get('value', 1.0)

                    rate = self._evaluate_rate_formula(
                        rate_formula,
                        species_conc,
                        kinetic_params,
                        local_params
                    )

                    # Apply stoichiometry to dynamic species
                    for reactant in reaction.get('reactants', []):
                        sp_id = reactant['species']
                        if sp_id in species_to_index:
                            stoich = reactant.get('stoichiometry', 1.0)
                            dydt[species_to_index[sp_id]] -= stoich * rate

                    for product in reaction.get('products', []):
                        sp_id = product['species']
                        if sp_id in species_to_index:
                            stoich = product.get('stoichiometry', 1.0)
                            dydt[species_to_index[sp_id]] += stoich * rate

                except Exception as e:
                    logger.debug(f"Error evaluating rate for reaction {reaction['id']}: {e}")
                    continue

            # Process rate rules (direct ODE specification)
            for rule in self.sbml_data.get('rules', []):
                if rule.get('type') != 'rate':
                    continue

                variable = rule.get('variable')
                if not variable or variable not in species_to_index:
                    continue

                rate_formula = rule.get('formula') or rule.get('math')
                if not rate_formula:
                    continue

                try:
                    rate = self._evaluate_rate_formula(
                        rate_formula,
                        species_conc,
                        kinetic_params,
                        {}  # Rate rules don't have local parameters
                    )
                    # Rate rules directly specify dx/dt for the variable
                    dydt[species_to_index[variable]] = rate
                except Exception as e:
                    logger.debug(f"Error evaluating rate rule for {variable}: {e}")
                    continue

            return tuple(dydt)

        return derivative_func

    def _evaluate_rate_formula(
        self,
        formula: str,
        species_conc: dict,
        kinetic_params: dict,
        local_params: dict
    ) -> float:
        """Evaluate a rate formula with given concentrations and parameters."""

        # Create evaluation context
        eval_context = {}
        eval_context.update(species_conc)
        eval_context.update(kinetic_params)
        eval_context.update(local_params)

        # Add global parameters from SBML
        for param_data in self.sbml_data['parameters']:
            param_id = param_data['id']
            if param_id not in eval_context:
                eval_context[param_id] = param_data.get('value', 1.0)

        try:
            result = self._safe_evaluate_formula(formula, eval_context)
            return float(result) if result is not None else 0.0
        except Exception as e:
            logger.debug(f"Error evaluating formula '{formula}': {e}")
            return 0.0

    def _safe_evaluate_formula(self, formula: str, context: dict) -> float:
        try:
            import numexpr as ne
            # numexpr requires all variables to be arrays, so convert to scalars
            for var, val in context.items():
                if isinstance(val, (int, float)):
                    context[var] = float(val)
            return ne.evaluate(formula, local_dict=context)
        except ImportError:
            # Fallback: Use restricted eval with only math operations allowed
            import math
            import ast
            import operator

            # Define allowed operations and functions
            allowed_ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            allowed_funcs = {
                'exp': math.exp,
                'log': math.log,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'pow': pow,
                'abs': abs,
            }

            allowed_constants = {
                'pi': math.pi,
                'e': math.e,
            }

            def safe_eval_node(node):
                if isinstance(node, ast.Num):  # number
                    return node.n
                elif isinstance(node, ast.Constant):  # Python 3.8+
                    return node.value
                elif isinstance(node, ast.Name):  # variable
                    var_name = node.id
                    if var_name in context:
                        return context[var_name]
                    elif var_name in allowed_constants:
                        return allowed_constants[var_name]
                    else:
                        raise ValueError(f"Unknown variable: {var_name}")
                elif isinstance(node, ast.BinOp):  # binary operation
                    left = safe_eval_node(node.left)
                    right = safe_eval_node(node.right)
                    op = allowed_ops.get(type(node.op))
                    if op:
                        return op(left, right)
                    else:
                        raise ValueError(f"Unsupported operation: {type(node.op)}")
                elif isinstance(node, ast.UnaryOp):  # unary operation
                    operand = safe_eval_node(node.operand)
                    op = allowed_ops.get(type(node.op))
                    if op:
                        return op(operand)
                    else:
                        raise ValueError(f"Unsupported unary operation: {type(node.op)}")
                elif isinstance(node, ast.Call):  # function call
                    func_name = node.func.id
                    if func_name in allowed_funcs:
                        args = [safe_eval_node(arg) for arg in node.args]
                        return allowed_funcs[func_name](*args)
                    else:
                        raise ValueError(f"Unsupported function: {func_name}")
                else:
                    raise ValueError(f"Unsupported node type: {type(node)}")

            try:
                tree = ast.parse(formula, mode='eval')
                return safe_eval_node(tree.body)
            except (SyntaxError, ValueError) as e:
                logger.debug(f"Safe evaluation failed for '{formula}': {e}")
                # Last resort: return 1.0 as default rate
                return 1.0


    def apply_sparsifier(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Apply sparsification to the signal."""
        # Filter to output species only
        output_species = [
            sp_id for sp_id, sp_obj in self._specieses.items()
            if sp_obj.contained_in_output
        ]

        filtered_signal = signal[output_species] if output_species else signal
        return self.sparsifier.sparsify(filtered_signal)

    def get_clean_signal(
        self,
        start_values: dict[str, typing.Any],
        sample_id: int,
        deriv_noised: bool = True
    ) -> pd.DataFrame:
        """Generate a clean signal using time-series simulation.

        Note: This is mainly for compatibility. For steady-state generation,
        use the numerical solvers directly.
        """
        from scipy import integrate

        # Get time points
        timestamps = start_values.get("timestamps", [1000])[sample_id]
        t = np.linspace(0, timestamps, int(timestamps) + 1)

        # Get initial conditions for dynamic species
        dynamic_species = [
            sp_id for sp_id, sp_obj in self._specieses.items()
            if sp_obj.contained_in_output or not any([
                sp_data.get('boundary_condition', False) or sp_data.get('constant', False)
                for sp_data in self.sbml_data['species'] if sp_data['id'] == sp_id
            ])
        ]

        y0 = [
            start_values["specieses"][sp_id][sample_id]
            for sp_id in dynamic_species
        ]

        # Get kinetic parameters
        kinetic_params = {
            name: param.get_at_timestamp(sample_id, 0.0)
            for name, param in self._kinetic_parameters.items()
        }

        # Solve ODE
        try:
            result = integrate.solve_ivp(
                fun=lambda t, y: self._deriv(t, y.tolist(), kinetic_params),
                y0=y0,
                t_span=(t[0], t[-1]),
                t_eval=t,
                method='LSODA',
                atol=1e-6,
                rtol=1e-3
            )

            # Convert to DataFrame with dynamic species
            signal_df = pd.DataFrame(
                result.y.T,
                columns=dynamic_species
            )

            # Add boundary species (they remain constant)
            for sp_data in self.sbml_data['species']:
                if sp_data.get('boundary_condition', False):
                    sp_id = sp_data['id']
                    # Get boundary species initial value
                    if sp_id in start_values["specieses"]:
                        boundary_value = start_values["specieses"][sp_id][sample_id]
                    else:
                        # Use SBML initial value
                        boundary_value = sp_data.get('initial_concentration',
                                                   sp_data.get('initial_amount', 1.0))
                        if boundary_value is None:
                            boundary_value = 1.0

                    # Add constant column for boundary species
                    signal_df[sp_id] = boundary_value

            return signal_df

        except Exception as e:
            logger.error(f"Error in get_clean_signal: {e}")
            # Return default DataFrame with dynamic species
            signal_df = pd.DataFrame({sp_id: [y0[i]] for i, sp_id in enumerate(dynamic_species)})

            # Add boundary species to default DataFrame too
            for sp_data in self.sbml_data['species']:
                if sp_data.get('boundary_condition', False):
                    sp_id = sp_data['id']
                    if sp_id in start_values["specieses"]:
                        boundary_value = start_values["specieses"][sp_id][sample_id]
                    else:
                        boundary_value = sp_data.get('initial_concentration',
                                                   sp_data.get('initial_amount', 1.0))
                        if boundary_value is None:
                            boundary_value = 1.0
                    signal_df[sp_id] = boundary_value

            return signal_df
