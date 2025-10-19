"""Pattern-based configuration adapter for steady-state datasets.

This module provides automatic or manual feature detection:
- AUTOMATIC: Based on column naming patterns
  - INPUTS: kinetic_parameter_* columns + *_start_value columns
  - OUTPUTS: Species columns (names that have corresponding *_start_value)
- MANUAL: User specifies input/output column names explicitly
"""

import pandas as pd
from typing import List, Optional

from simba_ml.prediction.steady_state.config.steady_state_data_config import DataConfig


def create_config_from_csv(
    csv_file: str,
    method: str = "automatic",
    input_features: Optional[List[str]] = None,
    output_features: Optional[List[str]] = None,
    max_input_features: Optional[int] = None,  # None = use all available
    max_output_features: Optional[int] = None,  # None = use all available
    mixing_ratios: Optional[List[float]] = None,
    test_split: float = 0.2,
    k_cross_validation: int = 5,
    verbose: bool = False
) -> DataConfig:
    """
    Create config with automatic or manual feature selection.

    Args:
        csv_file: Path to steady-state CSV file
        method: 'automatic' (pattern-based) or 'manual' (user-specified)
        input_features: Manual input column names (only for method='manual')
        output_features: Manual output column names (only for method='manual')
        max_input_features: Limit number of input features (None = use all available)
        max_output_features: Limit number of output features (None = use all available)
        mixing_ratios: Ratios for synthetic/observed data mixing
        test_split: Test set proportion (default 0.2)
        k_cross_validation: Number of CV folds (default 5)
        verbose: Print detailed analysis summary

    Returns:
        DataConfig with feature selection

    Raises:
        ValueError: If method is invalid or required parameters are missing

    Examples:
        >>> # Automatic detection (uses all features)
        >>> config = create_config_from_csv('data.csv', method='automatic')
        >>>
        >>> # Automatic with feature limits
        >>> config = create_config_from_csv('data.csv', method='automatic',
        ...     max_input_features=10, max_output_features=5)
        >>>
        >>> # Manual specification
        >>> config = create_config_from_csv('data.csv', method='manual',
        ...     input_features=['kinetic_parameter_k1', 'M_start_value'],
        ...     output_features=['M', 'MpY'])
    """
    if method == "automatic":
        return _create_automatic_config(
            csv_file, max_input_features, max_output_features,
            mixing_ratios, test_split, k_cross_validation, verbose
        )
    elif method == "manual":
        return _create_manual_config(
            csv_file, input_features, output_features,
            mixing_ratios, test_split, k_cross_validation, verbose
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'automatic' or 'manual'")


def _create_automatic_config(
    csv_file: str,
    max_input_features: Optional[int],
    max_output_features: Optional[int],
    mixing_ratios: Optional[List[float]],
    test_split: float,
    k_cross_validation: int,
    verbose: bool
) -> DataConfig:
    """Create config automatically using column naming patterns."""
    df = pd.read_csv(csv_file)

    # Extract features using column name patterns
    columns = list(df.columns)

    # Find kinetic parameter columns
    param_columns = [col for col in columns if col.startswith('kinetic_parameter_')]

    # Find start value columns
    start_value_columns = [col for col in columns if col.endswith('_start_value')]

    # Find species columns (those that have corresponding _start_value)
    species_columns = []
    for col in start_value_columns:
        # Remove '_start_value' suffix to get species name
        species_name = col.replace('_start_value', '')
        if species_name in columns:
            species_columns.append(species_name)

    # Build input/output lists
    all_inputs = param_columns + start_value_columns
    all_outputs = species_columns

    # Apply feature limits if specified
    if max_input_features is not None and len(all_inputs) > max_input_features:
        # Prioritize parameters, then ICs
        limited_inputs = param_columns[:max_input_features//2] if param_columns else []
        remaining_slots = max_input_features - len(limited_inputs)
        if remaining_slots > 0:
            limited_inputs.extend(start_value_columns[:remaining_slots])
        all_inputs = limited_inputs

    if max_output_features is not None and len(all_outputs) > max_output_features:
        all_outputs = all_outputs[:max_output_features]

    if verbose:
        print(f"Pattern-based config: {len(all_inputs)} inputs → {len(all_outputs)} outputs")

    if not all_inputs:
        raise ValueError(f"No input features found. Expected 'kinetic_parameter_*' or '*_start_value' columns.")
    if not all_outputs:
        raise ValueError(f"No output features found. Expected species columns with corresponding '_start_value'.")

    return DataConfig(
        start_value_params=all_inputs,
        prediction_params=all_outputs,
        mixing_ratios=mixing_ratios or [0.0, 0.5, 1.0],
        test_split=test_split,
        k_cross_validation=k_cross_validation
    )


def _create_manual_config(
    csv_file: str,
    input_features: Optional[List[str]],
    output_features: Optional[List[str]],
    mixing_ratios: Optional[List[float]],
    test_split: float,
    k_cross_validation: int,
    verbose: bool
) -> DataConfig:
    """Create config using manually specified features."""
    if not input_features or not output_features:
        raise ValueError("Manual method requires both input_features and output_features")

    # Validate that features exist in CSV
    df = pd.read_csv(csv_file)
    available_columns = set(df.columns)

    missing_inputs = set(input_features) - available_columns
    missing_outputs = set(output_features) - available_columns

    if missing_inputs:
        raise ValueError(f"Input features not found in CSV: {missing_inputs}")
    if missing_outputs:
        raise ValueError(f"Output features not found in CSV: {missing_outputs}")

    if verbose:
        print(f"Manual config: {len(input_features)} inputs → {len(output_features)} outputs")

    return DataConfig(
        start_value_params=input_features,
        prediction_params=output_features,
        mixing_ratios=mixing_ratios or [0.0, 0.5, 1.0],
        test_split=test_split,
        k_cross_validation=k_cross_validation
    )


def analyze_csv(csv_file: str):
    """Print detailed analysis of CSV structure.

    Args:
        csv_file: Path to CSV file
    """
    df = pd.read_csv(csv_file)
    columns = list(df.columns)

    param_columns = [col for col in columns if col.startswith('kinetic_parameter_')]
    start_value_columns = [col for col in columns if col.endswith('_start_value')]
    species_columns = [col.replace('_start_value', '') for col in start_value_columns
                      if col.replace('_start_value', '') in columns]
    other_columns = [col for col in columns
                    if not (col.startswith('kinetic_parameter_') or
                           col.endswith('_start_value') or
                           col in species_columns)]

    print("📊 CSV Structure Analysis")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn breakdown:")
    print(f"  • Kinetic parameters: {len(param_columns)}")
    print(f"  • Initial conditions: {len(start_value_columns)}")
    print(f"  • Species (steady-state): {len(species_columns)}")
    if other_columns:
        print(f"  • Other columns: {len(other_columns)} ({', '.join(other_columns)})")

    print(f"\n📝 Sample columns:")
    if param_columns:
        print(f"  Parameters: {param_columns[:3]}{'...' if len(param_columns) > 3 else ''}")
    if start_value_columns:
        print(f"  Initial conditions: {start_value_columns[:3]}{'...' if len(start_value_columns) > 3 else ''}")
    if species_columns:
        print(f"  Species: {species_columns[:3]}{'...' if len(species_columns) > 3 else ''}")

    print(f"\n✅ Recommended config:")
    print(f"  INPUTS ({len(param_columns + start_value_columns)} total):")
    print(f"    - {len(param_columns)} kinetic parameters")
    print(f"    - {len(start_value_columns)} initial concentrations")
    print(f"  OUTPUTS ({len(species_columns)} total):")
    print(f"    - {len(species_columns)} steady-state concentrations")
