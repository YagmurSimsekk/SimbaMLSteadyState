#!/usr/bin/env python3
"""Adaptive steady-state generator with better diagnostics and efficiency."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from simba_ml.simulation.system_model import SBMLSystemModel
from simba_ml.simulation.generators import SteadyStateGenerator
from simba_ml.sbml_parser.main_parser import MainSBMLParser

logger = logging.getLogger(__name__)


class ImprovedAdaptiveGenerator:
    """Adaptive generator with hierarchical variation and diagnostics."""

    def __init__(
        self,
        sbml_file_path: str,
        min_convergence_rate: float = 0.7,
        global_param_variation: Optional[float] = None,  # Auto-determined if None
        local_param_variation: float = 0.0,  # Keep local params constant by default
        species_variation: float = 0.3,  # IC variation (30% by default)
        verbose: bool = True
    ):
        """Initialize improved adaptive generator.

        Args:
            sbml_file_path: Path to SBML file
            min_convergence_rate: Target minimum convergence rate (0-1)
            global_param_variation: Fixed sigma for global params, or None for auto
            local_param_variation: Sigma for local params (default 0 = constant)
            species_variation: Sigma for initial concentrations (default 0.3 = 30%)
            verbose: Print progress messages
        """
        self.sbml_file_path = sbml_file_path
        self.min_convergence_rate = min_convergence_rate
        self.global_param_variation = global_param_variation
        self.local_param_variation = local_param_variation
        self.species_variation = species_variation
        self.verbose = verbose

        self.optimal_sigma = None
        self.model = None

        # Cache parsed SBML data to avoid re-parsing
        self._sbml_data = None

    def _create_model_with_variation(
        self,
        global_sigma: float,
        local_sigma: float = 0.0
    ) -> SBMLSystemModel:
        """Create model with specified parameter variations.

        Args:
            global_sigma: Variation for global parameters
            local_sigma: Variation for local parameters

        Returns:
            SBMLSystemModel with configured distributions
        """
        # Parse SBML once and cache (avoids creating temp models)
        if self._sbml_data is None:
            parser = MainSBMLParser(self.sbml_file_path)
            self._sbml_data = parser.process()

        # Use centralized helpers to build distributions
        param_distributions = SBMLSystemModel.create_lognormal_distributions(
            sbml_data=self._sbml_data,
            global_sigma=global_sigma,
            local_sigma=local_sigma
        )

        species_distributions = SBMLSystemModel.create_species_distributions(
            sbml_data=self._sbml_data,
            species_sigma=self.species_variation
        )

        return SBMLSystemModel(
            sbml_file_path=self.sbml_file_path,
            parameter_distributions=param_distributions,
            species_distributions=species_distributions
        )

    def _test_convergence_efficiently(
        self,
        model: SBMLSystemModel,
        n_test: int = 10
    ) -> Tuple[float, int, int]:
        """Efficient convergence test using SteadyStateGenerator.

        Args:
            model: Model to test
            n_test: Number of samples to test

        Returns:
            (convergence_rate, n_success, n_failed)
        """
        generator = SteadyStateGenerator(sm=model)

        try:
            # SteadyStateGenerator already has LSODA → bounded fallback
            dataset = generator.generate_signals(n=n_test)
            n_success = len(dataset)
            n_failed = n_test - n_success
            conv_rate = n_success / n_test if n_test > 0 else 0
            return conv_rate, n_success, n_failed
        except Exception as e:
            logger.debug(f"Convergence test failed: {e}")
            return 0.0, 0, n_test

    def find_optimal_variation(self) -> float:
        """Find optimal global parameter variation.

        Returns:
            Optimal sigma value
        """
        if self.global_param_variation is not None:
            # User specified - use it
            self.optimal_sigma = self.global_param_variation
            self.model = self._create_model_with_variation(
                self.optimal_sigma,
                self.local_param_variation
            )
            return self.optimal_sigma

        # Auto-find optimal
        model_id = Path(self.sbml_file_path).stem

        if self.verbose:
            print(f"Finding optimal parameter variation for {model_id}...")
            print(f"Target: >{self.min_convergence_rate*100:.0f}% convergence")
            print(f"Local params: {'constant' if self.local_param_variation == 0 else f'σ={self.local_param_variation:.3f}'}")

        # Hierarchical search with finer granularity
        sigma_schedule = [
            0.200, 0.150, 0.100, 0.075, 0.050,
            0.035, 0.025, 0.015, 0.010, 0.005, 0.000
        ]

        for sigma in sigma_schedule:
            if self.verbose:
                print(f"\n  Testing global σ={sigma:.3f} ({sigma*100:5.1f}%)...", end=' ')

            model = self._create_model_with_variation(sigma, self.local_param_variation)
            conv_rate, n_success, n_failed = self._test_convergence_efficiently(model, n_test=10)

            if self.verbose:
                status = "✅" if conv_rate >= self.min_convergence_rate else f"{n_success}/10"
                print(f"{status}")

            if conv_rate >= self.min_convergence_rate:
                if self.verbose:
                    print(f"\n  ✅ Acceptable variation found: global σ={sigma:.3f}")
                self.optimal_sigma = sigma
                self.model = model
                return sigma

        # Fallback to constant
        if self.verbose:
            print(f"\n  ⚠️  Using constant parameters (global σ=0)")

        self.optimal_sigma = 0.0
        self.model = self._create_model_with_variation(0.0, self.local_param_variation)
        return 0.0

    def generate_dataset(
        self,
        n_samples: int,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate dataset with optimal variation.

        Args:
            n_samples: Number of samples to generate
            output_file: Optional output CSV file path

        Returns:
            DataFrame with generated data
        """
        # Find optimal if not done yet
        if self.model is None:
            self.find_optimal_variation()

        if self.verbose:
            print(f"\nGenerating {n_samples} samples with global σ={self.optimal_sigma:.3f}...")

        # Use SteadyStateGenerator (has LSODA → bounded fallback)
        generator = SteadyStateGenerator(sm=self.model)
        dataset = generator.generate_signals(n=n_samples)

        if self.verbose:
            conv_rate = len(dataset) / n_samples if n_samples > 0 else 0
            print(f"  ✅ Generated {len(dataset)}/{n_samples} samples ({conv_rate*100:.1f}% success)")

        if output_file:
            dataset.to_csv(output_file, index=False)
            if self.verbose:
                print(f"  💾 Saved to: {output_file}")

        return dataset


def generate_datasets_for_model(
    sbml_file_path: str,
    output_dir: str = 'ml_datasets',
    pretrain_samples: int = 300,
    finetune_samples: int = 100,
    eval_samples: int = 100,
    min_convergence_rate: float = 0.7,
    global_param_variation: Optional[float] = None,
    local_param_variation: float = 0.0,
    species_variation: float = 0.3,
    verbose: bool = True
) -> Dict[str, Path]:
    """Generate pretrain/finetune/eval datasets for a model.

    Args:
        sbml_file_path: Path to SBML file
        output_dir: Output directory
        pretrain_samples: Pretrain dataset size
        finetune_samples: Finetune dataset size
        eval_samples: Eval dataset size
        min_convergence_rate: Target convergence rate
        global_param_variation: Fixed sigma for global params (None = auto)
        local_param_variation: Sigma for local params (0 = constant)
        species_variation: Sigma for initial concentrations (default 0.3 = 30%)
        verbose: Print progress

    Returns:
        Dictionary mapping split names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    model_id = Path(sbml_file_path).stem

    # Create generator
    generator = ImprovedAdaptiveGenerator(
        sbml_file_path=sbml_file_path,
        min_convergence_rate=min_convergence_rate,
        global_param_variation=global_param_variation,
        local_param_variation=local_param_variation,
        species_variation=species_variation,
        verbose=verbose
    )

    # Find optimal variation once
    generator.find_optimal_variation()

    # Generate datasets
    datasets = {
        'pretrain': pretrain_samples,
        'finetune': finetune_samples,
        'eval': eval_samples
    }

    output_files = {}

    for split, n_samples in datasets.items():
        output_file = output_path / f'{model_id}_{split}.csv'
        dataset = generator.generate_dataset(n_samples, output_file=str(output_file))
        output_files[split] = output_file

    if verbose:
        print(f"\n✅ All datasets generated for {model_id}")

    return output_files
