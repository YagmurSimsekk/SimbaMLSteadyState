"""Provides the interface for generators."""
import typing

from simba_ml.simulation.system_model import system_model_interface


class GeneratorInterface(typing.Protocol):
    """Protocol for a generator."""

    def __init__(self, sm: system_model_interface.SystemModelInterface) -> None:
        """Inits the generator.

        Args:
            sm: The system model
        """

    def generate_csvs(self, n: int = 1, save_dir: str = "./data/") -> None:
        """Generate csvs for n time-series.

        Args:
            n: number of timeseries
            save_dir: directory to save the csvs

        Note:
            This method does not need to generate n csvs.
            E.g. the SteadyStateGenerator only creates one csv,
            but from n time series.
        """
