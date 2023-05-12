"""Generator that generates a specific amounts of time points."""

from simba_ml.simulation.generators import time_series_generator
from simba_ml.simulation.system_model import system_model_interface


class TimePointsGenerator:
    """Defines how to generator time points from a Prediction Task."""

    def __init__(self, sm: system_model_interface.SystemModelInterface):
        """Initializes the `TimePointsGenerator`.

        Args:
            sm: A `SystemModel`, for which the signals should be built.
        """
        self.time_series_generator = time_series_generator.TimeSeriesGenerator(sm)

    def generate_timepoints(
        self, number_of_timepoints: int = 10, save_dir: str = "./data/"
    ) -> None:
        """Generates a specific amount of time points.

        Args:
            number_of_timepoints: The number of time points that should be generated.
            save_dir: The name of the directory to save the data. Default is './data/'.

        Raises:
            ValueError: If the number of time points the user
                wants generate is negative.
        """
        if number_of_timepoints == 0:
            return
        if number_of_timepoints < 0:
            raise ValueError(
                f"Number of timepoints must be greater than 0"
                f", but is {number_of_timepoints}"
            )
        number_of_generated_timepoints = 0
        file_counter = 0
        while number_of_generated_timepoints < number_of_timepoints:
            df = self.time_series_generator.generate_signal()
            number_of_generated_timepoints += len(df)

            if number_of_generated_timepoints > number_of_timepoints:
                df = df.head(number_of_timepoints - number_of_generated_timepoints)

            df.to_csv(
                f"{save_dir}{self.time_series_generator.sm.name}_{file_counter}.csv"
            )
            file_counter += 1
