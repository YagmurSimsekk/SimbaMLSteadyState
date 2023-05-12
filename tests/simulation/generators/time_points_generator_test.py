"""Test the time_points_generator module."""
import os
import shutil

import pandas as pd
import pytest as pt

from simba_ml.simulation.generators import time_points_generator
from simba_ml.simulation import distributions, species, noisers, system_model


def test_generate_csvs_with_number_of_datapoints():
    os.makedirs("./test_data/synthetic_data/", exist_ok=True)
    sm = create_example_prediction_task(5)
    generator = time_points_generator.TimePointsGenerator(sm)
    generator.generate_timepoints(
        number_of_timepoints=11,
        save_dir="./test_data/synthetic_data/",
    )
    assert (
        get_number_of_datapoints_in_folder(path_to_folder="./test_data/synthetic_data/")
        == 11
    )
    shutil.rmtree("./test_data/synthetic_data/")


def test_generate_csvs_with_number_of_datapoints_equals_zero():
    os.makedirs("./test_data/synthetic_data/", exist_ok=True)
    sm = create_example_prediction_task(5)
    generator = time_points_generator.TimePointsGenerator(sm)
    generator.generate_timepoints(
        number_of_timepoints=0,
        save_dir="./test_data/synthetic_data/",
    )
    assert (
        get_number_of_datapoints_in_folder(path_to_folder="./test_data/synthetic_data/")
        == 0
    )
    shutil.rmtree("./test_data/synthetic_data/")


def test_generate_csvs_with_number_of_datapoints_is_negative():
    sm = create_example_prediction_task(5)
    generator = time_points_generator.TimePointsGenerator(sm)
    with pt.raises(ValueError):
        generator.generate_timepoints(
            number_of_timepoints=-1,
            save_dir="./test_data/synthetic_data/",
        )


def create_example_prediction_task(number_of_timestamps=200):
    name = "Constant function"
    timestamps = distributions.Constant(number_of_timestamps)

    specieses = [species.Species("y", distributions.Constant(0))]

    kinetic_parameters = {}

    def deriv(_t, _y, _arguments):
        """Derivative of the function at the point _t.

        Returns:
            List[float]
        """
        return [0]

    noiser = noisers.additive_noiser.AdditiveNoiser(
        distributions.NormalDistribution(1, 0)
    )
    sm = system_model.SystemModel(
        name,
        specieses,
        kinetic_parameters,
        deriv=deriv,
        noiser=noiser,
        timestamps=timestamps,
    )
    return sm


def get_number_of_datapoints_in_file(path_to_data: str) -> int:
    df = pd.read_csv(f"{path_to_data}")
    return len(df)


def get_number_of_datapoints_in_folder(path_to_folder: str) -> int:
    number_of_datapoints = 0
    list_of_files = os.listdir(path_to_folder)
    for file in list_of_files:
        if file.endswith(".csv"):
            df = pd.read_csv(f"{path_to_folder}/{file}")
            number_of_datapoints += len(df)
    return number_of_datapoints
