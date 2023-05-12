"""Starts an interactive interface to visualize the a signal."""

import sys
import os

import importlib
import argparse

import streamlit as st
import pandas as pd


# pylint: disable=too-many-locals
def main() -> None:
    """Visualizes a system model.

    Visualizes a system model by showing the different steps of creating a signal.
    Creates a clean signal in the first step.
    This happens by integrating the system model `deriv` function.

    Note:
        The `deriv` function is already noised if a `DerivNoiser` is provided.

    Then, the noisers is applied to the clean signal, before in the last step,
        the signal gets sparsfied.
    """
    st.set_page_config(layout="wide")

    parser = argparse.ArgumentParser(description="Vizualize a system model.")
    parser.add_argument("--module", type=str, help="Module containing the system model")
    args = parser.parse_args()

    sys.path.append(os.getcwd())
    sm = importlib.import_module(__normalize_module_name(args.module)).sm
    st.title(f"Visualization of {sm.name}")

    st.button("Regenerate")

    start_values = sm.sample_start_values_from_hypercube(1)
    clean_signal = sm.get_clean_signal(start_values, 0)
    __show_signal("Clean-Data", clean_signal)
    noised_signal = sm.apply_noisifier(clean_signal)
    __show_signal("Noise", noised_signal)
    sparsed_signal = sm.apply_sparsifier(noised_signal)
    __show_signal("Output", sparsed_signal)


def __show_signal(signal_name: str, signal: pd.DataFrame) -> None:
    st.header(signal_name)
    with st.expander("Raw Data"):
        st.write(signal)
    st.line_chart(signal)


def __normalize_module_name(name: str) -> str:
    r"""Normalizes the module name.

    Removing the `.py` ending and replaces slash and backslash with `.`.

    Args:
        name: the unnormalized module name.

    Returns:
        The normalized module name.

    Example:
        >>> __normalize_module_name("module")
        'module'
        >>> __normalize_module_name("path/module")
        'path.module'
        >>> __normalize_module_name("path\\module")
        'path.module'
        >>> __normalize_module_name("module.py")
        'module'
    """
    return name.removesuffix(".py").replace("/", ".").replace("\\", ".")


if __name__ == "__main__":
    main()
