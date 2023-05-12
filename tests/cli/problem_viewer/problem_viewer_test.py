from unittest.mock import PropertyMock, patch

import pandas as pd


class StreamlitExpanderMock:
    """Mock a streamlit expander"""

    def __enter__(self):
        return "OK"

    def __exit__(self, type, value, tb):  # pylint: disable=redefined-builtin
        return True


class StreamlitColumnsMock:
    """Mock a streamlit columns"""

    def __enter__(self):
        return "OK"

    def __exit__(self, type, value, tb):  # pylint: disable=redefined-builtin
        return True


class StreamlitMock:
    """Mock streamlit by providing each function we use and logging the function_calls
    in self.calls"""

    def __init__(self, checkbox_value=False):
        self.calls = []
        self.checkbox_value = checkbox_value

    def title(self, arg):
        self.calls.append(("title", arg))

    def header(self, arg):
        self.calls.append(("header", arg))

    def expander(self, arg):
        self.calls.append(("expander", arg))
        return StreamlitExpanderMock()

    def columns(self, arg):
        self.calls.append(("columns", arg))
        return [StreamlitColumnsMock()] * arg

    def write(self, arg):
        self.calls.append(("write", arg))

    def line_chart(self, arg):
        pass

    def set_page_config(self, layout):
        self.calls.append(("page_config", layout))

    def button(self, arg):
        self.calls.append(("button", arg))

    def checkbox(self, arg):
        self.calls.append(("checkbox", arg))
        return self.checkbox_value

    def subheader(self, arg):
        self.calls.append(("subheader", arg))


@patch("simba_ml.cli.problem_viewer.problem_viewer.st", StreamlitMock())
@patch("simba_ml.example_problems.trigonometry.sm")
def test_problem_viewer_calls_streamlit_correctly(trigonometry_mock):
    # Mock simba_ml.example_problems.trigonometry so that we can test,
    # and providing a simple return of the sm object's methods.
    type(trigonometry_mock).name = PropertyMock(return_value="NAME_OF_PROBLEM")
    clean_signal = pd.DataFrame([[0, 1, 3], [2, 3, 3]])
    trigonometry_mock.get_clean_signal.return_value = clean_signal
    noised_signal = pd.DataFrame([[1, 2, 3], [4, 3, 3]])
    trigonometry_mock.apply_noisifier.return_value = noised_signal
    sparsed_signal = pd.DataFrame([[1, 2, 3], [4, 3, 4], [4, 5, 6]])
    trigonometry_mock.apply_sparsifier.return_value = sparsed_signal
    generated_signal = pd.DataFrame([[1, 2], [2, 3], [3, 4], [4, 5]])
    trigonometry_mock.generate_signal.return_value = generated_signal

    # the main method reads the arguments, so we have to provide them
    # by patching sys.argv
    with patch(
        "sys.argv", ["__main__", "--module", "simba_ml.example_problems.trigonometry"]
    ):
        # pylint: disable=import-outside-toplevel
        from simba_ml.cli.problem_viewer.problem_viewer import (
            main,
            st,
        )

        main()
        expected = [
            ("page_config", "wide"),
            ("title", "Visualization of NAME_OF_PROBLEM"),
            ("button", "Regenerate"),
            ("header", "Clean-Data"),
            ("expander", "Raw Data"),
            ("write", clean_signal),
            ("header", "Noise"),
            ("expander", "Raw Data"),
            ("write", noised_signal),
            ("header", "Output"),
            ("expander", "Raw Data"),
            ("write", sparsed_signal),
        ]
        # compare how the problem_viewer actually called streamlit
        # by how we expected it to call streamlit
        assert st.calls == expected  # pylint: disable=no-member
