"""Specifies a variable of a problem task."""
import typing

from simba_ml.simulation import distributions


class Species:
    """Specifies a variable of a problem task.

    Attributes:
        name: The name of the variable.
        distribution: A condition determining the value of the parameter at the start.
        contained_in_output: Whether or not this object should be included
            in the final output data.
        min_value:  Minimal allowed value for this species.
            If None, the species is considered to have not minimal value.
        max_value:  Maximal allowed value for this species.
            If None, the species is considered to have not maximal value.
    """

    def __init__(
        self,
        name: str,
        distribution: distributions.Distribution[float],
        contained_in_output: bool = True,
        min_value: typing.Optional[float] = None,
        max_value: typing.Optional[float] = None,
    ) -> None:
        """Inits Species with the provided params.

        Args:
            name: The name of the variable.
            distribution: A condition determining the value of the parameter
                at the start.
            contained_in_output: Whether this object should be included
                in the final output data.
            min_value:  Minimal allowed value for this species.
                If None, the species is considered to have not minimal value.
            max_value:  Maximal allowed value for this species.
                If None, the species is considered to have not maximal value.

        Raises:
            ValueError: If min_value is greater than max_value.
        """
        if min_value and max_value and min_value > max_value:
            raise ValueError("min_value should be smaller or equal then max_value.")

        self.name = name
        self.distribution = distribution
        self.contained_in_output = contained_in_output
        self.min = min_value
        self.max = max_value

    def get_initial_values_from_hypercube_sampling(self, n: int) -> typing.List[float]:
        """Creates a config for hypercube sampling.

        Args:
            n: Number of samples to create.

        Returns:
            A list of samples.
        """
        return [float(v) for v in self.distribution.get_samples_from_hypercube(n)]

    def __str__(self) -> str:
        """Returns a string representation of the object.

        Returns:
            A string representation of the object.
        """
        return f"Species(name={self.name})"
