"""Defines an abstract definition of `Distribution`."""
import typing

T = typing.TypeVar("T")


class Distribution(typing.Protocol[T]):
    """A Distribution presents a set of values a property can have.

    Note:
        If no explicit way for sampling with the hypercube,
        the following code-snippet can probably be used:
        ```
        exactness = 1000
        vals = self.get_random_values(n * exactness)
        vals = np.sort(vals)
        return [
            np.random.choice(vals[i:i+exactness
            ]) for i in range(0, len(vals), exactness)]
        ```
    """

    def get_random_values(self, n: int) -> list[T]:
        """Samples a random value due to the type of Distribution.

        Args:
            n: The number of values.
        """

    def get_samples_from_hypercube(self, n: int) -> list[T]:
        """Samples n values from a hypercube.

        Args:
            n: the number of samples.
        """
