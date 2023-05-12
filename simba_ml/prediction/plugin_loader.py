"""A simple plugin loader."""
import os
import sys
import importlib


class PluginInterface:
    """Represents a plugin interface.

    A plugin has a single register function.
    """

    # sourcery skip: do-not-use-staticmethod
    @staticmethod
    def register() -> None:
        """Register the necessary items in the game character factory."""


def import_module(name: str) -> PluginInterface:
    """Imports a module given a name.

    Args:
        name: The name of the module to import.

    Returns:
        The imported module.
    """
    return importlib.import_module(name)  # type: ignore


def load_plugins(plugins: list[str]) -> None:
    """Loads the plugins defined in the plugins list.

    Args:
        plugins: A list of plugins (complete names) to load.
    """
    sys.path.append(os.getcwd())
    for plugin_file in plugins:
        plugin = import_module(plugin_file)
        plugin.register()
