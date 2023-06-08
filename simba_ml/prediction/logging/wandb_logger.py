"""Wrapper for wandb logging."""
import typing

import wandb

from simba_ml.prediction.logging import logging_config

P = typing.ParamSpec("P")
T = typing.TypeVar("T")


class WandbLogger:
    """Wrapper for wandb logging."""

    def __init__(self, config: logging_config.LoggingConfig | None) -> None:
        """Initializes the wandb logger.

        Args:
            config: The config of the wandb logger.
        """
        self.config = config

    def init(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Initializes wandb.

        Args:
            *args: The arguments to pass to wandb.init.
            **kwargs: The keyword arguments to pass to wandb.init.
        """
        if self.config is None:
            return
        wandb.init(
            *args, **kwargs, project=self.config.project, entity=self.config.entity
        )

    def __getattr__(self, name: str) -> typing.Callable[..., T | None]:
        """Passes the message to wandb if wandb logging is enabled.

        Args:
            name: The name of the wandb function to call.

        Returns:
            A function that calls the corresponding wandb function
            if wandb logging is enabled.
        """
        func = getattr(wandb, name)

        def outer(func: typing.Callable[..., T]) -> typing.Callable[..., T | None]:
            def pass_message(*args: typing.Any, **kwargs: typing.Any) -> T | None:
                return func(*args, **kwargs) if self.config is not None else None

            return pass_message

        return outer(func)
