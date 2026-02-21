"""Modal executor stub for synix-bench.

Placeholder for future Modal cloud execution support.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class ModalExecutor:
    """Execute agent tools via Modal cloud functions.

    This is a stub -- Modal integration is not yet implemented.
    Implements the ToolExecutor protocol defined in synix.executor.base.
    """

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "ModalExecutor is not yet implemented. "
            "Use ContainerExecutor or LocalExecutor instead."
        )

    def __call__(self, name: str, args: dict) -> str:
        raise NotImplementedError("ModalExecutor is not yet implemented.")

    def get_patch(self) -> str:
        raise NotImplementedError("ModalExecutor is not yet implemented.")
