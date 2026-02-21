"""InfraProvider protocol for compute infrastructure.

Abstracts over local (podman), Modal, RunPod, etc.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class InfraProvider(Protocol):
    """Protocol for infrastructure providers."""

    def deploy(self, config: dict) -> str:
        """Deploy compute resources. Returns an endpoint URL or identifier."""
        ...

    def teardown(self, endpoint: str) -> None:
        """Tear down deployed resources."""
        ...

    def status(self, endpoint: str) -> dict:
        """Check status of deployed resources."""
        ...

    def get_endpoint(self) -> str:
        """Get the API endpoint URL for the deployed resources."""
        ...
