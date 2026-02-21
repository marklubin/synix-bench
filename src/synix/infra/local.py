"""Local infrastructure provider.

Checks podman/docker socket availability. No deployment needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


class LocalProvider:
    """Local infrastructure — just verifies podman/docker is available."""

    def __init__(self, container_cmd: str = "podman", docker_host: str | None = None) -> None:
        self.container_cmd = container_cmd
        self.docker_host = docker_host or os.environ.get(
            "DOCKER_HOST",
            f"unix:///run/user/{os.getuid()}/podman/podman.sock",
        )

    def deploy(self, config: dict) -> str:
        """No-op for local. Just verify socket exists."""
        self._check_socket()
        return "local"

    def teardown(self, endpoint: str) -> None:
        """No-op for local."""

    def status(self, endpoint: str) -> dict:
        """Check if podman/docker socket is available."""
        socket_path = self.docker_host.replace("unix://", "")
        available = Path(socket_path).exists()
        return {
            "provider": "local",
            "container_cmd": self.container_cmd,
            "socket": self.docker_host,
            "available": available,
        }

    def get_endpoint(self) -> str:
        """Local has no remote endpoint."""
        return "local"

    def _check_socket(self) -> None:
        """Verify the container socket exists."""
        socket_path = self.docker_host.replace("unix://", "")
        if not Path(socket_path).exists():
            log.warning(
                "Container socket not found at %s. "
                "Start with: systemctl --user start podman.socket",
                socket_path,
            )
