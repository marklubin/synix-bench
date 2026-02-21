"""Modal infrastructure provider stub.

Placeholder for future Modal.com integration.
"""

from __future__ import annotations


class ModalProvider:
    """Modal compute provider — not yet implemented."""

    def deploy(self, config: dict) -> str:
        raise NotImplementedError("Modal provider not yet implemented")

    def teardown(self, endpoint: str) -> None:
        raise NotImplementedError("Modal provider not yet implemented")

    def status(self, endpoint: str) -> dict:
        raise NotImplementedError("Modal provider not yet implemented")

    def get_endpoint(self) -> str:
        raise NotImplementedError("Modal provider not yet implemented")
