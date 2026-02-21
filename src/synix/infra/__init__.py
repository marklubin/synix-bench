"""Infrastructure providers for synix-bench."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synix.core.config import InfraConfig
    from synix.infra.modal import ModalProvider


def create_provider(config: InfraConfig) -> ModalProvider | None:
    """Create an infrastructure provider from config.

    Returns None for local provider (no wrapper needed).
    """
    if config.provider == "modal":
        import os

        from synix.infra.modal import ModalProvider

        api_token = os.environ.get(config.modal_api_token_env, "")
        return ModalProvider(
            api_token=api_token,
            vllm_endpoint=config.modal_vllm_endpoint,
        )
    return None
