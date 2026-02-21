"""Tests for ModalProvider — deploy, health check, teardown, factory."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from synix.core.config import InfraConfig, RunConfig
from synix.infra import create_provider
from synix.infra.modal import (
    DEFAULT_VLLM_ENDPOINT,
    MODAL_APPS,
    ModalProvider,
)


class TestModalProviderInit:
    def test_defaults(self):
        p = ModalProvider()
        assert p.vllm_endpoint == DEFAULT_VLLM_ENDPOINT
        assert p.api_token == ""
        assert p._deployed_apps == []

    def test_custom_endpoint(self):
        p = ModalProvider(vllm_endpoint="https://example.com/v1")
        assert p.vllm_endpoint == "https://example.com/v1"

    def test_custom_api_token(self):
        p = ModalProvider(api_token="test-token")
        assert p.api_token == "test-token"


class TestDeployApp:
    @patch("synix.infra.modal.subprocess.run")
    def test_deploy_calls_modal_cli(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Deployed synix-vllm\n", stderr="")
        p = ModalProvider()

        p._deploy_app("vllm")

        mock_run.assert_called_once_with(
            ["modal", "deploy", MODAL_APPS["vllm"]["module"]],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "vllm" in p._deployed_apps

    @patch("synix.infra.modal.subprocess.run")
    def test_deploy_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: auth failed")
        p = ModalProvider()

        with pytest.raises(RuntimeError, match="modal deploy failed"):
            p._deploy_app("vllm")
        assert p._deployed_apps == []


class TestHealthCheck:
    @patch("synix.infra.modal.urllib.request.urlopen")
    def test_healthy_vllm(self, mock_urlopen):
        body = {"status": "ok"}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(body).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = ModalProvider(api_token="test-tok")
        assert p._health_check("vllm", timeout=5) is True

    @patch("synix.infra.modal.urllib.request.urlopen")
    def test_healthy_embeddings(self, mock_urlopen):
        body = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(body).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = ModalProvider(api_token="test-tok")
        assert p._health_check("embeddings", timeout=5) is True

    @patch("synix.infra.modal.time.sleep")
    @patch("synix.infra.modal.time.monotonic")
    @patch("synix.infra.modal.urllib.request.urlopen")
    def test_timeout(self, mock_urlopen, mock_time, mock_sleep):
        # First call: start=0, then loop checks: 0 < 5 (enter), elapsed=0,
        # after error + sleep: 999 >= 5 (exit loop)
        call_count = 0

        def advancing_clock():
            nonlocal call_count
            call_count += 1
            # First two calls (start + while check): return 0
            # After that: return past deadline
            return 0 if call_count <= 2 else 999

        mock_time.side_effect = advancing_clock
        mock_urlopen.side_effect = ConnectionError("refused")

        p = ModalProvider()
        assert p._health_check("vllm", timeout=5) is False


class TestPing:
    @patch("synix.infra.modal.urllib.request.urlopen")
    def test_ping_success(self, mock_urlopen):
        body = {"status": "ok"}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(body).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = ModalProvider()
        assert p._ping("vllm") is True

    @patch("synix.infra.modal.urllib.request.urlopen")
    def test_ping_failure(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionError("refused")

        p = ModalProvider()
        assert p._ping("vllm") is False


class TestTeardown:
    @patch("synix.infra.modal.subprocess.run")
    def test_teardown_stops_deployed_apps(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        p = ModalProvider()
        p._deployed_apps = ["vllm"]

        p.teardown()

        mock_run.assert_called_once_with(
            ["modal", "app", "stop", "synix-vllm"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert p._deployed_apps == []

    @patch("synix.infra.modal.subprocess.run")
    def test_teardown_handles_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
        p = ModalProvider()
        p._deployed_apps = ["vllm"]

        # Should not raise, just warn
        p.teardown()
        assert p._deployed_apps == []


class TestStatus:
    @patch("synix.infra.modal.urllib.request.urlopen")
    def test_status_returns_dict(self, mock_urlopen):
        body = {"choices": [{"message": {"content": "OK"}}]}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(body).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = ModalProvider()
        s = p.status()
        assert "vllm" in s
        assert "embeddings" in s


class TestDeploy:
    @patch.object(ModalProvider, "_health_check", return_value=True)
    @patch.object(ModalProvider, "_deploy_app")
    def test_deploy_returns_endpoint(self, mock_deploy_app, mock_health):
        p = ModalProvider()
        url = p.deploy(timeout=10)
        assert url == DEFAULT_VLLM_ENDPOINT
        mock_deploy_app.assert_called_once_with("vllm")
        mock_health.assert_called_once_with("vllm", 10)

    @patch.object(ModalProvider, "_health_check", return_value=False)
    @patch.object(ModalProvider, "_deploy_app")
    def test_deploy_raises_on_health_failure(self, mock_deploy_app, mock_health):
        p = ModalProvider()
        with pytest.raises(RuntimeError, match="health check failed"):
            p.deploy(timeout=5)


class TestCreateProvider:
    def test_local_returns_none(self):
        config = InfraConfig(provider="local")
        assert create_provider(config) is None

    @patch.dict("os.environ", {"SYNIX_API_TOKEN": "test-token-123"})
    def test_modal_returns_provider(self):
        config = InfraConfig(
            provider="modal",
            modal_vllm_endpoint="https://custom.endpoint/v1",
        )
        prov = create_provider(config)
        assert isinstance(prov, ModalProvider)
        assert prov.api_token == "test-token-123"
        assert prov.vllm_endpoint == "https://custom.endpoint/v1"


class TestInfraConfigModal:
    def test_new_fields_default(self):
        config = InfraConfig()
        assert config.modal_skip_deploy is False
        assert config.modal_timeout == 300
        assert config.modal_vllm_endpoint is None
        assert config.modal_api_token_env == "SYNIX_API_TOKEN"

    def test_from_dict_with_modal_fields(self):
        config = InfraConfig.from_dict({
            "provider": "modal",
            "modal_skip_deploy": True,
            "modal_timeout": 600,
        })
        assert config.provider == "modal"
        assert config.modal_skip_deploy is True
        assert config.modal_timeout == 600

    def test_roundtrip_via_run_config(self):
        config = RunConfig(
            suite="swebench",
            strategy="naive",
            infra=InfraConfig(
                provider="modal",
                modal_vllm_endpoint="https://custom/v1",
            ),
        )
        d = config.to_dict()
        assert d["infra"]["provider"] == "modal"
        assert d["infra"]["modal_vllm_endpoint"] == "https://custom/v1"

        restored = RunConfig.from_dict(d)
        assert restored.infra.provider == "modal"
        assert restored.infra.modal_vllm_endpoint == "https://custom/v1"

    def test_local_infra_not_serialized(self):
        config = RunConfig(suite="lens", strategy="null")
        d = config.to_dict()
        assert "infra" not in d


class TestHealthPayload:
    def test_vllm_payload(self):
        payload = json.loads(ModalProvider._health_payload("vllm"))
        assert "messages" in payload
        assert payload["max_tokens"] == 8

    def test_embeddings_payload(self):
        payload = json.loads(ModalProvider._health_payload("embeddings"))
        assert "input" in payload
        assert payload["model"] == "Qwen/Qwen3-Embedding-0.6B"

    def test_unknown_app_raises(self):
        with pytest.raises(ValueError, match="Unknown app"):
            ModalProvider._health_payload("unknown")
