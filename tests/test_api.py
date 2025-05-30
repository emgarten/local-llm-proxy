from pathlib import Path
import importlib

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock
import yaml


@pytest.fixture()
def create_config(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / ".local_llm_proxy"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.yaml"
    cfg_data = {
        "providers": {
            "test-model": {
                "endpoint": "https://mock.upstream/chat/completions",
                "model": "remote-model",
                "auth": {
                    "type": "apikey",
                    "envKey": "TEST_API_KEY_ENV",
                },
            }
        }
    }
    cfg_file.write_text(yaml.dump(cfg_data))
    return cfg_file


@pytest.fixture()
def create_config_azcli(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / ".local_llm_proxy"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.yaml"
    cfg_data = {
        "providers": {
            "test-model": {
                "endpoint": "https://mock.upstream/chat/completions",
                "model": "remote-model",
                "auth": {"type": "azcli"},
            }
        }
    }
    cfg_file.write_text(yaml.dump(cfg_data))
    return cfg_file


def test_chat_proxy_success(monkeypatch: pytest.MonkeyPatch, create_config: Path, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("HOME", str(create_config.parent.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "secret-token")

    proxy_app = importlib.import_module("local_llm_proxy.proxy_app")

    httpx_mock.add_response(url="https://mock.upstream/chat/completions", json={"ok": True})

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    req = httpx_mock.get_requests()[0]
    assert req.headers["Authorization"] == "Bearer secret-token"


def test_chat_proxy_upstream_error(monkeypatch: pytest.MonkeyPatch, create_config: Path, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("HOME", str(create_config.parent.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "token")

    proxy_app = importlib.import_module("local_llm_proxy.proxy_app")

    import httpx

    httpx_mock.add_exception(httpx.ConnectError("boom"))

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 502
        assert resp.json() == {"error": "Upstream failure"}


def test_chat_proxy_azcli(monkeypatch: pytest.MonkeyPatch, create_config_azcli: Path, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("HOME", str(create_config_azcli.parent.parent))

    token_obj = type("Tok", (), {"token": "cli-token"})()

    class DummyCred:
        def get_token(self, scope: str) -> object:
            assert scope == "https://cognitiveservices.azure.com/.default"
            return token_obj

    proxy_app = importlib.import_module("local_llm_proxy.proxy_app")
    monkeypatch.setattr("local_llm_proxy.auth_providers.AzureCliCredential", lambda: DummyCred())

    httpx_mock.add_response(url="https://mock.upstream/chat/completions", json={"ok": True})

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    req = httpx_mock.get_requests()[0]
    assert req.headers["Authorization"] == "Bearer cli-token"
