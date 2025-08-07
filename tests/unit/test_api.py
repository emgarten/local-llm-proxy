from pathlib import Path
import importlib
import typing
import httpx
import json

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock
import yaml


class GeneratorStream(httpx.AsyncByteStream):
    def __init__(self, gen: typing.AsyncIterator[bytes]) -> None:
        self._gen = gen

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        async for chunk in self._gen:
            yield chunk

    async def aclose(self) -> None:
        if hasattr(self._gen, "aclose"):
            await self._gen.aclose()


@pytest.fixture()
def create_config(tmp_path: Path) -> Path:
    cfg_file = tmp_path / ".prompt-passage.yaml"
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
def create_config_azure(tmp_path: Path) -> Path:
    cfg_file = tmp_path / ".prompt-passage.yaml"
    cfg_data = {
        "providers": {
            "test-model": {
                "endpoint": "https://mock.upstream/chat/completions",
                "model": "remote-model",
                "auth": {"type": "azure"},
            }
        }
    }
    cfg_file.write_text(yaml.dump(cfg_data))
    return cfg_file


@pytest.fixture()
def create_config_service_auth(tmp_path: Path) -> Path:
    cfg_file = tmp_path / ".prompt-passage.yaml"
    cfg_data = {
        "service": {"auth": {"type": "apikey", "key": "svc-key"}},
        "providers": {
            "test-model": {
                "endpoint": "https://mock.upstream/chat/completions",
                "model": "remote-model",
                "auth": {
                    "type": "apikey",
                    "envKey": "TEST_API_KEY_ENV",
                },
            }
        },
    }
    cfg_file.write_text(yaml.dump(cfg_data))
    return cfg_file


@pytest.fixture()
def create_config_transform(tmp_path: Path) -> Path:
    cfg_file = tmp_path / ".prompt-passage.yaml"
    cfg_data = {
        "providers": {
            "test-transform": {
                "endpoint": "https://mock.upstream/chat/completions",
                "model": "remote-model",
                "transform": ".messages as $m | .inputs=$m | del(.messages)",
                "auth": {
                    "type": "apikey",
                    "envKey": "TEST_API_KEY_ENV",
                },
            }
        }
    }
    cfg_file.write_text(yaml.dump(cfg_data))
    return cfg_file


def test_chat_proxy_success(monkeypatch: pytest.MonkeyPatch, create_config: Path, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("HOME", str(create_config.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "secret-token")

    proxy_app = importlib.import_module("prompt_passage.proxy_app")

    httpx_mock.add_response(url="https://mock.upstream/chat/completions", json={"ok": True})

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    req = httpx_mock.get_requests()[0]
    assert req.headers["Authorization"] == "Bearer secret-token"


def test_chat_proxy_upstream_error(monkeypatch: pytest.MonkeyPatch, create_config: Path, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("HOME", str(create_config.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "token")

    proxy_app = importlib.import_module("prompt_passage.proxy_app")

    import httpx

    httpx_mock.add_exception(httpx.ConnectError("boom"))

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 502
        assert resp.json() == {"error": "Upstream failure"}


def test_chat_proxy_azure(monkeypatch: pytest.MonkeyPatch, create_config_azure: Path, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("HOME", str(create_config_azure.parent))

    token_obj = type("Tok", (), {"token": "cli-token"})()

    class DummyCred:
        def get_token(self, scope: str) -> object:
            assert scope == "https://cognitiveservices.azure.com/.default"
            return token_obj

    proxy_app = importlib.import_module("prompt_passage.proxy_app")
    monkeypatch.setattr("prompt_passage.auth_providers.DefaultAzureCredential", lambda: DummyCred())

    httpx_mock.add_response(url="https://mock.upstream/chat/completions", json={"ok": True})

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    req = httpx_mock.get_requests()[0]
    assert req.headers["Authorization"] == "Bearer cli-token"


def test_chat_proxy_stream(monkeypatch: pytest.MonkeyPatch, create_config: Path, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("HOME", str(create_config.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "tok")

    proxy_app = importlib.import_module("prompt_passage.proxy_app")

    async def gen() -> typing.AsyncIterator[bytes]:
        yield b'data: {"id":1}\n\n'
        yield b"data: [DONE]\n\n"

    stream = GeneratorStream(gen())
    httpx_mock.add_response(
        url="https://mock.upstream/chat/completions",
        headers={"content-type": "text/event-stream"},
        stream=stream,
    )

    with TestClient(proxy_app.app) as client:
        with client.stream(
            "POST",
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as resp:
            chunks = list(resp.iter_bytes())

    assert b"data: {" in chunks[0]


def test_chat_proxy_transform(
    monkeypatch: pytest.MonkeyPatch, create_config_transform: Path, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("HOME", str(create_config_transform.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "tok")
    import sys

    sys.modules.pop("prompt_passage.proxy_app", None)
    proxy_app = importlib.import_module("prompt_passage.proxy_app")
    httpx_mock.add_response(url="https://mock.upstream/chat/completions", json={"ok": True})

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-transform/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    req = httpx_mock.get_requests()[0]
    data = json.loads(req.content.decode("utf-8"))
    assert data == {
        "model": "remote-model",
        "inputs": [{"role": "user", "content": "hi"}],
    }
    sys.modules.pop("prompt_passage.proxy_app", None)


def test_chat_proxy_unknown_provider(monkeypatch: pytest.MonkeyPatch, create_config: Path) -> None:
    monkeypatch.setenv("HOME", str(create_config.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "tok")

    proxy_app = importlib.import_module("prompt_passage.proxy_app")

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/missing/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 404
        assert resp.json() == {"error": "Unknown provider"}


def test_chat_proxy_upstream_500(monkeypatch: pytest.MonkeyPatch, create_config: Path, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("HOME", str(create_config.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "tok")

    proxy_app = importlib.import_module("prompt_passage.proxy_app")

    httpx_mock.add_response(status_code=500, json={"err": 1})
    httpx_mock.add_response(status_code=500, json={"err": 1})

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 500


def test_chat_proxy_stream_upstream_error(
    monkeypatch: pytest.MonkeyPatch, create_config: Path, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("HOME", str(create_config.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "tok")

    proxy_app = importlib.import_module("prompt_passage.proxy_app")

    httpx_mock.add_exception(httpx.ConnectError("fail"))

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 502
        assert resp.json() == {"error": "Upstream failure"}


def test_chat_proxy_stream_500(monkeypatch: pytest.MonkeyPatch, create_config: Path, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("HOME", str(create_config.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "tok")

    proxy_app = importlib.import_module("prompt_passage.proxy_app")

    def make_stream() -> GeneratorStream:
        async def gen() -> typing.AsyncIterator[bytes]:
            yield b"oops"

        return GeneratorStream(gen())

    httpx_mock.add_response(status_code=500, headers={"content-type": "text/event-stream"}, stream=make_stream())
    httpx_mock.add_response(status_code=500, headers={"content-type": "text/event-stream"}, stream=make_stream())

    with TestClient(proxy_app.app) as client:
        with client.stream(
            "POST",
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as resp:
            list(resp.iter_bytes())

        assert resp.status_code == 500


def test_service_auth_valid(
    monkeypatch: pytest.MonkeyPatch, create_config_service_auth: Path, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("HOME", str(create_config_service_auth.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "tok")

    proxy_app = importlib.import_module("prompt_passage.proxy_app")

    httpx_mock.add_response(url="https://mock.upstream/chat/completions", json={"ok": True})

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer svc-key"},
        )
        assert resp.status_code == 200


def test_service_auth_invalid(monkeypatch: pytest.MonkeyPatch, create_config_service_auth: Path) -> None:
    monkeypatch.setenv("HOME", str(create_config_service_auth.parent))
    monkeypatch.setenv("TEST_API_KEY_ENV", "tok")

    proxy_app = importlib.import_module("prompt_passage.proxy_app")

    with TestClient(proxy_app.app) as client:
        resp = client.post(
            "/provider/test-model/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
        assert resp.json() == {"error": "Unauthorized"}
