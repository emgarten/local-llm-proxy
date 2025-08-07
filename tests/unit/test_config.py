from pathlib import Path

import pytest
from pydantic import ValidationError

from prompt_passage.config import load_config, parse_config
from prompt_passage.auth_providers import ApiKeyProvider
from prompt_passage.config import default_config_path


def test_load_config_file_not_found(tmp_path: Path) -> None:
    """Test FileNotFoundError when config file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "non_existent.yaml")


def test_parse_config_valid_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid config resolves an API key from the environment."""
    monkeypatch.setenv("TEST_ENV_KEY", "abc123")
    raw = {
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "apikey", "envKey": "TEST_ENV_KEY"},
            }
        }
    }
    cfg = parse_config(raw)
    assert cfg.providers["p1"].auth.api_key == "abc123"


def test_parse_config_missing_default_provider() -> None:
    """Defaults referencing unknown providers should fail validation."""
    raw = {
        "defaults": {"provider": "missing"},
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "apikey", "key": "k"},
            }
        },
    }
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_parse_config_providers_empty() -> None:
    """Configuration must contain at least one provider."""
    with pytest.raises(ValidationError):
        parse_config({"providers": {}})


def test_parse_config_apikey_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An apikey auth entry without key or envKey should fail."""
    monkeypatch.delenv("MISSING", raising=False)
    raw = {
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "apikey", "envKey": "MISSING"},
            }
        }
    }
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_parse_config_azure_returns_none() -> None:
    """Auth of type azure should not resolve an API key."""
    raw = {
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "azure"},
            }
        }
    }
    cfg = parse_config(raw)
    prov = cfg.providers["p1"].auth
    assert prov.api_key is None
    assert prov.provider is not None


def test_parse_config_apikey_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "x")
    raw = {
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "apikey", "envKey": "ENV"},
            }
        }
    }
    cfg = parse_config(raw)
    auth = cfg.providers["p1"].auth
    assert isinstance(auth.provider, ApiKeyProvider)


def test_parse_config_transform() -> None:
    raw = {
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "apikey", "key": "k"},
                "transform": ".messages as $m | .inputs=$m | del(.messages)",
            }
        }
    }
    cfg = parse_config(raw)
    prov = cfg.providers["p1"]
    transformed = prov.apply_transform({"model": "m", "messages": [1]})
    assert "inputs" in transformed and "messages" not in transformed


def test_parse_config_service_section() -> None:
    raw = {
        "service": {"port": 1234, "auth": {"type": "apikey", "key": "tok"}},
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "apikey", "key": "k"},
            }
        },
    }
    cfg = parse_config(raw)
    assert cfg.service is not None
    assert cfg.service.port == 1234
    assert cfg.service.auth is not None
    assert cfg.service.auth.key == "tok"


def test_parse_config_service_auth_missing_key() -> None:
    raw = {
        "service": {"auth": {"type": "apikey", "key": ""}},
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "apikey", "key": "k"},
            }
        },
    }
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_parse_config_service_missing() -> None:
    raw = {
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "apikey", "key": "k"},
            }
        },
    }
    cfg = parse_config(raw)
    assert cfg.service is None


def test_parse_config_service_port_default() -> None:
    raw = {
        "service": {"auth": {"type": "apikey", "key": "token"}},
        "providers": {
            "p1": {
                "endpoint": "https://example.com",
                "model": "m",
                "auth": {"type": "apikey", "key": "k"},
            }
        },
    }
    cfg = parse_config(raw)
    assert cfg.service is not None
    assert cfg.service.port == 8095


def test_default_config_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPT_PASSAGE_CONFIG_PATH", "/tmp/custom.yaml")
    assert default_config_path() == Path("/tmp/custom.yaml")


def test_default_config_path_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROMPT_PASSAGE_CONFIG_PATH", raising=False)
    monkeypatch.setenv("HOME", "/tmp/home")
    assert default_config_path() == Path("/tmp/home/.prompt-passage.yaml")
