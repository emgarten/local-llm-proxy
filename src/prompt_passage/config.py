"""Configuration loader for the proxy service.

Reads a YAML configuration file (default: `models.yaml`), validates its structure
according to the defined schema (including `providers`, `auth` settings, etc.),
and resolves API keys from environment variables if specified.
The result is an immutable `RootConfig` object containing the parsed configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Literal, Any, cast

import yaml
from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
    model_validator,
    PrivateAttr,
)

from .auth_providers import ApiKeyProvider, AzureCliProvider, TokenProvider
import jq


def default_config_path() -> Path:
    """Return the default user configuration file path."""
    override = os.getenv("PROMPT_PASSAGE_CONFIG_PATH")
    if override:
        return Path(override)
    return Path.home() / ".prompt-passage.yaml"


class AuthConfig(BaseModel):
    """Authentication configuration for a provider."""

    type: Literal["apikey", "azure"]
    envKey: str | None = None
    key: str | None = None
    _resolved_api_key: str | None = PrivateAttr(None)
    _token_provider: TokenProvider | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _resolve_and_validate_auth(self) -> "AuthConfig":
        if self.type == "apikey":
            if self.key is not None:
                self._resolved_api_key = self.key
            elif self.envKey is not None:
                if not self.envKey:  # Ensure envKey is not an empty string
                    raise ValueError("'envKey' for apikey auth must not be an empty string.")
                token_from_env = os.getenv(self.envKey)
                if not token_from_env:
                    raise ValueError(f"Environment variable '{self.envKey}' not set or empty for apikey auth")
                self._resolved_api_key = token_from_env
            else:
                raise ValueError("For apikey auth, either 'key' or 'envKey' must be provided.")
        elif self.type == "azure":
            # For 'azure', the token is expected to be handled externally (e.g., via Azure CLI login).
            # This configuration model does not resolve a token for 'azure'.
            # Warn if key/envKey are provided for azure, as they are not used by this model.
            if self.key is not None or self.envKey is not None:
                # Consider logging a warning here if a logger is available
                # print("Warning: 'key' or 'envKey' are provided for 'azure' auth type but will not be used for API key resolution by this model.")
                pass
        # Instantiate token provider after validation
        self._token_provider = self._build_provider()
        return self

    @property
    def api_key(self) -> str | None:
        """
        Returns the resolved API key if auth type is 'apikey'.
        Returns None for 'azure' as this model doesn't handle its token.
        """
        if self.type == "apikey":
            # This should be set by _resolve_and_validate_auth if type is apikey
            if self._resolved_api_key is None:
                raise ValueError(
                    "API key for 'apikey' auth was not resolved. This indicates an issue in validation logic."
                )
            return self._resolved_api_key
        return None

    def _build_provider(self) -> TokenProvider:
        if self.type == "apikey":
            assert self._resolved_api_key is not None
            return ApiKeyProvider(self._resolved_api_key)
        return AzureCliProvider()

    @property
    def provider(self) -> TokenProvider:
        assert self._token_provider is not None
        return self._token_provider


class ProviderCfg(BaseModel):
    """Run-time configuration for a single provider entry."""

    endpoint: str  # Endpoint URL, potentially with placeholders like {service}
    model: str  # Name of the LLM model, e.g., "o4-mini"
    auth: AuthConfig
    transform: str | None = None
    _provider: TokenProvider | None = PrivateAttr(None)
    _transform_prog: jq.Program | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _init_provider(self) -> "ProviderCfg":
        self._provider = self.auth.provider
        if self.transform is not None:
            self._transform_prog = jq.compile(self.transform)
        return self

    @property
    def token_provider(self) -> TokenProvider:
        assert self._provider is not None
        return self._provider

    def apply_transform(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._transform_prog is None:
            return body
        return cast(dict[str, Any], self._transform_prog.input(body).first())


class DefaultsCfg(BaseModel):
    """Default settings, like the default provider name."""

    provider: str


class ServiceAuthCfg(BaseModel):
    """Authentication configuration for the proxy service itself."""

    type: Literal["apikey"]
    key: str

    @field_validator("key")
    @classmethod
    def _key_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Service auth 'key' must not be empty")
        return v


class ServiceCfg(BaseModel):
    """Configuration for the running proxy service."""

    port: int = 8095
    auth: ServiceAuthCfg | None = None


class RootConfig(BaseModel):
    """Root configuration model."""

    defaults: DefaultsCfg | None = None
    service: ServiceCfg | None = None
    providers: Dict[str, ProviderCfg]

    @field_validator("providers")
    @classmethod
    def _validate_providers_not_empty(cls, v: Dict[str, ProviderCfg]) -> Dict[str, ProviderCfg]:
        if not v:
            raise ValueError("The 'providers' mapping in the configuration cannot be empty.")
        return v

    @model_validator(mode="after")
    def _validate_default_provider_exists(self) -> "RootConfig":
        if self.defaults and self.defaults.provider:
            if self.defaults.provider not in self.providers:
                raise ValueError(f"Default provider '{self.defaults.provider}' not found in the 'providers' list.")
        return self


def load_config(path: str | Path = "models.yaml") -> RootConfig:
    """
    Parse the YAML configuration file at *path* and return a `RootConfig` object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("rt", encoding="utf-8") as fp:
        raw_data = yaml.safe_load(fp)

    if not raw_data:
        raise ValueError(f"Configuration file is empty or invalid: {path}")

    try:
        config = parse_config(raw_data)
    except ValidationError as exc:
        # Provide a more context-rich error message if possible
        # For example, by iterating through exc.errors()
        error_details = "\n".join(
            f"  - {err['loc']}: {err['msg']} (input was {err.get('input')})" for err in exc.errors()
        )
        raise ValueError(f"Invalid configuration in '{path}':\n{error_details}") from exc

    return config


def parse_config(raw_data: Any) -> RootConfig:
    """
    Parse a raw configuration dictionary and return a `RootConfig` object.
    """

    return RootConfig(**raw_data)
