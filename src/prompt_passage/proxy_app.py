from __future__ import annotations

from typing import Dict, AsyncIterator
from contextlib import asynccontextmanager

import httpx
import logging
from uvicorn.logging import DefaultFormatter
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import json

from .config import load_config, ProviderCfg, default_config_path
from .forwarder import Forwarder

_handler = logging.StreamHandler()
_handler.setFormatter(DefaultFormatter(fmt="%(levelprefix)s %(message)s", use_colors=True))
_root = logging.getLogger()
_root.handlers.clear()
_root.addHandler(_handler)
_root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def _pretty(data: bytes) -> str:
    """Return a prettified string representation of *data* if it's JSON."""
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception:
        return data.decode("utf-8", errors="replace")
    return json.dumps(obj, indent=2, ensure_ascii=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup
    global _provider_map, _forwarder, _service_auth_key  # noqa: PLW0603

    config_path = default_config_path()
    cfg = load_config(config_path)
    _provider_map = cfg.providers
    _service_auth_key = cfg.service.auth.key if cfg.service and cfg.service.auth else None
    _forwarder = Forwarder(_provider_map)

    logger.info("Available providers:")
    for name, cfg in _provider_map.items():
        logger.info(f"  - {name}")

    yield

    # Shutdown
    if _forwarder:
        await _forwarder.aclose()


app = FastAPI(title="Prompt Passage", version="1.0.0", lifespan=lifespan)

_provider_map: Dict[str, ProviderCfg] = {}
_forwarder: Forwarder | None = None
_service_auth_key: str | None = None


@app.post("/provider/{provider}/openai/deployments/{model}/chat/completions")
async def chat_proxy_oai_deployment(provider: str, model: str, request: Request) -> Response:
    """Proxy for chat completions to the specified provider with deployment path."""
    return await chat_proxy(provider, request)


@app.post("/provider/{provider}/deployments/{model}/chat/completions")
async def chat_proxy_deployment(provider: str, model: str, request: Request) -> Response:
    """Proxy for chat completions to the specified provider with deployment path."""
    return await chat_proxy(provider, request)


@app.post("/provider/{provider}/{subpath:path}/chat/completions")
async def chat_proxy_wildcard(provider: str, subpath: str, request: Request) -> Response:
    """Proxy for chat completions to the specified provider with flexible subpath routing."""
    return await chat_proxy(provider, request)


@app.post("/provider/{provider}/chat/completions")
async def chat_proxy(provider: str, request: Request) -> Response:
    if _service_auth_key is not None:
        if request.headers.get("Authorization") != f"Bearer {_service_auth_key}":
            return Response(
                content='{"error": "Unauthorized"}',
                media_type="application/json",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
    if provider not in _provider_map:
        return Response(
            content='{"error": "Unknown provider"}',
            media_type="application/json",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    cfg = _provider_map[provider]
    endpoint = str(cfg.endpoint)

    out_headers = {}
    out_headers["Content-Type"] = "application/json"

    token = cfg.token_provider.get_token()
    if token:
        out_headers["Authorization"] = f"Bearer {token}"

    body_bytes = await request.body()
    stream = False
    if body_bytes:
        try:
            # Override the model to match the config
            body_json = json.loads(body_bytes.decode("utf-8"))
            body_json["model"] = cfg.model
            stream = bool(body_json.get("stream", False))
            if cfg.transform is not None:
                body_json = cfg.apply_transform(body_json)
            body = json.dumps(body_json).encode("utf-8")
        except json.JSONDecodeError:
            body = body_bytes
    else:
        body = body_bytes

    logger.info("Forwarding request to %s", endpoint)
    logger.info("Outgoing body:\n%s", _pretty(body))

    assert _forwarder is not None
    try:
        if stream:
            upstream = await _forwarder.stream(endpoint, body, out_headers)
        else:
            upstream = await _forwarder.forward(endpoint, body, out_headers)
    except httpx.RequestError as exc:
        logger.exception("Failed to reach upstream: %s", exc)
        raise

    if stream:
        logger.info("Streaming response with status %s", upstream.status_code)

        async def _aiter() -> AsyncIterator[bytes]:
            async for chunk in upstream.aiter_raw():
                yield chunk

        return StreamingResponse(
            _aiter(),
            status_code=upstream.status_code,
            headers=dict(upstream.headers),
            media_type=upstream.headers.get("content-type"),
            background=BackgroundTask(upstream.aclose),
        )
    else:
        resp_pretty = _pretty(upstream.content)
        logger.info(
            "Upstream response (%s):\n%s",
            upstream.status_code,
            resp_pretty,
        )
        try:
            usage = json.loads(upstream.content.decode("utf-8")).get("usage")
        except Exception:
            usage = None
        if usage is not None:
            logger.info("Usage results: %s", usage)

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=dict(upstream.headers),
            media_type=upstream.headers.get("content-type"),
        )


@app.exception_handler(httpx.RequestError)
async def _httpx_error(_: Request, exc: httpx.RequestError) -> Response:
    """Return a generic 502 response on httpx failures."""
    logger.error("Upstream request error: %s", exc)
    return Response(
        content='{"error": "Upstream failure"}',
        media_type="application/json",
        status_code=status.HTTP_502_BAD_GATEWAY,
    )
