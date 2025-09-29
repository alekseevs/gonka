"""FastAPI application exposing the MLNode load balancer."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .backends import BackendPool
from .config import Settings
from .monitor import monitor_backend

SUPPORTED_METHODS = [
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "OPTIONS",
    "HEAD",
]


async def _proxy_request(
    request: Request,
    backend_pool: BackendPool,
    client: httpx.AsyncClient,
    mount_path: str,
) -> Response:
    try:
        backend = await backend_pool.pick_backend()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    target_path = request.url.path
    if mount_path:
        target_path = target_path[len(mount_path) :]
        if not target_path.startswith("/"):
            target_path = "/" + target_path

    url = f"{backend.url}{target_path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    async def _body_iterator():
        async for chunk in request.stream():
            yield chunk

    context = client.stream(
        request.method,
        url,
        params=request.query_params,
        headers=headers,
        content=_body_iterator(),
        timeout=httpx.Timeout(None, read=backend_pool.settings.request_timeout),
    )
    try:
        upstream = await context.__aenter__()

        filtered_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
        }

        async def _cleanup():
            try:
                await context.__aexit__(None, None, None)
            finally:
                await backend_pool.release_backend(backend)

        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=filtered_headers,
            background=BackgroundTask(_cleanup),
        )
    except Exception as exc:
        with contextlib.suppress(Exception):
            await context.__aexit__(None, None, None)
        await backend_pool.release_backend(backend)
        raise HTTPException(status_code=502, detail="Upstream request failed") from exc


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.load()
    pool = BackendPool(settings)
    client = httpx.AsyncClient(http2=True)
    monitor_tasks = [asyncio.create_task(monitor_backend(backend, client)) for backend in pool]

    app.state.settings = settings
    app.state.backend_pool = pool
    app.state.http_client = client
    app.state.monitor_tasks = monitor_tasks

    try:
        yield
    finally:
        for task in monitor_tasks:
            task.cancel()
        await asyncio.gather(*monitor_tasks, return_exceptions=True)
        await client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/api/v1/state")
async def get_state(request: Request):
    pool: BackendPool = request.app.state.backend_pool
    return JSONResponse(pool.to_dict())


@app.get("/health")
async def health(request: Request):
    pool: BackendPool = request.app.state.backend_pool
    if not pool.any_healthy():
        raise HTTPException(status_code=503, detail="No healthy inference backends")
    return JSONResponse({"status": "healthy"})


@app.api_route("/v1/{path:path}", methods=SUPPORTED_METHODS)
async def proxy_inference(path: str, request: Request):
    pool: BackendPool = request.app.state.backend_pool
    client: httpx.AsyncClient = request.app.state.http_client
    return await _proxy_request(request, pool, client, mount_path="")


@app.api_route("/{path:path}", methods=SUPPORTED_METHODS, include_in_schema=False)
async def proxy_fallback(path: str, request: Request):
    pool: BackendPool = request.app.state.backend_pool
    client: httpx.AsyncClient = request.app.state.http_client
    # For non-inference routes just proxy to the first backend to maintain compatibility.
    backend = next(iter(pool.backends), None)
    if backend is None:
        raise HTTPException(status_code=503, detail="No MLNode backends configured")
    # Temporarily mark backend as busy to keep accounting consistent.
    await backend.mark_request_start()
    path_component = request.url.path or "/"
    url = f"{backend.url}{path_component}"
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    async def _body_iterator():
        async for chunk in request.stream():
            yield chunk

    context = client.stream(
        request.method,
        url,
        params=request.query_params,
        headers=headers,
        content=_body_iterator(),
        timeout=httpx.Timeout(None, read=pool.settings.request_timeout),
    )
    try:
        upstream = await context.__aenter__()

        filtered_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
        }

        async def _cleanup():
            try:
                await context.__aexit__(None, None, None)
            finally:
                await backend.mark_request_done()

        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=filtered_headers,
            background=BackgroundTask(_cleanup),
        )
    except Exception as exc:
        with contextlib.suppress(Exception):
            await context.__aexit__(None, None, None)
        await backend.mark_request_done()
        raise HTTPException(status_code=502, detail="Upstream request failed") from exc


__all__ = ["app"]
