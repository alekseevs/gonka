"""Background monitoring for MLNode backends."""

from __future__ import annotations

import asyncio

import httpx

from .backends import Backend


async def _fetch_json(client: httpx.AsyncClient, url: str, timeout: float) -> dict | None:
    try:
        response = await client.get(url, timeout=timeout)
    except Exception:
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


async def _fetch_status(client: httpx.AsyncClient, url: str, timeout: float) -> bool:
    try:
        response = await client.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


async def monitor_backend(backend: Backend, client: httpx.AsyncClient) -> None:
    """Continuously update backend state and health."""
    settings = backend.settings
    state_url = f"{backend.url}/api/v1/state"
    health_url = f"{backend.url}/health"

    while True:
        state_data = await _fetch_json(client, state_url, settings.state_timeout)
        backend.state = state_data.get("state") if state_data else None
        backend.healthy = await _fetch_status(client, health_url, settings.health_timeout)
        await asyncio.sleep(settings.refresh_interval)

