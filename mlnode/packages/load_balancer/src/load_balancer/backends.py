"""Backend tracking primitives for the MLNode load balancer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from .config import Settings


@dataclass
class Backend:
    """Represents a single MLNode backend instance."""

    url: str
    settings: Settings
    state: Optional[str] = None
    healthy: bool = False
    active_requests: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def mark_request_start(self) -> None:
        async with self._lock:
            self.active_requests += 1

    async def mark_request_done(self) -> None:
        async with self._lock:
            if self.active_requests > 0:
                self.active_requests -= 1

    def is_available(self) -> bool:
        return self.healthy and self.state == "INFERENCE"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "state": self.state,
            "healthy": self.healthy,
            "active_requests": self.active_requests,
        }


class BackendPool:
    """Collection of MLNode backends with load balancing utilities."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backends = [Backend(url=b, settings=settings) for b in settings.backend_urls]
        self._pick_lock = asyncio.Lock()

    def __iter__(self):
        return iter(self.backends)

    def __len__(self) -> int:
        return len(self.backends)

    async def pick_backend(self) -> Backend:
        """Return the healthiest backend with the least active requests."""
        async with self._pick_lock:
            candidates = [b for b in self.backends if b.is_available()]
            if not candidates:
                raise RuntimeError("No healthy inference backends available")

            backend = min(candidates, key=lambda b: b.active_requests)
            await backend.mark_request_start()
            return backend

    async def release_backend(self, backend: Backend) -> None:
        await backend.mark_request_done()

    def aggregate_state(self) -> str:
        states = [b.state for b in self.backends if b.state]
        priority = ["INFERENCE", "POW", "TRAIN", "STOPPED"]
        for state in priority:
            if state in states:
                return state
        return "STOPPED"

    def any_healthy(self) -> bool:
        return any(b.is_available() for b in self.backends)

    def to_dict(self) -> dict:
        return {
            "state": self.aggregate_state(),
            "nodes": [b.to_dict() for b in self.backends],
        }


__all__ = ["Backend", "BackendPool"]
