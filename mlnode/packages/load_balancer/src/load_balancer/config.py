"""Configuration utilities for the MLNode load balancer service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

DEFAULT_REFRESH_INTERVAL = 2.0
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_STATE_TIMEOUT = 5.0
DEFAULT_HEALTH_TIMEOUT = 2.0


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the load balancer."""

    backend_urls: List[str]
    refresh_interval: float = DEFAULT_REFRESH_INTERVAL
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    state_timeout: float = DEFAULT_STATE_TIMEOUT
    health_timeout: float = DEFAULT_HEALTH_TIMEOUT

    @classmethod
    def load(cls) -> "Settings":
        """Load configuration from environment variables."""
        raw_backends = os.getenv("MLNODE_BACKENDS", "")
        backends = [b.strip().rstrip("/") for b in raw_backends.split(",") if b.strip()]
        if not backends:
            raise RuntimeError(
                "MLNODE_BACKENDS environment variable must contain at least one backend URL"
            )

        def _float_env(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise RuntimeError(f"Invalid float value for {name}: {raw}") from exc

        return cls(
            backend_urls=backends,
            refresh_interval=_float_env("MLNODE_REFRESH_INTERVAL", DEFAULT_REFRESH_INTERVAL),
            request_timeout=_float_env("MLNODE_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT),
            state_timeout=_float_env("MLNODE_STATE_TIMEOUT", DEFAULT_STATE_TIMEOUT),
            health_timeout=_float_env("MLNODE_HEALTH_TIMEOUT", DEFAULT_HEALTH_TIMEOUT),
        )


__all__ = ["Settings"]
