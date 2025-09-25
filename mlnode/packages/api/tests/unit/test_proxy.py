from types import SimpleNamespace
from typing import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.proxy import ProxyMiddleware, setup_vllm_proxy, start_vllm_proxy, stop_vllm_proxy


class DummyURL(SimpleNamespace):
    """Minimal URL object that reflects the request scope."""

    def __init__(self, scope):
        super().__init__()
        self._scope = scope

    @property
    def path(self) -> str:
        return self._scope.get("path", "")


class DummyRequest:
    """Lightweight stand-in for FastAPI Request used in unit tests."""

    def __init__(self, path: str, method: str = "GET", query_string: str = ""):
        raw_path = path
        if query_string:
            raw_path = f"{path}?{query_string}"

        self.scope = {
            "type": "http",
            "path": path,
            "raw_path": raw_path.encode("latin-1"),
            "query_string": query_string.encode("latin-1"),
            "state": {},
        }
        self.method = method
        self.headers = {}
        self.query_params = {}
        self.state = {}

    @property
    def url(self):
        return DummyURL(self.scope)

    async def stream(self):
        if False:  # pragma: no cover - we just need an async generator
            yield b""


@pytest.fixture
def proxy_middleware():
    mock_app = MagicMock()
    return ProxyMiddleware(mock_app)


@pytest.fixture
def make_request() -> Callable[[str, str, str], DummyRequest]:
    def _make_request(path: str, method: str = "GET", query_string: str = "") -> DummyRequest:
        return DummyRequest(path=path, method=method, query_string=query_string)

    return _make_request


@pytest.mark.asyncio
async def test_proxy_middleware_routes_v1_to_vllm(proxy_middleware, make_request):
    """Test that /v1 requests are routed to vLLM backend."""

    # Mock the proxy method on the middleware instance
    with patch.object(proxy_middleware, "_proxy_to_vllm", new_callable=AsyncMock) as mock_proxy:
        mock_proxy.return_value = MagicMock()

        # Mock call_next
        call_next = AsyncMock()

        # Test /v1 routing
        request = make_request("/v1/models")
        await proxy_middleware.dispatch(request, call_next)

        # Should call proxy, not call_next
        mock_proxy.assert_awaited_once_with(request)
        call_next.assert_not_called()


@pytest.mark.asyncio
async def test_proxy_middleware_routes_api_to_main(proxy_middleware, make_request):
    """Test that /api requests are routed to main API."""

    # Mock call_next
    call_next = AsyncMock()
    call_next.return_value = MagicMock()

    # Test /api routing
    request = make_request("/api/v1/inference")
    await proxy_middleware.dispatch(request, call_next)

    # Should call call_next, not proxy
    call_next.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_proxy_middleware_default_routing(proxy_middleware, make_request):
    """Test that other requests default to main API."""

    # Mock call_next
    call_next = AsyncMock()
    call_next.return_value = MagicMock()

    # Test default routing
    request = make_request("/health")
    await proxy_middleware.dispatch(request, call_next)

    # Should call call_next
    call_next.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_versioned_prefix_trimmed_for_api_routes(proxy_middleware, make_request):
    """Ensure versioned requests are rewritten to the legacy prefixes."""

    observed = {}

    async def call_next(request):
        observed["path"] = request.scope.get("path")
        observed["raw_path"] = request.scope.get("raw_path")
        observed["url_path"] = request.url.path
        return MagicMock()

    request = make_request("/v3.0.8/api/v1/state")
    await proxy_middleware.dispatch(request, call_next)

    assert observed["path"] == "/api/v1/state"
    assert observed["url_path"] == "/api/v1/state"
    assert observed["raw_path"] == b"/api/v1/state"


@pytest.mark.asyncio
async def test_versioned_prefix_preserves_query_string(proxy_middleware, make_request):
    """Ensure versioned requests keep their query strings when trimmed."""

    observed = {}

    async def call_next(request):
        observed["path"] = request.scope.get("path")
        observed["raw_path"] = request.scope.get("raw_path")
        observed["query_string"] = request.scope.get("query_string")
        return MagicMock()

    request = make_request("/v3.0.8/api/v1/state", query_string="foo=1")
    await proxy_middleware.dispatch(request, call_next)

    assert observed["path"] == "/api/v1/state"
    assert observed["raw_path"] == b"/api/v1/state?foo=1"
    assert observed["query_string"] == b"foo=1"


@pytest.mark.asyncio
async def test_proxy_returns_503_when_backends_not_healthy(proxy_middleware, make_request):
    """Test that proxy returns 503 when no backends are healthy."""
    from api.proxy import vllm_backend_ports, vllm_healthy

    # Setup backends but mark them as unhealthy
    original_backends = vllm_backend_ports.copy()
    original_healthy = vllm_healthy.copy()

    vllm_backend_ports.clear()
    vllm_backend_ports.extend([5001, 5002])
    vllm_healthy.update({5001: False, 5002: False})

    try:
        # Mock call_next
        call_next = AsyncMock()

        # Test /v1 routing when backends are unhealthy
        request = make_request("/v1/models")
        result = await proxy_middleware.dispatch(request, call_next)

        # Should return 503, not call call_next
        assert result.status_code == 503
        assert b"vLLM backend not ready" in result.body
        call_next.assert_not_called()
    finally:
        # Restore original state
        vllm_backend_ports.clear()
        vllm_backend_ports.extend(original_backends)
        vllm_healthy.clear()
        vllm_healthy.update(original_healthy)


def test_setup_vllm_proxy():
    """Test vLLM proxy setup."""
    backend_ports = [5001, 5002, 5003]
    
    setup_vllm_proxy(backend_ports)
    
    # Import here to get the updated global state
    from api.proxy import vllm_backend_ports, vllm_counts, vllm_healthy
    
    assert vllm_backend_ports == backend_ports
    assert all(port in vllm_counts for port in backend_ports)
    assert all(port in vllm_healthy for port in backend_ports)


@pytest.mark.asyncio
async def test_start_stop_vllm_proxy():
    """Test vLLM proxy start and stop."""
    
    # Test start
    await start_vllm_proxy()
    
    # Import here to get the updated global state
    from api.proxy import vllm_client
    assert vllm_client is not None
    
    # Test stop
    await stop_vllm_proxy()
    
    # Import again to get the updated state
    from api.proxy import vllm_client
    assert vllm_client is None
