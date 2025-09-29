# MLNode Load Balancer

This service provides a lightweight FastAPI-based proxy that sits in front of multiple
MLNode instances. It keeps track of their health and current state and exposes
aggregated `/api/v1/state` and `/health` endpoints. Requests hitting inference
routes (`/v1/*`) are routed to the least-busy healthy backend that reports the
`INFERENCE` state.

## Configuration

The balancer is configured through environment variables:

- `MLNODE_BACKENDS` – comma-separated list of MLNode base URLs (required).
- `MLNODE_REFRESH_INTERVAL` – monitor refresh interval in seconds (default `2.0`).
- `MLNODE_REQUEST_TIMEOUT` – upstream request timeout in seconds (default `30.0`).
- `MLNODE_STATE_TIMEOUT` – timeout for state checks (default `5.0`).
- `MLNODE_HEALTH_TIMEOUT` – timeout for health checks (default `2.0`).

## Development

Install dependencies with poetry and run the FastAPI app with uvicorn:

```bash
poetry install
MLNODE_BACKENDS=http://mlnode1:8080,http://mlnode2:8080 \
    uvicorn load_balancer.app:app --reload --host 0.0.0.0 --port 8080
```

## Behaviour

- `/api/v1/state` responds with the aggregated state and a breakdown per backend.
- `/health` returns `200` if at least one backend is healthy and running inference.
- `/v1/*` requests are proxied to the least busy healthy backend.
- Any other paths are forwarded to the first configured backend for backward
  compatibility.
