"""
Prometheus observability integration for FastAPI.
Tracks request count and latency per endpoint.
"""

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Request, Response
from starlette.responses import Response as StarletteResponse
import time

# Metrics definitions
REQUEST_COUNT = Counter(
    'http_requests_total', 'Total HTTP requests',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'http_request_latency_seconds', 'HTTP request latency (seconds)',
    ['endpoint']
)

async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(process_time)
    return response

def setup_prometheus(app: FastAPI):
    @app.get("/metrics")
    async def metrics():
        # Prometheus expects a plain text response with the right content type
        return StarletteResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    app.middleware('http')(prometheus_middleware)
