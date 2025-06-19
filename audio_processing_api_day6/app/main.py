"""
Production-ready FastAPI application with comprehensive middleware,
error handling, and monitoring capabilities.
"""
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.encoders import jsonable_encoder

from app.core.config import get_settings
from app.routes import api_v1_endpoints, ws_endpoints, health
from app.schemas.output_schemas import ErrorResponse

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management."""
    logger.info("Starting Audio Processing API...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    yield
    logger.info("Shutting down Audio Processing API...")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="""
    Advanced Audio Processing API with ML-powered features.

    * ASR: Multi-language speech recognition with Whisper and Vosk
    * Translation: Neural machine translation (MarianMT, OpenNMT)
    * TTS: Text-to-speech synthesis with voice cloning (YourTTS, ElevenLabs)
    * Real-time: WebSocket support for streaming audio
    * Voice Cloning: Preserve speaker characteristics
    * Analytics: Comprehensive reporting and similarity scoring
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS and GZip middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add Prometheus Instrumentator BEFORE startup
if getattr(settings, "ENABLE_METRICS", False):
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
    logger.info("Metrics endpoint enabled at /metrics")

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware for request logging and performance monitoring."""
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} processed in {process_time:.3f}s")
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with structured error responses."""
    error_response = ErrorResponse(
        error=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
        request_id=getattr(request.state, 'request_id', None)
    )
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail} for {request.method} {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(error_response)
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed field information."""
    error_details = [
        {"field": ".".join(str(x) for x in error["loc"]), "message": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    error_response = ErrorResponse(
        error="VALIDATION_ERROR",
        message="Request validation failed",
        details={"validation_errors": error_details}
    )
    logger.warning(f"Validation error for {request.method} {request.url.path}: {error_details}")
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(error_response)
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    error_response = ErrorResponse(
        error="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        details={"error_type": type(exc).__name__} if settings.DEBUG else None
    )
    logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)} for {request.method} {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(error_response)
    )

# Include routers
app.include_router(api_v1_endpoints.router, prefix="/api/v1", tags=["Audio Processing"])
app.include_router(ws_endpoints.router, prefix="/api/v1", tags=["Real-time Processing"])
app.include_router(health.router, prefix="/api/v1", tags=["System Health"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with service information."""
    return {
        "message": "Welcome to the Advanced Audio Processing API",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/docs",
        "health": "/api/v1/health",
        "metrics": "/metrics" if settings.ENABLE_METRICS else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
