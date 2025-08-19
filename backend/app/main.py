# backend/app/main.py
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import time
import os
from .api.v1.api import api_router
from .api.v1.auth import router as auth_router
from .core.config import get_settings
from .core.rate_limiter import create_rate_limit_middleware
from .core.db import engine
from .models.users import Base as UserBase
from .models.players import Base as PlayerBase
from .middleware.rate_limiting import RateLimitMiddleware
from .middleware.validation import ValidationMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Create database tables
try:
    UserBase.metadata.create_all(bind=engine)
    PlayerBase.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Error creating database tables: {e}")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Football Fouls Analytics API",
        description="API for analyzing football player fouls and disciplinary actions",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middlewares (order matters - first added is executed last)
    
    # Rate limiting middleware
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    use_redis = os.getenv("USE_REDIS_RATE_LIMITING", "true").lower() == "true"
    app.add_middleware(
        RateLimitMiddleware,
        redis_url=redis_url,
        use_redis=use_redis
    )
    
    # Input validation middleware
    enable_sanitization = os.getenv("ENABLE_INPUT_SANITIZATION", "true").lower() == "true"
    strict_validation = os.getenv("STRICT_VALIDATION_MODE", "false").lower() == "true"
    app.add_middleware(
        ValidationMiddleware,
        enable_sanitization=enable_sanitization,
        strict_mode=strict_validation
    )
    
    # Include routers
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
    app.include_router(api_router, prefix="/api/v1")
    
    return app

app = create_app()

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "status_code": 422,
            "timestamp": time.time()
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {
        "message": "Football Fouls Analytics API", 
        "version": settings.api_version,
        "status": "operational",
        "docs": "/docs" if settings.debug else "disabled",
        "timestamp": time.time()
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.api_version
    }

@app.get("/api/v1")
def api_info():
    return {
        "message": "Football Fouls Analytics API v1",
        "version": settings.api_version,
        "endpoints": {
            "auth": "/api/v1/auth",
            "players": "/api/v1/players",
            "predictions": "/api/v1/predictions",
            "analytics": "/api/v1/analytics"
        }
    }

@app.get("/version")
def version():
    # sanitize: não vaza senha
    redacted = settings.database_url
    if "@" in redacted and "://" in redacted:
        scheme_and_user, host_part = redacted.split("://", 1)
        if "@" in host_part:
            userinfo, hostrest = host_part.split("@", 1)
            if ":" in userinfo:
                user, _pwd = userinfo.split(":", 1)
                redacted = f"{scheme_and_user}://{user}:***@{hostrest}"
    return {"database_url": redacted}