from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Optional, Tuple
import time
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
import redis.asyncio as redis
import json
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class RateLimitExceeded(HTTPException):
    """Custom exception for rate limit exceeded."""
    def __init__(self, detail: str, retry_after: int = None):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)} if retry_after else None
        )

class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket."""
        async with self.lock:
            now = time.time()
            # Add tokens based on time elapsed
            time_passed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until enough tokens are available."""
        if self.tokens >= tokens:
            return 0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate

class SlidingWindowCounter:
    """Sliding window counter for rate limiting."""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # seconds
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def is_allowed(self) -> Tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, retry_after)."""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True, 0
            
            # Calculate retry after time
            oldest_request = self.requests[0]
            retry_after = int(oldest_request + self.window_size - now) + 1
            return False, retry_after

class InMemoryRateLimiter:
    """In-memory rate limiter using token bucket and sliding window."""
    
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.windows: Dict[str, SlidingWindowCounter] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window: int, 
        algorithm: str = "sliding_window"
    ) -> Tuple[bool, int]:
        """Check if request is allowed."""
        await self._cleanup_old_entries()
        
        if algorithm == "token_bucket":
            return await self._check_token_bucket(key, limit, window)
        else:
            return await self._check_sliding_window(key, limit, window)
    
    async def _check_token_bucket(self, key: str, capacity: int, refill_rate: float) -> Tuple[bool, int]:
        """Check using token bucket algorithm."""
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(capacity, refill_rate)
        
        bucket = self.buckets[key]
        allowed = await bucket.consume()
        retry_after = int(bucket.time_until_available()) if not allowed else 0
        
        return allowed, retry_after
    
    async def _check_sliding_window(self, key: str, max_requests: int, window_size: int) -> Tuple[bool, int]:
        """Check using sliding window algorithm."""
        if key not in self.windows:
            self.windows[key] = SlidingWindowCounter(window_size, max_requests)
        
        return await self.windows[key].is_allowed()
    
    async def _cleanup_old_entries(self):
        """Clean up old entries to prevent memory leaks."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # Remove buckets that haven't been used recently
        keys_to_remove = []
        for key, bucket in self.buckets.items():
            if now - bucket.last_refill > self.cleanup_interval:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.buckets[key]
        
        # Clean up sliding windows (they clean themselves during is_allowed calls)
        self.last_cleanup = now

class RedisRateLimiter:
    """Redis-based rate limiter for distributed systems."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory rate limiting.")
            self.redis_client = None
    
    async def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window: int, 
        algorithm: str = "sliding_window"
    ) -> Tuple[bool, int]:
        """Check if request is allowed using Redis."""
        if not self.redis_client:
            return True, 0  # Allow if Redis is not available
        
        try:
            if algorithm == "sliding_window":
                return await self._sliding_window_redis(key, limit, window)
            else:
                return await self._token_bucket_redis(key, limit, window)
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            return True, 0  # Allow on error
    
    async def _sliding_window_redis(self, key: str, limit: int, window: int) -> Tuple[bool, int]:
        """Sliding window implementation using Redis."""
        now = time.time()
        pipeline = self.redis_client.pipeline()
        
        # Remove old entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(now): now})
        
        # Set expiration
        pipeline.expire(key, window + 1)
        
        results = await pipeline.execute()
        current_requests = results[1]
        
        if current_requests < limit:
            return True, 0
        
        # Calculate retry after
        oldest_requests = await self.redis_client.zrange(key, 0, 0, withscores=True)
        if oldest_requests:
            oldest_time = oldest_requests[0][1]
            retry_after = int(oldest_time + window - now) + 1
            return False, retry_after
        
        return False, window
    
    async def _token_bucket_redis(self, key: str, capacity: int, refill_rate: float) -> Tuple[bool, int]:
        """Token bucket implementation using Redis Lua script."""
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Refill tokens
        local time_passed = now - last_refill
        tokens = math.min(capacity, tokens + time_passed * refill_rate)
        
        local allowed = 0
        local retry_after = 0
        
        if tokens >= tokens_requested then
            tokens = tokens - tokens_requested
            allowed = 1
        else
            retry_after = math.ceil((tokens_requested - tokens) / refill_rate)
        end
        
        -- Update bucket
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)  -- 1 hour expiration
        
        return {allowed, retry_after}
        """
        
        result = await self.redis_client.eval(
            lua_script, 
            1, 
            key, 
            capacity, 
            refill_rate, 
            1,  # tokens requested
            time.time()
        )
        
        return bool(result[0]), int(result[1])

class RateLimitConfig:
    """Rate limit configuration."""
    
    def __init__(self):
        # Default limits (requests per minute)
        self.default_limit = 60
        self.default_window = 60
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/v1/auth/login": {"limit": 5, "window": 300},  # 5 per 5 minutes
            "/api/v1/auth/register": {"limit": 3, "window": 300},  # 3 per 5 minutes
            "/api/v1/players/search": {"limit": 100, "window": 60},  # 100 per minute
            "/api/v1/players/bulk": {"limit": 5, "window": 300},  # 5 per 5 minutes
            "/api/v1/ml/predict": {"limit": 20, "window": 60},  # 20 per minute
        }
        
        # User tier limits
        self.tier_limits = {
            "free": {"limit": 100, "window": 3600},  # 100 per hour
            "premium": {"limit": 1000, "window": 3600},  # 1000 per hour
            "enterprise": {"limit": 10000, "window": 3600},  # 10000 per hour
        }
    
    def get_limit(self, endpoint: str, user_tier: str = "free") -> Tuple[int, int]:
        """Get rate limit for endpoint and user tier."""
        # Check endpoint-specific limits first
        if endpoint in self.endpoint_limits:
            config = self.endpoint_limits[endpoint]
            return config["limit"], config["window"]
        
        # Check user tier limits
        if user_tier in self.tier_limits:
            config = self.tier_limits[user_tier]
            return config["limit"], config["window"]
        
        # Return default
        return self.default_limit, self.default_window

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, app, redis_url: str = None, use_redis: bool = True):
        super().__init__(app)
        self.config = RateLimitConfig()
        self.use_redis = use_redis and redis_url
        
        if self.use_redis:
            self.limiter = RedisRateLimiter(redis_url)
        else:
            self.limiter = InMemoryRateLimiter()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Connect to Redis if needed
        if self.use_redis and hasattr(self.limiter, 'redis_client') and not self.limiter.redis_client:
            await self.limiter.connect()
        
        # Skip rate limiting for health checks and static files
        if self._should_skip_rate_limiting(request):
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get user tier (from JWT token if available)
        user_tier = self._get_user_tier(request)
        
        # Get rate limit for this endpoint
        limit, window = self.config.get_limit(request.url.path, user_tier)
        
        # Check rate limit
        key = f"rate_limit:{client_id}:{request.url.path}"
        allowed, retry_after = await self.limiter.is_allowed(key, limit, window)
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_id} on {request.url.path}. "
                f"Limit: {limit}/{window}s, Retry after: {retry_after}s"
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {limit} requests per {window} seconds.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Window"] = str(window)
        
        return response
    
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Check if rate limiting should be skipped for this request."""
        skip_paths = [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static"
        ]
        
        return any(request.url.path.startswith(path) for path in skip_paths)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from JWT token
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    def _get_user_tier(self, request: Request) -> str:
        """Get user tier from request (JWT token)."""
        # This would be extracted from JWT token in a real implementation
        return getattr(request.state, 'user_tier', 'free')

# Decorator for function-level rate limiting
def rate_limit(limit: int, window: int, key_func=None):
    """Decorator for rate limiting specific functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would need to be implemented based on your specific needs
            # For now, it's a placeholder
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Utility functions
async def get_rate_limit_status(client_id: str, endpoint: str, limiter) -> Dict:
    """Get current rate limit status for a client."""
    config = RateLimitConfig()
    limit, window = config.get_limit(endpoint)
    
    key = f"rate_limit:{client_id}:{endpoint}"
    
    if isinstance(limiter, RedisRateLimiter) and limiter.redis_client:
        try:
            # Get current count from Redis
            now = time.time()
            count = await limiter.redis_client.zcount(key, now - window, now)
            remaining = max(0, limit - count)
            
            return {
                "limit": limit,
                "remaining": remaining,
                "reset_time": int(now + window),
                "window": window
            }
        except Exception:
            pass
    
    # Fallback for in-memory limiter
    return {
        "limit": limit,
        "remaining": limit,  # Can't easily determine for in-memory
        "reset_time": int(time.time() + window),
        "window": window
    }