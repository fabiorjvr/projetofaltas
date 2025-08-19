import time
import asyncio
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis
import json
from datetime import datetime, timedelta
from ..core.config import settings

class RateLimiter:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.memory_store: Dict[str, Dict] = {}
        
        if redis_url or settings.REDIS_URL:
            try:
                import redis
                self.redis_client = redis.from_url(
                    redis_url or settings.REDIS_URL,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
            except Exception as e:
                print(f"Redis connection failed, using memory store: {e}")
                self.redis_client = None
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try to get user ID from JWT token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from .auth import auth_service
                token = auth_header.split(" ")[1]
                token_data = auth_service.verify_token(token)
                return f"user:{token_data.user_id}"
            except:
                pass
        
        # Fallback to IP address
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        client_host = getattr(request.client, "host", "unknown")
        return f"ip:{client_host}"
    
    def _get_key(self, client_id: str, endpoint: str, window: str) -> str:
        """Generate Redis/memory key."""
        return f"rate_limit:{client_id}:{endpoint}:{window}"
    
    async def _increment_redis(self, key: str, window_seconds: int) -> tuple[int, int]:
        """Increment counter in Redis with sliding window."""
        pipe = self.redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds)
        results = pipe.execute()
        
        current_count = results[0]
        ttl = self.redis_client.ttl(key)
        
        return current_count, ttl
    
    def _increment_memory(self, key: str, window_seconds: int) -> tuple[int, int]:
        """Increment counter in memory with sliding window."""
        now = time.time()
        
        if key not in self.memory_store:
            self.memory_store[key] = {
                "count": 0,
                "window_start": now,
                "requests": []
            }
        
        store_data = self.memory_store[key]
        
        # Clean old requests outside the window
        cutoff_time = now - window_seconds
        store_data["requests"] = [
            req_time for req_time in store_data["requests"]
            if req_time > cutoff_time
        ]
        
        # Add current request
        store_data["requests"].append(now)
        store_data["count"] = len(store_data["requests"])
        
        # Calculate TTL
        oldest_request = min(store_data["requests"]) if store_data["requests"] else now
        ttl = int(window_seconds - (now - oldest_request))
        
        return store_data["count"], max(ttl, 0)
    
    async def check_rate_limit(
        self,
        request: Request,
        max_requests: int,
        window_seconds: int,
        endpoint: Optional[str] = None
    ) -> tuple[bool, dict]:
        """Check if request is within rate limit."""
        client_id = self._get_client_id(request)
        endpoint = endpoint or str(request.url.path)
        
        # Create time window identifier
        window_start = int(time.time() // window_seconds) * window_seconds
        key = self._get_key(client_id, endpoint, str(window_start))
        
        try:
            if self.redis_client:
                current_count, ttl = await self._increment_redis(key, window_seconds)
            else:
                current_count, ttl = self._increment_memory(key, window_seconds)
            
            # Rate limit info
            rate_limit_info = {
                "limit": max_requests,
                "remaining": max(0, max_requests - current_count),
                "reset": int(time.time() + ttl),
                "retry_after": ttl if current_count > max_requests else 0
            }
            
            # Check if limit exceeded
            if current_count > max_requests:
                return False, rate_limit_info
            
            return True, rate_limit_info
            
        except Exception as e:
            print(f"Rate limiting error: {e}")
            # On error, allow the request
            return True, {
                "limit": max_requests,
                "remaining": max_requests,
                "reset": int(time.time() + window_seconds),
                "retry_after": 0
            }

# Global rate limiter instance
rate_limiter = RateLimiter()

# Rate limiting configurations
RATE_LIMIT_CONFIGS = {
    "default": {"max_requests": 100, "window_seconds": 3600},  # 100 per hour
    "auth": {"max_requests": 10, "window_seconds": 900},      # 10 per 15 min
    "api": {"max_requests": 1000, "window_seconds": 3600},    # 1000 per hour
    "ml": {"max_requests": 50, "window_seconds": 3600},       # 50 per hour
    "upload": {"max_requests": 20, "window_seconds": 3600},   # 20 per hour
}

def create_rate_limit_middleware(config_name: str = "default"):
    """Create rate limiting middleware for specific endpoints."""
    config = RATE_LIMIT_CONFIGS.get(config_name, RATE_LIMIT_CONFIGS["default"])
    
    async def rate_limit_middleware(request: Request, call_next):
        # Check rate limit
        allowed, rate_info = await rate_limiter.check_rate_limit(
            request,
            config["max_requests"],
            config["window_seconds"]
        )
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "rate_limit": rate_info
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["retry_after"])
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        
        return response
    
    return rate_limit_middleware

# Decorator for route-specific rate limiting
def rate_limit(config_name: str = "default"):
    """Decorator for applying rate limiting to specific routes."""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            config = RATE_LIMIT_CONFIGS.get(config_name, RATE_LIMIT_CONFIGS["default"])
            
            allowed, rate_info = await rate_limiter.check_rate_limit(
                request,
                config["max_requests"],
                config["window_seconds"]
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(rate_info["limit"]),
                        "X-RateLimit-Remaining": str(rate_info["remaining"]),
                        "X-RateLimit-Reset": str(rate_info["reset"]),
                        "Retry-After": str(rate_info["retry_after"])
                    }
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator