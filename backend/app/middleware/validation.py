from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any, Optional, List, Union
import json
import logging
import re
from datetime import datetime
from pydantic import BaseModel, ValidationError
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

class ValidationConfig:
    """Configuration for input validation."""
    
    def __init__(self):
        # Maximum request size (in bytes)
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        
        # Maximum JSON depth
        self.max_json_depth = 10
        
        # Maximum array length
        self.max_array_length = 1000
        
        # Maximum string length
        self.max_string_length = 10000
        
        # Blocked patterns (potential security threats)
        self.blocked_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'\beval\s*\(',  # eval() calls
            r'\bexec\s*\(',  # exec() calls
            r'\b(union|select|insert|update|delete|drop|create|alter)\b',  # SQL keywords
            r'\.\./',  # Path traversal
            r'\\x[0-9a-fA-F]{2}',  # Hex encoded characters
            r'%[0-9a-fA-F]{2}',  # URL encoded characters (suspicious patterns)
        ]
        
        # Compiled regex patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.blocked_patterns]
        
        # Content type validation
        self.allowed_content_types = {
            'application/json',
            'application/x-www-form-urlencoded',
            'multipart/form-data',
            'text/plain'
        }
        
        # File upload validation
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_file_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.svg',  # Images
            '.pdf', '.doc', '.docx', '.txt',  # Documents
            '.csv', '.xlsx', '.json'  # Data files
        }

class SecurityValidator:
    """Security-focused input validator."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate_string(self, value: str, field_name: str = "field") -> str:
        """Validate string input for security threats."""
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        
        # Check length
        if len(value) > self.config.max_string_length:
            raise ValueError(f"{field_name} exceeds maximum length of {self.config.max_string_length}")
        
        # Check for blocked patterns
        for pattern in self.config.compiled_patterns:
            if pattern.search(value):
                logger.warning(f"Blocked pattern detected in {field_name}: {pattern.pattern}")
                raise ValueError(f"{field_name} contains potentially malicious content")
        
        return value
    
    def validate_json_structure(self, data: Any, depth: int = 0) -> Any:
        """Recursively validate JSON structure."""
        if depth > self.config.max_json_depth:
            raise ValueError(f"JSON depth exceeds maximum of {self.config.max_json_depth}")
        
        if isinstance(data, dict):
            if len(data) > 1000:  # Prevent DoS via large objects
                raise ValueError("JSON object has too many keys")
            
            validated = {}
            for key, value in data.items():
                # Validate key
                if not isinstance(key, str):
                    raise ValueError("JSON keys must be strings")
                
                validated_key = self.validate_string(key, f"key '{key}'")
                validated[validated_key] = self.validate_json_structure(value, depth + 1)
            
            return validated
        
        elif isinstance(data, list):
            if len(data) > self.config.max_array_length:
                raise ValueError(f"Array length exceeds maximum of {self.config.max_array_length}")
            
            return [self.validate_json_structure(item, depth + 1) for item in data]
        
        elif isinstance(data, str):
            return self.validate_string(data)
        
        elif isinstance(data, (int, float, bool)) or data is None:
            return data
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def validate_file_upload(self, filename: str, content_type: str, size: int) -> bool:
        """Validate file upload parameters."""
        # Check file size
        if size > self.config.max_file_size:
            raise ValueError(f"File size exceeds maximum of {self.config.max_file_size} bytes")
        
        # Check file extension
        if filename:
            extension = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
            if extension not in self.config.allowed_file_extensions:
                raise ValueError(f"File extension '{extension}' not allowed")
        
        # Validate filename for security
        if filename:
            self.validate_string(filename, "filename")
            
            # Additional filename checks
            if '..' in filename or '/' in filename or '\\' in filename:
                raise ValueError("Filename contains invalid path characters")
        
        return True

class InputSanitizer:
    """Input sanitization utilities."""
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return value
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Normalize whitespace
        value = ' '.join(value.split())
        
        # Remove control characters except newlines and tabs
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\t')
        
        return value.strip()
    
    @staticmethod
    def sanitize_html(value: str) -> str:
        """Basic HTML sanitization (remove tags)."""
        if not isinstance(value, str):
            return value
        
        # Remove HTML tags
        html_pattern = re.compile(r'<[^>]+>')
        value = html_pattern.sub('', value)
        
        # Decode HTML entities
        html_entities = {
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&#x27;': "'",
            '&#x2F;': '/',
        }
        
        for entity, char in html_entities.items():
            value = value.replace(entity, char)
        
        return value
    
    @staticmethod
    def sanitize_sql(value: str) -> str:
        """Basic SQL injection prevention."""
        if not isinstance(value, str):
            return value
        
        # Remove or escape dangerous SQL characters
        dangerous_chars = {
            "'": "''",  # Escape single quotes
            '"': '""',  # Escape double quotes
            ';': '',    # Remove semicolons
            '--': '',   # Remove SQL comments
            '/*': '',   # Remove SQL comments
            '*/': '',   # Remove SQL comments
        }
        
        for char, replacement in dangerous_chars.items():
            value = value.replace(char, replacement)
        
        return value

class ValidationMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for input validation and sanitization."""
    
    def __init__(self, app, enable_sanitization: bool = True, strict_mode: bool = False):
        super().__init__(app)
        self.config = ValidationConfig()
        self.validator = SecurityValidator(self.config)
        self.sanitizer = InputSanitizer()
        self.enable_sanitization = enable_sanitization
        self.strict_mode = strict_mode
    
    async def dispatch(self, request: Request, call_next):
        """Process request with validation."""
        try:
            # Skip validation for certain endpoints
            if self._should_skip_validation(request):
                return await call_next(request)
            
            # Validate request size
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > self.config.max_request_size:
                return self._create_error_response(
                    "Request too large",
                    f"Request size exceeds maximum of {self.config.max_request_size} bytes",
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                )
            
            # Validate content type
            content_type = request.headers.get('content-type', '').split(';')[0]
            if content_type and content_type not in self.config.allowed_content_types:
                return self._create_error_response(
                    "Invalid content type",
                    f"Content type '{content_type}' not allowed",
                    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
                )
            
            # Validate and sanitize request body
            if request.method in ['POST', 'PUT', 'PATCH']:
                request = await self._validate_request_body(request)
            
            # Validate query parameters
            request = await self._validate_query_params(request)
            
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            
            return response
            
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return self._create_error_response(
                "Validation failed",
                str(e),
                status.HTTP_422_UNPROCESSABLE_ENTITY
            )
        
        except ValueError as e:
            logger.warning(f"Input validation error: {e}")
            return self._create_error_response(
                "Invalid input",
                str(e),
                status.HTTP_400_BAD_REQUEST
            )
        
        except Exception as e:
            logger.error(f"Validation middleware error: {e}")
            if self.strict_mode:
                return self._create_error_response(
                    "Validation error",
                    "Request could not be validated",
                    status.HTTP_400_BAD_REQUEST
                )
            else:
                # Continue processing in non-strict mode
                return await call_next(request)
    
    async def _validate_request_body(self, request: Request) -> Request:
        """Validate and sanitize request body."""
        content_type = request.headers.get('content-type', '').split(';')[0]
        
        if content_type == 'application/json':
            body = await request.body()
            if body:
                try:
                    # Parse JSON
                    data = json.loads(body.decode('utf-8'))
                    
                    # Validate structure
                    validated_data = self.validator.validate_json_structure(data)
                    
                    # Sanitize if enabled
                    if self.enable_sanitization:
                        validated_data = self._sanitize_data(validated_data)
                    
                    # Replace request body with validated data
                    new_body = json.dumps(validated_data).encode('utf-8')
                    request._body = new_body
                    
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON: {e}")
        
        return request
    
    async def _validate_query_params(self, request: Request) -> Request:
        """Validate and sanitize query parameters."""
        if request.query_params:
            validated_params = {}
            
            for key, value in request.query_params.items():
                # Validate key and value
                validated_key = self.validator.validate_string(key, f"query parameter '{key}'")
                validated_value = self.validator.validate_string(value, f"query parameter value for '{key}'")
                
                # Sanitize if enabled
                if self.enable_sanitization:
                    validated_key = self.sanitizer.sanitize_string(validated_key)
                    validated_value = self.sanitizer.sanitize_string(validated_value)
                
                validated_params[validated_key] = validated_value
            
            # Update request query params
            request.scope["query_string"] = '&'.join(
                f"{key}={value}" for key, value in validated_params.items()
            ).encode('utf-8')
        
        return request
    
    def _sanitize_data(self, data: Any) -> Any:
        """Recursively sanitize data structure."""
        if isinstance(data, dict):
            return {
                self.sanitizer.sanitize_string(key): self._sanitize_data(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, str):
            return self.sanitizer.sanitize_string(data)
        else:
            return data
    
    def _should_skip_validation(self, request: Request) -> bool:
        """Check if validation should be skipped for this request."""
        skip_paths = [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static"
        ]
        
        return any(request.url.path.startswith(path) for path in skip_paths)
    
    def _create_error_response(self, error: str, message: str, status_code: int) -> JSONResponse:
        """Create standardized error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "status_code": status_code
            }
        )

# Decorator for function-level validation
def validate_input(schema: BaseModel = None, sanitize: bool = True):
    """Decorator for validating function inputs."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if schema:
                # Validate against Pydantic schema
                try:
                    validated_data = schema(**kwargs)
                    kwargs.update(validated_data.dict())
                except ValidationError as e:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Validation error: {e}"
                    )
            
            if sanitize:
                # Sanitize string inputs
                sanitizer = InputSanitizer()
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        kwargs[key] = sanitizer.sanitize_string(value)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Utility functions
def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password_strength(password: str) -> Dict[str, bool]:
    """Validate password strength."""
    checks = {
        'min_length': len(password) >= 8,
        'has_uppercase': any(c.isupper() for c in password),
        'has_lowercase': any(c.islower() for c in password),
        'has_digit': any(c.isdigit() for c in password),
        'has_special': any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password),
    }
    
    checks['is_strong'] = all(checks.values())
    return checks

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\|?*]', '', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
    
    return filename

def validate_json_schema(data: Dict, schema: Dict) -> bool:
    """Basic JSON schema validation."""
    # This is a simplified implementation
    # In production, use jsonschema library
    
    def validate_field(value, field_schema):
        field_type = field_schema.get('type')
        
        if field_type == 'string' and not isinstance(value, str):
            return False
        elif field_type == 'integer' and not isinstance(value, int):
            return False
        elif field_type == 'number' and not isinstance(value, (int, float)):
            return False
        elif field_type == 'boolean' and not isinstance(value, bool):
            return False
        elif field_type == 'array' and not isinstance(value, list):
            return False
        elif field_type == 'object' and not isinstance(value, dict):
            return False
        
        return True
    
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    # Check required fields
    for field in required:
        if field not in data:
            return False
    
    # Validate each field
    for field, value in data.items():
        if field in properties:
            if not validate_field(value, properties[field]):
                return False
    
    return True