"""
Production-grade security system with JWT authentication, refresh tokens,
role-based access control, and comprehensive audit logging.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

# Load configuration
settings = get_settings()

# FastAPI security dependency
security = HTTPBearer()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Logger for auth events
auth_logger = logging.getLogger("auth_audit")
if not auth_logger.handlers:
    handler = logging.FileHandler("auth_audit.log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    auth_logger.addHandler(handler)
auth_logger.setLevel(logging.INFO)

# In-memory blacklist (replace with Redis/DB in production)
blacklisted_tokens = set()

class SecurityService:
    """Advanced security service with authentication and auditing."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a plain password."""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify plain password against hash."""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Generate access token with optional expiry."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Generate refresh token with longer expiry."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS))
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
        """Decode and validate JWT token, checking type and blacklist."""
        try:
            if token in blacklisted_tokens:
                raise HTTPException(status_code=401, detail="Token has been revoked")

            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])

            if payload.get("type") != token_type:
                raise HTTPException(status_code=401, detail=f"Invalid {token_type} token")

            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    def blacklist_token(token: str):
        """Add a JWT token to the blacklist."""
        blacklisted_tokens.add(token)

    @staticmethod
    def log_auth_attempt(username: str, success: bool, ip_address: str = "unknown", reason: str = ""):
        """Log authentication attempts for auditing and traceability."""
        msg = f"User: {username} | IP: {ip_address} | Success: {success} | Reason: {reason}"
        auth_logger.info(msg)

# FastAPI dependency to extract and verify the current user
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    return SecurityService.verify_token(credentials.credentials)

# FastAPI dependency factory for role-based access
def require_role(required_role: str) -> Callable:
    def role_checker(credentials: HTTPAuthorizationCredentials = Depends(security)):
        payload = SecurityService.verify_token(credentials.credentials)
        user_role = payload.get("role", "user")
        if user_role != required_role and user_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges"
            )
        return payload
    return role_checker
