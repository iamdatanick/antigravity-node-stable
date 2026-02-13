"""JWT validation middleware for Keycloak integration."""

import asyncio
import logging
import os

import httpx
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

logger = logging.getLogger("antigravity.auth")

KEYCLOAK_URL = os.environ.get("KEYCLOAK_URL", "http://keycloak:8080")
KEYCLOAK_REALM = os.environ.get("KEYCLOAK_REALM", "antigravity")
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() == "true"

_jwks_cache = None
_jwks_lock = asyncio.Lock()
security = HTTPBearer(auto_error=False)


async def get_jwks():
    global _jwks_cache
    if _jwks_cache is None:
        async with _jwks_lock:
            # Double-check locking pattern
            if _jwks_cache is None:
                jwks_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.get(jwks_url)
                        resp.raise_for_status()
                        _jwks_cache = resp.json()
                except Exception as e:
                    logger.warning(f"Failed to fetch JWKS: {e}", exc_info=True)
                    return None
    return _jwks_cache


async def validate_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    if not AUTH_ENABLED:
        return {"sub": "anonymous", "roles": ["readonly"]}

    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    token = credentials.credentials
    jwks = await get_jwks()
    if jwks is None:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        key = None
        for k in jwks.get("keys", []):
            if k["kid"] == kid:
                key = k
                break

        if key is None:
            async with _jwks_lock:
                _jwks_cache = None  # Force refresh on next request
            raise HTTPException(status_code=401, detail="Invalid token key ID")

        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=os.environ.get("KEYCLOAK_CLIENT_ID", "antigravity-api"),
            issuer=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}",
        )
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
