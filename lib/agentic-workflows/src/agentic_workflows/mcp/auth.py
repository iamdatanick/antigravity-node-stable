"""MCP Authentication Module.

Provides OAuth 2.1 PKCE, API Key, and Bearer token authentication for MCP servers.

Usage:
    from agentic_workflows.mcp.auth import OAuth21Client, PKCEChallenge

    oauth = OAuth21Client(
        client_id="your-client-id",
        authorization_url="https://auth.example.com/authorize",
        token_url="https://auth.example.com/token",
    )
    auth_url = await oauth.get_authorization_url(redirect_uri="http://localhost:3000/callback")
    tokens = await oauth.exchange_code(code, redirect_uri)
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class AuthType(Enum):
    """Authentication types for MCP servers."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH21 = "oauth21"


@dataclass
class PKCEChallenge:
    """PKCE (Proof Key for Code Exchange) challenge for OAuth 2.1.

    OAuth 2.1 requires PKCE for all authorization code grants.

    Example:
        challenge = PKCEChallenge.generate()
        # Use challenge.code_challenge in authorization request
        # Use challenge.code_verifier in token exchange
    """

    code_verifier: str
    code_challenge: str
    code_challenge_method: str = "S256"

    @classmethod
    def generate(cls) -> "PKCEChallenge":
        """Generate a new PKCE challenge.

        Returns:
            PKCEChallenge with verifier and challenge.
        """
        # Generate code verifier (43-128 characters)
        code_verifier = secrets.token_urlsafe(64)[:96]

        # Generate code challenge using S256
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")

        return cls(
            code_verifier=code_verifier,
            code_challenge=code_challenge,
            code_challenge_method="S256",
        )


@dataclass
class TokenSet:
    """OAuth 2.1 token set."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    id_token: Optional[str] = None

    # Computed fields
    issued_at: float = field(default_factory=time.time)

    @property
    def expires_at(self) -> Optional[float]:
        """Get expiration timestamp."""
        if self.expires_in is None:
            return None
        return self.issued_at + self.expires_in

    @property
    def is_expired(self) -> bool:
        """Check if token is expired (with 60s buffer)."""
        if self.expires_at is None:
            return False
        return time.time() >= (self.expires_at - 60)

    def to_header(self) -> str:
        """Get Authorization header value."""
        return f"{self.token_type} {self.access_token}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
            "id_token": self.id_token,
            "issued_at": self.issued_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenSet":
        """Create from dictionary."""
        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in"),
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
            id_token=data.get("id_token"),
            issued_at=data.get("issued_at", time.time()),
        )


@dataclass
class OAuth21Config:
    """OAuth 2.1 client configuration."""

    client_id: str
    authorization_url: str
    token_url: str
    client_secret: Optional[str] = None  # Not required for public clients
    scopes: List[str] = field(default_factory=list)
    audience: Optional[str] = None

    # PKCE is REQUIRED in OAuth 2.1
    use_pkce: bool = True

    # Discovery
    issuer: Optional[str] = None
    jwks_uri: Optional[str] = None


class OAuth21Client:
    """OAuth 2.1 client with PKCE support.

    Implements OAuth 2.1 authorization code flow with PKCE.
    PKCE is mandatory in OAuth 2.1 for all clients.

    Example:
        client = OAuth21Client(
            client_id="mcp-client",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )

        # Step 1: Get authorization URL
        auth_url = await client.get_authorization_url(
            redirect_uri="http://localhost:3000/callback",
            scopes=["mcp:read", "mcp:write"],
        )

        # Step 2: User authorizes and returns with code
        # Step 3: Exchange code for tokens
        tokens = await client.exchange_code(
            code="auth_code_from_redirect",
            redirect_uri="http://localhost:3000/callback",
        )

        # Step 4: Use access token
        headers = {"Authorization": tokens.to_header()}

        # Step 5: Refresh when expired
        if tokens.is_expired:
            tokens = await client.refresh_tokens(tokens.refresh_token)
    """

    def __init__(
        self,
        client_id: str,
        authorization_url: str,
        token_url: str,
        client_secret: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        audience: Optional[str] = None,
    ):
        """Initialize OAuth 2.1 client.

        Args:
            client_id: OAuth client ID.
            authorization_url: Authorization endpoint URL.
            token_url: Token endpoint URL.
            client_secret: Client secret (optional for public clients).
            scopes: Default scopes to request.
            audience: API audience (for Auth0-style providers).
        """
        self.config = OAuth21Config(
            client_id=client_id,
            authorization_url=authorization_url,
            token_url=token_url,
            client_secret=client_secret,
            scopes=scopes or [],
            audience=audience,
        )
        self._current_pkce: Optional[PKCEChallenge] = None
        self._state: Optional[str] = None

    async def get_authorization_url(
        self,
        redirect_uri: str,
        scopes: Optional[List[str]] = None,
        state: Optional[str] = None,
        extra_params: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate authorization URL with PKCE challenge.

        Args:
            redirect_uri: Callback URL after authorization.
            scopes: Scopes to request (overrides default).
            state: State parameter for CSRF protection.
            extra_params: Additional URL parameters.

        Returns:
            Authorization URL to redirect user to.
        """
        # Generate PKCE challenge
        self._current_pkce = PKCEChallenge.generate()

        # Generate state if not provided
        self._state = state or secrets.token_urlsafe(32)

        # Build params
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": self._state,
            "code_challenge": self._current_pkce.code_challenge,
            "code_challenge_method": self._current_pkce.code_challenge_method,
        }

        # Add scopes
        request_scopes = scopes or self.config.scopes
        if request_scopes:
            params["scope"] = " ".join(request_scopes)

        # Add audience if configured
        if self.config.audience:
            params["audience"] = self.config.audience

        # Add extra params
        if extra_params:
            params.update(extra_params)

        # Build URL
        return f"{self.config.authorization_url}?{urlencode(params)}"

    async def exchange_code(
        self,
        code: str,
        redirect_uri: str,
        state: Optional[str] = None,
    ) -> TokenSet:
        """Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback.
            redirect_uri: Same redirect URI used in authorization.
            state: State to verify (optional, recommended).

        Returns:
            TokenSet with access token and optionally refresh token.

        Raises:
            ValueError: If PKCE not initialized or state mismatch.
            RuntimeError: If token exchange fails.
        """
        if not self._current_pkce:
            raise ValueError("PKCE challenge not initialized. Call get_authorization_url first.")

        if state and self._state and state != self._state:
            raise ValueError("State mismatch - possible CSRF attack")

        try:
            import httpx

            data = {
                "grant_type": "authorization_code",
                "client_id": self.config.client_id,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": self._current_pkce.code_verifier,
            }

            # Add client secret if configured
            if self.config.client_secret:
                data["client_secret"] = self.config.client_secret

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                token_data = response.json()

            # Clear PKCE after successful exchange
            self._current_pkce = None
            self._state = None

            return TokenSet.from_dict(token_data)

        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            raise RuntimeError(f"Token exchange failed: {e}")

    async def refresh_tokens(self, refresh_token: str) -> TokenSet:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from previous token set.

        Returns:
            New TokenSet with refreshed access token.

        Raises:
            RuntimeError: If refresh fails.
        """
        try:
            import httpx

            data = {
                "grant_type": "refresh_token",
                "client_id": self.config.client_id,
                "refresh_token": refresh_token,
            }

            if self.config.client_secret:
                data["client_secret"] = self.config.client_secret

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                token_data = response.json()

            return TokenSet.from_dict(token_data)

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise RuntimeError(f"Token refresh failed: {e}")


class APIKeyAuth:
    """API Key authentication for MCP servers.

    Example:
        auth = APIKeyAuth(api_key="sk-xxx", header_name="X-API-Key")
        headers = auth.get_headers()
    """

    def __init__(
        self,
        api_key: str,
        header_name: str = "Authorization",
        prefix: str = "Bearer",
    ):
        """Initialize API key auth.

        Args:
            api_key: The API key.
            header_name: Header name to use.
            prefix: Prefix for the header value (e.g., "Bearer", "Api-Key").
        """
        self.api_key = api_key
        self.header_name = header_name
        self.prefix = prefix

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers.

        Returns:
            Dict with authentication header.
        """
        if self.prefix:
            return {self.header_name: f"{self.prefix} {self.api_key}"}
        return {self.header_name: self.api_key}


class BasicAuth:
    """HTTP Basic authentication.

    Example:
        auth = BasicAuth(username="user", password="pass")
        headers = auth.get_headers()
    """

    def __init__(self, username: str, password: str):
        """Initialize basic auth.

        Args:
            username: Username.
            password: Password.
        """
        self.username = username
        self.password = password

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers.

        Returns:
            Dict with Basic auth header.
        """
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}


@dataclass
class AuthConfig:
    """Unified authentication configuration.

    Example:
        # OAuth 2.1
        config = AuthConfig(
            auth_type=AuthType.OAUTH21,
            oauth_config={
                "client_id": "xxx",
                "authorization_url": "https://...",
                "token_url": "https://...",
            }
        )

        # API Key
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            api_key="sk-xxx",
        )
    """

    auth_type: AuthType = AuthType.NONE
    api_key: Optional[str] = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    username: Optional[str] = None
    password: Optional[str] = None
    oauth_config: Optional[Dict[str, Any]] = None
    tokens: Optional[TokenSet] = None

    def get_authenticator(self) -> Optional[Any]:
        """Get the appropriate authenticator instance.

        Returns:
            Authenticator instance or None.
        """
        if self.auth_type == AuthType.API_KEY and self.api_key:
            return APIKeyAuth(
                api_key=self.api_key,
                header_name=self.api_key_header,
                prefix=self.api_key_prefix,
            )
        elif self.auth_type == AuthType.BASIC and self.username and self.password:
            return BasicAuth(username=self.username, password=self.password)
        elif self.auth_type == AuthType.OAUTH21 and self.oauth_config:
            return OAuth21Client(**self.oauth_config)
        return None


__all__ = [
    "AuthType",
    "PKCEChallenge",
    "TokenSet",
    "OAuth21Config",
    "OAuth21Client",
    "APIKeyAuth",
    "BasicAuth",
    "AuthConfig",
]
