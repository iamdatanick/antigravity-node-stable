"""MCP Server Registry — manage third-party MCP servers with OAuth support.

Allows registering external MCP servers (databases, SaaS APIs, etc.) and
handles OAuth client_credentials flow for authenticated connections.

Usage:
    POST /mcp/servers   — register a new server
    GET  /mcp/servers   — list all registered servers
    POST /mcp/servers/{name}/tools — call a tool on a registered server
"""

import logging
import threading
import time

import httpx

logger = logging.getLogger("antigravity.mcp_registry")


class OAuthTokenManager:
    """Manages OAuth 2.0 tokens for MCP server connections.

    Supports client_credentials grant type. Tokens are cached until
    they expire (with a 60s buffer).
    """

    def __init__(self):
        self._tokens: dict[str, dict] = {}
        self._lock = threading.Lock()

    async def get_token(self, server_name: str, oauth_config: dict) -> str:
        """Get a valid access token, refreshing if expired."""
        with self._lock:
            cached = self._tokens.get(server_name)
            if cached and cached["expires_at"] > time.time():
                return cached["access_token"]

        # Fetch new token
        token_url = oauth_config["token_url"]
        client_id = oauth_config["client_id"]
        client_secret = oauth_config["client_secret"]
        scopes = oauth_config.get("scopes", [])

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": " ".join(scopes) if scopes else "",
                },
            )
            resp.raise_for_status()
            token_data = resp.json()

        access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)

        with self._lock:
            self._tokens[server_name] = {
                "access_token": access_token,
                "expires_at": time.time() + expires_in - 60,
            }

        logger.info(f"OAuth token acquired for MCP server '{server_name}' (expires in {expires_in}s)")
        return access_token

    def clear(self, server_name: str):
        """Remove cached token for a server."""
        with self._lock:
            self._tokens.pop(server_name, None)


class MCPServerRegistry:
    """Registry for MCP server connections (built-in and third-party).

    Each registered server has:
        - name: unique identifier
        - url: SSE or HTTP endpoint
        - transport: "sse" or "http"
        - oauth_config: optional OAuth settings for authenticated access
        - description: human-readable description
    """

    def __init__(self):
        self.servers: dict[str, dict] = {}
        self._token_manager = OAuthTokenManager()
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        url: str,
        transport: str = "sse",
        oauth_config: dict | None = None,
        description: str = "",
    ):
        """Register a third-party MCP server."""
        with self._lock:
            self.servers[name] = {
                "url": url,
                "transport": transport,
                "oauth": oauth_config,
                "description": description,
                "registered_at": time.time(),
            }
        logger.info(f"Registered MCP server: {name} → {url} (transport={transport}, oauth={'yes' if oauth_config else 'no'})")

    def remove(self, name: str) -> bool:
        """Remove a registered MCP server."""
        with self._lock:
            if name in self.servers:
                del self.servers[name]
                self._token_manager.clear(name)
                logger.info(f"Removed MCP server: {name}")
                return True
        return False

    def list_servers(self) -> dict:
        """List all registered servers with their status."""
        builtin = {
            "filesystem": {
                "url": "http://mcp-filesystem:8000/sse",
                "transport": "sse",
                "type": "builtin",
                "description": "Read files from mounted volumes",
            },
            "memory": {
                "url": "http://mcp-starrocks:8000/sse",
                "transport": "sse",
                "type": "builtin",
                "description": "Query StarRocks memory tables",
            },
        }

        third_party = {}
        for name, server in self.servers.items():
            third_party[name] = {
                "url": server["url"],
                "transport": server["transport"],
                "type": "third_party",
                "description": server["description"],
                "oauth": server["oauth"] is not None,
            }

        return {
            "builtin": builtin,
            "third_party": third_party,
            "total": len(builtin) + len(third_party),
        }

    async def _get_headers(self, name: str) -> dict:
        """Get HTTP headers for a server, including OAuth token if configured."""
        server = self.servers.get(name)
        if not server:
            return {}

        headers = {"Content-Type": "application/json"}

        if server.get("oauth"):
            token = await self._token_manager.get_token(name, server["oauth"])
            headers["Authorization"] = f"Bearer {token}"

        return headers

    async def call_tool(self, server_name: str, tool_name: str, params: dict) -> dict:
        """Call a tool on a registered MCP server.

        For SSE transport, sends a JSON-RPC request to the server.
        Handles OAuth token injection automatically.
        """
        server = self.servers.get(server_name)
        if not server:
            return {"error": f"MCP server '{server_name}' not found"}

        headers = await self._get_headers(server_name)
        url = server["url"]

        # MCP JSON-RPC tool call
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=rpc_request, headers=headers)
                resp.raise_for_status()
                result = resp.json()

            logger.info(f"MCP tool call: {server_name}/{tool_name} → success")
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401 and server.get("oauth"):
                # Token expired — clear cache and retry once
                self._token_manager.clear(server_name)
                headers = await self._get_headers(server_name)
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, json=rpc_request, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
        except Exception as e:
            logger.error(f"MCP tool call failed: {server_name}/{tool_name}: {e}")
            return {"error": str(e)}


# Singleton registry instance
registry = MCPServerRegistry()
