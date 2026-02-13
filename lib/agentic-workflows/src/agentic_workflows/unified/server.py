"""
Unified Server for Agentic Workflows.

This module provides a UnifiedServer that exposes agents via multiple protocols:
- A2A protocol endpoints
- MCP protocol endpoints
- MCP-UI widget endpoints
- OpenAI Apps SDK endpoints

All protocols are served from a single FastAPI instance with protocol-specific
mounts and routes.

Example:
    >>> from agentic_workflows.unified import (
    ...     UnifiedServer,
    ...     UnifiedAgent,
    ...     create_unified_server,
    ... )
    >>>
    >>> # Create agents
    >>> helper = UnifiedAgent(name="helper", instructions="...")
    >>> specialist = UnifiedAgent(name="specialist", instructions="...")
    >>>
    >>> # Create unified server
    >>> server = create_unified_server(
    ...     agents=[helper, specialist],
    ...     name="my-server",
    ...     host="0.0.0.0",
    ...     port=8000,
    ... )
    >>>
    >>> # Run the server
    >>> await server.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_workflows.unified.agent import UnifiedAgent
    from agentic_workflows.unified.mcp import UnifiedMCPServer

logger = logging.getLogger(__name__)


class ServerProtocol(Enum):
    """Supported server protocols."""

    A2A = "a2a"
    MCP = "mcp"
    MCP_UI = "mcp_ui"
    OPENAI_APPS = "openai_apps"
    HTTP_REST = "http"


@dataclass
class ServerEndpoint:
    """Configuration for a server endpoint."""

    protocol: ServerProtocol
    path: str
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedServerConfig:
    """Configuration for unified server."""

    # Server identity
    name: str = "unified-server"
    version: str = "1.0.0"
    description: str = ""

    # Network settings
    host: str = "127.0.0.1"
    port: int = 8000

    # Protocol endpoints
    enable_a2a: bool = True
    enable_mcp: bool = True
    enable_mcp_ui: bool = True
    enable_openai_apps: bool = True
    enable_http_api: bool = True

    # Paths
    a2a_path: str = "/a2a"
    mcp_path: str = "/mcp"
    mcp_ui_path: str = "/ui"
    apps_path: str = "/apps"
    api_path: str = "/api"

    # CORS
    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_methods: list[str] = field(default_factory=lambda: ["*"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])

    # A2A settings
    a2a_organization: str = "Agentic Workflows"
    a2a_streaming: bool = True

    # Session management
    session_ttl: int = 3600  # 1 hour
    max_sessions: int = 1000


@dataclass
class AgentRegistration:
    """Registration info for an agent on the server."""

    agent: UnifiedAgent
    path: str
    enabled: bool = True
    protocols: list[ServerProtocol] = field(default_factory=list)


class UnifiedServer:
    """Unified server serving agents via multiple protocols.

    The UnifiedServer provides a single FastAPI application that serves:
    - A2A protocol: /.well-known/agent.json, /a2a (JSON-RPC)
    - MCP protocol: /mcp (stdio-over-http or SSE)
    - MCP-UI: /ui (widget resources)
    - OpenAI Apps: /apps (widget tools)
    - HTTP REST: /api (simple REST endpoints)

    Example:
        >>> from agentic_workflows.unified import UnifiedServer, UnifiedAgent
        >>>
        >>> agent = UnifiedAgent(name="helper", instructions="...")
        >>>
        >>> server = UnifiedServer()
        >>> server.register_agent(agent)
        >>>
        >>> # Run with uvicorn
        >>> await server.run()
    """

    def __init__(self, config: UnifiedServerConfig | None = None):
        """Initialize unified server.

        Args:
            config: Server configuration.
        """
        self.config = config or UnifiedServerConfig()

        # Registered agents
        self._agents: dict[str, AgentRegistration] = {}

        # FastAPI app (lazy)
        self._app: Any = None

        # Task storage
        self._tasks: dict[str, dict[str, Any]] = {}

        # Session storage
        self._sessions: dict[str, dict[str, Any]] = {}

        # MCP server (lazy)
        self._mcp_server: UnifiedMCPServer | None = None

    def _get_app(self) -> Any:
        """Get or create FastAPI application."""
        if self._app is None:
            try:
                from fastapi import FastAPI, HTTPException, Request, Response
                from fastapi.middleware.cors import CORSMiddleware
                from fastapi.responses import JSONResponse, StreamingResponse

                self._app = FastAPI(
                    title=self.config.name,
                    version=self.config.version,
                    description=self.config.description,
                )

                # Add CORS
                self._app.add_middleware(
                    CORSMiddleware,
                    allow_origins=self.config.allow_origins,
                    allow_credentials=True,
                    allow_methods=self.config.allow_methods,
                    allow_headers=self.config.allow_headers,
                )

                # Setup routes
                self._setup_routes()

            except ImportError:
                raise ImportError(
                    "FastAPI not installed. Install with: pip install fastapi uvicorn"
                )

        return self._app

    def _setup_routes(self) -> None:
        """Setup all protocol routes."""
        app = self._app

        if self.config.enable_a2a:
            self._setup_a2a_routes()

        if self.config.enable_mcp:
            self._setup_mcp_routes()

        if self.config.enable_mcp_ui:
            self._setup_mcp_ui_routes()

        if self.config.enable_openai_apps:
            self._setup_apps_routes()

        if self.config.enable_http_api:
            self._setup_http_routes()

        # Health check
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "server": self.config.name,
                "version": self.config.version,
                "agents": list(self._agents.keys()),
            }

    def _setup_a2a_routes(self) -> None:
        """Setup A2A protocol routes."""
        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse

        app = self._app

        # Agent card discovery
        @app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Get combined agent card for all registered agents."""
            return JSONResponse(self._get_combined_agent_card())

        # Per-agent cards
        @app.get("/.well-known/agents/{agent_name}.json")
        async def get_specific_agent_card(agent_name: str):
            """Get agent card for specific agent."""
            if agent_name not in self._agents:
                return JSONResponse(
                    {"error": f"Agent '{agent_name}' not found"},
                    status_code=404,
                )
            agent = self._agents[agent_name].agent
            return JSONResponse(agent.get_a2a_card())

        # A2A JSON-RPC endpoint
        @app.post(self.config.a2a_path)
        async def a2a_rpc(request: Request):
            """Handle A2A JSON-RPC requests."""
            try:
                body = await request.json()
                method = body.get("method", "")
                params = body.get("params", {})
                request_id = body.get("id", str(uuid.uuid4()))

                result = await self._handle_a2a_method(method, params)

                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": result,
                    }
                )

            except Exception as e:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": body.get("id") if "body" in dir() else None,
                        "error": {
                            "code": -32603,
                            "message": str(e),
                        },
                    }
                )

        # A2A streaming endpoint
        @app.post(f"{self.config.a2a_path}/stream")
        async def a2a_stream(request: Request):
            """Handle A2A streaming requests."""
            body = await request.json()
            params = body.get("params", {})

            async def event_generator():
                async for event in self._handle_a2a_stream(params):
                    yield f"data: {json.dumps(event)}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )

    async def _handle_a2a_method(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle A2A JSON-RPC method."""
        if method == "message/send":
            return await self._a2a_send_message(params)
        elif method == "tasks/get":
            return await self._a2a_get_task(params)
        elif method == "tasks/cancel":
            return await self._a2a_cancel_task(params)
        else:
            raise ValueError(f"Unknown method: {method}")

    async def _a2a_send_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle A2A message/send."""
        task_id = params.get("id", str(uuid.uuid4()))
        message = params.get("message", {})
        skill_id = params.get("skillId")
        session_id = params.get("sessionId")

        # Extract agent from skill_id or use default
        agent_name = None
        if skill_id:
            # Check if skill belongs to specific agent
            for name, reg in self._agents.items():
                agent_tools = [t.name for t in reg.agent._tools.values()]
                if skill_id in agent_tools:
                    agent_name = name
                    break

        if agent_name is None and self._agents:
            agent_name = list(self._agents.keys())[0]

        if agent_name is None:
            return {
                "id": task_id,
                "status": {"state": "failed", "error": "No agents registered"},
            }

        agent = self._agents[agent_name].agent

        # Extract text from message
        text_parts = []
        for part in message.get("parts", []):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        user_input = "\n".join(text_parts)

        # Run agent
        result = await agent.run(user_input, session_id)

        # Store task
        task = result.to_a2a_task()
        task["id"] = task_id
        task["sessionId"] = session_id or task_id
        self._tasks[task_id] = task

        return task

    async def _a2a_get_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle A2A tasks/get."""
        task_id = params.get("id")
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        return task

    async def _a2a_cancel_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle A2A tasks/cancel."""
        task_id = params.get("id")
        if task_id in self._tasks:
            self._tasks[task_id]["status"]["state"] = "canceled"
            return {"success": True}
        return {"success": False}

    async def _handle_a2a_stream(
        self,
        params: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle A2A streaming."""
        task_id = params.get("id", str(uuid.uuid4()))
        message = params.get("message", {})
        session_id = params.get("sessionId")

        # Get first agent
        if not self._agents:
            yield {"error": "No agents registered"}
            return

        agent = list(self._agents.values())[0].agent

        # Extract text
        text_parts = []
        for part in message.get("parts", []):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        user_input = "\n".join(text_parts)

        # Emit task status
        yield {
            "type": "status",
            "taskId": task_id,
            "status": {"state": "working"},
        }

        # Run agent
        result = await agent.run(user_input, session_id)

        # Emit message
        yield {
            "type": "message",
            "taskId": task_id,
            "message": {
                "role": "agent",
                "parts": [{"type": "text", "text": result.output}],
            },
        }

        # Emit completion
        yield {
            "type": "status",
            "taskId": task_id,
            "status": {"state": "completed" if result.success else "failed"},
        }

    def _setup_mcp_routes(self) -> None:
        """Setup MCP protocol routes."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        app = self._app

        # MCP tools list
        @app.get(f"{self.config.mcp_path}/tools")
        async def list_mcp_tools():
            """List all MCP tools."""
            tools = []
            for reg in self._agents.values():
                agent_tools = reg.agent.get_tools()
                tools.extend(agent_tools)
            return JSONResponse({"tools": tools})

        # MCP tool call
        @app.post(f"{self.config.mcp_path}/tools/call")
        async def call_mcp_tool(request: Request):
            """Call an MCP tool."""
            body = await request.json()
            tool_name = body.get("name")
            arguments = body.get("arguments", {})

            # Find tool
            for reg in self._agents.values():
                if tool_name in reg.agent._tools:
                    result = await reg.agent._execute_tool(tool_name, arguments)
                    return JSONResponse(
                        {
                            "content": [{"type": "text", "text": str(result)}],
                        }
                    )

            return JSONResponse(
                {"error": f"Tool not found: {tool_name}"},
                status_code=404,
            )

        # MCP resources list
        @app.get(f"{self.config.mcp_path}/resources")
        async def list_mcp_resources():
            """List MCP resources."""
            return JSONResponse({"resources": []})

    def _setup_mcp_ui_routes(self) -> None:
        """Setup MCP-UI routes."""
        from fastapi import Request
        from fastapi.responses import HTMLResponse, JSONResponse

        app = self._app

        # UI resource endpoint
        @app.get(f"{self.config.mcp_ui_path}/{{resource_id}}")
        async def get_ui_resource(resource_id: str):
            """Get UI resource by ID."""
            # Would serve UI resources here
            return JSONResponse(
                {
                    "error": "Resource not found",
                },
                status_code=404,
            )

        # Widget preview
        @app.get(f"{self.config.mcp_ui_path}/preview/{{widget_type}}")
        async def preview_widget(widget_type: str, request: Request):
            """Preview a widget type."""
            try:
                from agentic_workflows.mcp_ui import create_component_html

                html = create_component_html(widget_type, {})
                return HTMLResponse(html)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=400)

    def _setup_apps_routes(self) -> None:
        """Setup OpenAI Apps SDK routes."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        app = self._app

        # Apps widget list
        @app.get(f"{self.config.apps_path}/widgets")
        async def list_widgets():
            """List available widgets."""
            widgets = []
            for reg in self._agents.values():
                for tool in reg.agent._tools.values():
                    if tool.metadata.get("is_widget"):
                        widgets.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "template": tool.metadata.get("template"),
                            }
                        )
            return JSONResponse({"widgets": widgets})

        # Apps widget call
        @app.post(f"{self.config.apps_path}/widgets/{{widget_name}}")
        async def call_widget(widget_name: str, request: Request):
            """Call a widget tool."""
            body = await request.json()

            for reg in self._agents.values():
                tool = reg.agent._tools.get(widget_name)
                if tool and tool.metadata.get("is_widget"):
                    result = await reg.agent._execute_tool(widget_name, body)
                    return JSONResponse(result if isinstance(result, dict) else {"result": result})

            return JSONResponse(
                {"error": f"Widget not found: {widget_name}"},
                status_code=404,
            )

    def _setup_http_routes(self) -> None:
        """Setup HTTP REST API routes."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        app = self._app

        # List agents
        @app.get(f"{self.config.api_path}/agents")
        async def list_agents():
            """List all registered agents."""
            agents = []
            for name, reg in self._agents.items():
                agents.append(
                    {
                        "name": name,
                        "status": reg.agent.get_status(),
                    }
                )
            return JSONResponse({"agents": agents})

        # Get agent
        @app.get(f"{self.config.api_path}/agents/{{agent_name}}")
        async def get_agent(agent_name: str):
            """Get agent details."""
            if agent_name not in self._agents:
                return JSONResponse(
                    {"error": f"Agent not found: {agent_name}"},
                    status_code=404,
                )
            return JSONResponse(self._agents[agent_name].agent.get_status())

        # Chat with agent
        @app.post(f"{self.config.api_path}/agents/{{agent_name}}/chat")
        async def chat_with_agent(agent_name: str, request: Request):
            """Send message to agent."""
            if agent_name not in self._agents:
                return JSONResponse(
                    {"error": f"Agent not found: {agent_name}"},
                    status_code=404,
                )

            body = await request.json()
            message = body.get("message", "")
            session_id = body.get("session_id")

            agent = self._agents[agent_name].agent
            result = await agent.run(message, session_id)

            return JSONResponse(
                {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "session_id": agent._session_id,
                }
            )

        # List tasks
        @app.get(f"{self.config.api_path}/tasks")
        async def list_tasks():
            """List all tasks."""
            return JSONResponse({"tasks": list(self._tasks.values())})

    def _get_combined_agent_card(self) -> dict[str, Any]:
        """Get combined A2A agent card for all agents."""
        all_skills = []
        for reg in self._agents.values():
            card = reg.agent.get_a2a_card()
            # Prefix skills with agent name
            for skill in card.get("skills", []):
                skill["id"] = f"{reg.agent.name}:{skill['id']}"
                skill["name"] = f"[{reg.agent.name}] {skill['name']}"
                all_skills.append(skill)

        return {
            "name": self.config.name,
            "description": self.config.description
            or f"Unified server with {len(self._agents)} agents",
            "url": f"http://{self.config.host}:{self.config.port}",
            "version": self.config.version,
            "protocolVersion": "0.3",
            "provider": {
                "organization": self.config.a2a_organization,
            },
            "capabilities": {
                "streaming": self.config.a2a_streaming,
                "tasks": True,
                "multiAgent": len(self._agents) > 1,
            },
            "skills": all_skills,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
        }

    def register_agent(
        self,
        agent: UnifiedAgent,
        path: str | None = None,
        protocols: list[ServerProtocol] | None = None,
    ) -> None:
        """Register an agent with the server.

        Args:
            agent: Agent to register.
            path: Optional custom path prefix.
            protocols: Protocols to enable (all by default).
        """
        agent_path = path or f"/{agent.name}"
        agent_protocols = protocols or [
            ServerProtocol.A2A,
            ServerProtocol.MCP,
            ServerProtocol.HTTP_REST,
        ]

        self._agents[agent.name] = AgentRegistration(
            agent=agent,
            path=agent_path,
            protocols=agent_protocols,
        )

        logger.info(f"Registered agent '{agent.name}' at {agent_path}")

    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent.

        Args:
            name: Agent name.

        Returns:
            True if removed.
        """
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def get_agent(self, name: str) -> UnifiedAgent | None:
        """Get registered agent by name.

        Args:
            name: Agent name.

        Returns:
            Agent or None.
        """
        reg = self._agents.get(name)
        return reg.agent if reg else None

    async def run(self) -> None:
        """Run the server with uvicorn."""
        try:
            import uvicorn

            app = self._get_app()

            config = uvicorn.Config(
                app,
                host=self.config.host,
                port=self.config.port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            await server.serve()

        except ImportError:
            raise ImportError("uvicorn not installed. Install with: pip install uvicorn")

    def run_sync(self) -> None:
        """Run the server synchronously."""
        asyncio.run(self.run())

    def get_app(self) -> Any:
        """Get the FastAPI application.

        Returns:
            FastAPI app instance.
        """
        return self._get_app()

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "host": self.config.host,
            "port": self.config.port,
            "agents": {name: reg.agent.get_status() for name, reg in self._agents.items()},
            "tasks_count": len(self._tasks),
            "sessions_count": len(self._sessions),
            "protocols": {
                "a2a": self.config.enable_a2a,
                "mcp": self.config.enable_mcp,
                "mcp_ui": self.config.enable_mcp_ui,
                "openai_apps": self.config.enable_openai_apps,
                "http": self.config.enable_http_api,
            },
        }


def create_unified_server(
    agents: list[UnifiedAgent] | None = None,
    name: str = "unified-server",
    host: str = "127.0.0.1",
    port: int = 8000,
    **kwargs,
) -> UnifiedServer:
    """Factory function to create a unified server.

    Args:
        agents: List of agents to register.
        name: Server name.
        host: Host to bind to.
        port: Port to listen on.
        **kwargs: Additional configuration.

    Returns:
        Configured UnifiedServer.

    Example:
        >>> from agentic_workflows.unified import create_unified_agent
        >>>
        >>> agent1 = create_unified_agent(name="helper")
        >>> agent2 = create_unified_agent(name="specialist")
        >>>
        >>> server = create_unified_server(
        ...     agents=[agent1, agent2],
        ...     name="my-server",
        ...     port=8080,
        ... )
        >>>
        >>> # Run
        >>> server.run_sync()
    """
    config = UnifiedServerConfig(
        name=name,
        host=host,
        port=port,
        **kwargs,
    )

    server = UnifiedServer(config)

    if agents:
        for agent in agents:
            server.register_agent(agent)

    return server
