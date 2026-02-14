"""Widget server for OpenAI Apps SDK integration.

This module provides the WidgetServer class that exposes widgets
via the MCP (Model Context Protocol) with support for:
- Widget registration with templates and handlers
- Tool invocations that return structured content with _meta
- Resource endpoints for serving widget HTML
- Widget session state management

Based on FastMCP patterns for efficient MCP server implementation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from mcp.server import Server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    TextContent,
    Tool,
)

from .components import WIDGET_TEMPLATES, get_template
from .state import (
    InMemoryStateStore,
    StateStore,
    WidgetSessionManager,
)
from .widget_types import (
    StructuredContent,
    WidgetMeta,
    WidgetState,
    WidgetTemplate,
)

logger = logging.getLogger(__name__)

# Type for widget handlers
T = TypeVar("T")
WidgetHandler = Callable[[dict[str, Any], WidgetState | None], Awaitable[StructuredContent]]


@dataclass
class WidgetRegistration:
    """Registration for a widget tool.

    Attributes:
        name: Tool name
        template_uri: Widget template URI
        handler: Async handler function
        description: Tool description
        input_schema: JSON Schema for tool inputs
        stateful: Whether widget maintains state
        session_ttl: Session TTL in seconds
    """

    name: str
    template_uri: str
    handler: WidgetHandler
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    stateful: bool = False
    session_ttl: int = 3600


class WidgetServer:
    """MCP server for OpenAI Apps SDK widgets.

    Provides a high-level interface for creating widget-enabled MCP servers
    that integrate with the OpenAI Apps SDK.

    Example:
        >>> server = WidgetServer(name="my-widgets")
        >>>
        >>> @server.widget("product_list", "widget://list-view")
        ... async def product_list(args, state):
        ...     products = await fetch_products(args.get("category"))
        ...     return ListViewPayload(items=products).to_structured_content()
        >>>
        >>> await server.run()
    """

    def __init__(
        self,
        name: str = "widget-server",
        version: str = "1.0.0",
        session_manager: WidgetSessionManager | None = None,
        state_store: StateStore | None = None,
    ):
        """Initialize widget server.

        Args:
            name: Server name.
            version: Server version.
            session_manager: Custom session manager.
            state_store: Custom state store for sessions.
        """
        self.name = name
        self.version = version
        self.server = Server(name)
        self._widgets: dict[str, WidgetRegistration] = {}
        self._custom_templates: dict[str, WidgetTemplate] = {}
        self._initialized = False

        # Session management
        if session_manager:
            self.session_manager = session_manager
        elif state_store:
            self.session_manager = WidgetSessionManager(store=state_store)
        else:
            self.session_manager = WidgetSessionManager(store=InMemoryStateStore())

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP protocol handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available widget tools."""
            tools = []

            for name, reg in self._widgets.items():
                # Build input schema
                schema = (
                    reg.input_schema.copy()
                    if reg.input_schema
                    else {
                        "type": "object",
                        "properties": {},
                    }
                )

                # Add widgetSessionId for stateful widgets
                if reg.stateful:
                    schema.setdefault("properties", {})
                    schema["properties"]["widgetSessionId"] = {
                        "type": "string",
                        "description": "Session ID for stateful widget (optional, auto-generated if not provided)",
                    }
                    schema["properties"]["widgetState"] = {
                        "type": "object",
                        "description": "Current widget state from client",
                    }

                tools.append(
                    Tool(
                        name=name,
                        description=reg.description or f"Widget: {name}",
                        inputSchema=schema,
                    )
                )

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Execute a widget tool."""
            if name not in self._widgets:
                return [
                    TextContent(type="text", text=json.dumps({"error": f"Unknown widget: {name}"}))
                ]

            reg = self._widgets[name]

            try:
                # Handle session state for stateful widgets
                state: WidgetState | None = None
                session_id = arguments.pop("widgetSessionId", None)
                client_state = arguments.pop("widgetState", None)

                if reg.stateful:
                    if session_id:
                        # Get existing session
                        state = await self.session_manager.get_state(session_id)
                        if state is None:
                            # Session expired, create new
                            session_id = await self.session_manager.create_session(
                                session_id=session_id,
                                ttl=reg.session_ttl,
                            )
                            state = await self.session_manager.get_state(session_id)
                    else:
                        # Create new session
                        session_id = await self.session_manager.create_session(
                            ttl=reg.session_ttl,
                        )
                        state = await self.session_manager.get_state(session_id)

                    # Merge client state if provided
                    if client_state and state:
                        state = await self.session_manager.merge_state(
                            session_id, client_state, deep=True
                        )

                # Call handler
                result = await reg.handler(arguments, state)

                # Add _meta with widget template and session info
                meta = WidgetMeta(
                    output_template=reg.template_uri,
                    widget_session_id=session_id or "",
                    tool_invocation={"tool": name, "arguments": arguments},
                )
                result["_meta"] = meta.to_dict()

                # Update state if changed
                if reg.stateful and state and session_id:
                    await self.session_manager.update_state(
                        session_id,
                        state.state_data,
                        extend_ttl=True,
                    )

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2),
                    )
                ]

            except Exception as e:
                logger.exception(f"Widget handler error: {name}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": str(e),
                                "_meta": {
                                    "openai/outputTemplate": "widget://status",
                                    "status": "error",
                                },
                            }
                        ),
                    )
                ]

        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            """List widget template resources."""
            resources = []

            # Built-in templates
            for name, template in WIDGET_TEMPLATES.items():
                resources.append(
                    Resource(
                        uri=template.uri,
                        name=template.name or name,
                        description=template.description,
                        mimeType=template.mime_type,
                    )
                )

            # Custom templates
            for name, template in self._custom_templates.items():
                resources.append(
                    Resource(
                        uri=template.uri,
                        name=template.name or name,
                        description=template.description,
                        mimeType=template.mime_type,
                    )
                )

            return resources

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read widget template HTML."""
            # Extract template name from URI
            if uri.startswith("widget://"):
                name = uri[9:]  # Remove "widget://" prefix
            else:
                name = uri

            # Check built-in templates
            template = get_template(name)
            if template:
                return template.html

            # Check custom templates
            if name in self._custom_templates:
                return self._custom_templates[name].html

            raise ValueError(f"Unknown widget template: {uri}")

    def register_widget(
        self,
        name: str,
        template_uri: str,
        handler: WidgetHandler,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        stateful: bool = False,
        session_ttl: int = 3600,
    ) -> None:
        """Register a widget tool.

        Args:
            name: Tool name (unique identifier).
            template_uri: Widget template URI (e.g., "widget://list-view").
            handler: Async handler function (args, state) -> StructuredContent.
            description: Tool description for LLM.
            input_schema: JSON Schema for tool inputs.
            stateful: Whether widget maintains state across calls.
            session_ttl: Session TTL in seconds for stateful widgets.

        Example:
            >>> async def my_handler(args, state):
            ...     return {"items": [...], "type": "list_view"}
            >>>
            >>> server.register_widget(
            ...     name="my_list",
            ...     template_uri="widget://list-view",
            ...     handler=my_handler,
            ...     description="Display a list of items",
            ... )
        """
        self._widgets[name] = WidgetRegistration(
            name=name,
            template_uri=template_uri,
            handler=handler,
            description=description,
            input_schema=input_schema or {},
            stateful=stateful,
            session_ttl=session_ttl,
        )
        logger.info(f"Registered widget: {name} -> {template_uri}")

    def widget(
        self,
        name: str,
        template_uri: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        stateful: bool = False,
        session_ttl: int = 3600,
    ):
        """Decorator for registering widget handlers.

        Args:
            name: Tool name.
            template_uri: Widget template URI.
            description: Tool description.
            input_schema: JSON Schema for inputs.
            stateful: Whether widget maintains state.
            session_ttl: Session TTL for stateful widgets.

        Returns:
            Decorator function.

        Example:
            >>> @server.widget("products", "widget://list-view", stateful=True)
            ... async def products(args, state):
            ...     items = await fetch_products()
            ...     return ListViewPayload(items=items).to_structured_content()
        """

        def decorator(handler: WidgetHandler) -> WidgetHandler:
            self.register_widget(
                name=name,
                template_uri=template_uri,
                handler=handler,
                description=description or handler.__doc__ or "",
                input_schema=input_schema,
                stateful=stateful,
                session_ttl=session_ttl,
            )
            return handler

        return decorator

    def register_template(
        self,
        name: str,
        template: WidgetTemplate,
    ) -> None:
        """Register a custom widget template.

        Args:
            name: Template name.
            template: Widget template definition.

        Example:
            >>> template = WidgetTemplate(
            ...     uri="widget://custom",
            ...     html="<div>Custom widget</div>",
            ...     name="Custom",
            ...     description="A custom widget"
            ... )
            >>> server.register_template("custom", template)
        """
        self._custom_templates[name] = template
        logger.info(f"Registered template: {name} -> {template.uri}")

    async def get_session(self, session_id: str) -> WidgetState | None:
        """Get session state by ID.

        Args:
            session_id: Session identifier.

        Returns:
            WidgetState or None if not found.
        """
        return await self.session_manager.get_state(session_id)

    async def update_session(
        self,
        session_id: str,
        state_data: dict[str, Any],
    ) -> WidgetState | None:
        """Update session state.

        Args:
            session_id: Session identifier.
            state_data: New state data.

        Returns:
            Updated WidgetState or None if not found.
        """
        return await self.session_manager.update_state(session_id, state_data)

    async def run(self):
        """Run the MCP server using stdio transport.

        This is the main entry point for running the widget server.
        It handles the MCP protocol over stdin/stdout.

        Example:
            >>> server = WidgetServer()
            >>> # Register widgets...
            >>> await server.run()
        """
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=self.name,
                    server_version=self.version,
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(
                            prompts_changed=False,
                            resources_changed=True,
                            tools_changed=True,
                        ),
                        experimental_capabilities={},
                    ),
                ),
            )

    async def close(self):
        """Close the server and cleanup resources."""
        await self.session_manager.close()


class WidgetServerBuilder:
    """Builder for creating widget servers with fluent API.

    Example:
        >>> server = (
        ...     WidgetServerBuilder("my-server")
        ...     .with_template("custom", custom_template)
        ...     .with_widget("list", "widget://list-view", list_handler)
        ...     .with_stateful_widget("cart", "widget://form", cart_handler)
        ...     .build()
        ... )
    """

    def __init__(self, name: str = "widget-server", version: str = "1.0.0"):
        """Initialize builder.

        Args:
            name: Server name.
            version: Server version.
        """
        self.name = name
        self.version = version
        self._templates: list[tuple[str, WidgetTemplate]] = []
        self._widgets: list[tuple[str, str, WidgetHandler, dict]] = []
        self._state_store: StateStore | None = None
        self._session_ttl: int = 3600

    def with_state_store(self, store: StateStore) -> WidgetServerBuilder:
        """Set custom state store.

        Args:
            store: State store implementation.

        Returns:
            Self for chaining.
        """
        self._state_store = store
        return self

    def with_session_ttl(self, ttl: int) -> WidgetServerBuilder:
        """Set default session TTL.

        Args:
            ttl: TTL in seconds.

        Returns:
            Self for chaining.
        """
        self._session_ttl = ttl
        return self

    def with_template(
        self,
        name: str,
        template: WidgetTemplate,
    ) -> WidgetServerBuilder:
        """Add a custom template.

        Args:
            name: Template name.
            template: Widget template.

        Returns:
            Self for chaining.
        """
        self._templates.append((name, template))
        return self

    def with_widget(
        self,
        name: str,
        template_uri: str,
        handler: WidgetHandler,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> WidgetServerBuilder:
        """Add a stateless widget.

        Args:
            name: Widget name.
            template_uri: Template URI.
            handler: Handler function.
            description: Widget description.
            input_schema: Input schema.

        Returns:
            Self for chaining.
        """
        self._widgets.append(
            (
                name,
                template_uri,
                handler,
                {
                    "description": description,
                    "input_schema": input_schema,
                    "stateful": False,
                },
            )
        )
        return self

    def with_stateful_widget(
        self,
        name: str,
        template_uri: str,
        handler: WidgetHandler,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        session_ttl: int | None = None,
    ) -> WidgetServerBuilder:
        """Add a stateful widget.

        Args:
            name: Widget name.
            template_uri: Template URI.
            handler: Handler function.
            description: Widget description.
            input_schema: Input schema.
            session_ttl: Session TTL (default: builder default).

        Returns:
            Self for chaining.
        """
        self._widgets.append(
            (
                name,
                template_uri,
                handler,
                {
                    "description": description,
                    "input_schema": input_schema,
                    "stateful": True,
                    "session_ttl": session_ttl or self._session_ttl,
                },
            )
        )
        return self

    def build(self) -> WidgetServer:
        """Build the widget server.

        Returns:
            Configured WidgetServer instance.
        """
        server = WidgetServer(
            name=self.name,
            version=self.version,
            state_store=self._state_store,
        )

        # Register templates
        for name, template in self._templates:
            server.register_template(name, template)

        # Register widgets
        for name, template_uri, handler, options in self._widgets:
            server.register_widget(
                name=name,
                template_uri=template_uri,
                handler=handler,
                **options,
            )

        return server


def create_widget_server(
    name: str = "widget-server",
    version: str = "1.0.0",
    state_store: StateStore | None = None,
) -> WidgetServer:
    """Factory function for creating a widget server.

    Args:
        name: Server name.
        version: Server version.
        state_store: Optional state store for sessions.

    Returns:
        New WidgetServer instance.

    Example:
        >>> server = create_widget_server("my-widgets")
        >>> server.register_widget("list", "widget://list-view", handler)
        >>> await server.run()
    """
    return WidgetServer(
        name=name,
        version=version,
        state_store=state_store,
    )
