"""MCP-UI Server Integration.

This module provides the MCPUIServer class and integration utilities
for creating MCP servers with UI support.

Features:
- MCPUIServer: Extended MCP server with UI resource capabilities
- @ui_tool decorator: Mark tools that return UI resources
- OpenAI Apps widget adapter: Compatibility with ChatGPT widgets
- FastMCP-style patterns for efficient server implementation

Based on MCP-UI SDK patterns with backward compatibility for
existing openai_apps widgets.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TypeVar, ParamSpec, overload

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    CallToolResult,
)

from .ui_types import (
    UIResource,
    UIResourceContent,
    UIMetadata,
    MimeType,
    ContentType,
    FrameSize,
    UIActionType,
    UIActionResult,
    UIStructuredContent,
)
from .resource import (
    create_ui_resource,
    RawHtmlResource,
    ExternalUrlResource,
    RemoteDomResource,
    wrap_html_with_adapters,
    get_apps_sdk_adapter_script,
)
from .actions import (
    UIActionHandler,
    UIActionHandlerConfig,
    create_action_handler,
)
from .components import (
    create_component_html,
    create_component_resource,
    ButtonConfig,
    InputConfig,
    SelectConfig,
    TableConfig,
    ChartConfig,
    ListConfig,
    CardConfig,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# Type for UI tool handlers
UIToolHandler = Callable[..., Awaitable[UIResource | UIStructuredContent | dict[str, Any]]]


@dataclass
class UIToolRegistration:
    """Registration for a UI tool.

    Attributes:
        name: Tool name.
        handler: Async handler function.
        description: Tool description.
        input_schema: JSON Schema for inputs.
        template_uri: Default template URI for responses.
        return_type: Expected return type (resource, structured, auto).
        enable_apps_sdk: Enable Apps SDK adapter for responses.
    """
    name: str
    handler: UIToolHandler
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    template_uri: str = ""
    return_type: str = "auto"  # "resource", "structured", "auto"
    enable_apps_sdk: bool = True


@dataclass
class UIResourceTemplate:
    """Template for UI resources.

    Attributes:
        uri: Template URI.
        name: Template name.
        description: Template description.
        html: HTML content.
        mime_type: Content MIME type.
    """
    uri: str
    name: str
    description: str = ""
    html: str = ""
    mime_type: str = "text/html"


class MCPUIServer:
    """MCP Server with UI resource support.

    Extends standard MCP server patterns with support for returning
    UI resources from tools, compatible with the MCP-UI specification.

    Example:
        >>> server = MCPUIServer(name="my-ui-server")
        >>>
        >>> @server.ui_tool("show_products")
        ... async def show_products(category: str) -> UIResource:
        ...     products = await fetch_products(category)
        ...     html = render_product_list(products)
        ...     return RawHtmlResource.create(html)
        >>>
        >>> await server.run()
    """

    def __init__(
        self,
        name: str = "mcp-ui-server",
        version: str = "1.0.0",
        enable_apps_sdk: bool = True,
        action_handler: UIActionHandler | None = None,
    ):
        """Initialize MCP-UI server.

        Args:
            name: Server name for MCP identification.
            version: Server version string.
            enable_apps_sdk: Enable Apps SDK adapter by default.
            action_handler: Custom action handler for UI callbacks.
        """
        self.name = name
        self.version = version
        self.enable_apps_sdk = enable_apps_sdk
        self.server = Server(name)
        self._ui_tools: dict[str, UIToolRegistration] = {}
        self._templates: dict[str, UIResourceTemplate] = {}
        self._action_handler = action_handler or create_action_handler()
        self._initialized = False

        # Setup MCP handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP protocol handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available UI tools."""
            tools = []

            for name, reg in self._ui_tools.items():
                schema = reg.input_schema.copy() if reg.input_schema else {
                    "type": "object",
                    "properties": {},
                }

                tools.append(Tool(
                    name=name,
                    description=reg.description or f"UI Tool: {name}",
                    inputSchema=schema,
                ))

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Execute a UI tool."""
            if name not in self._ui_tools:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )]

            reg = self._ui_tools[name]

            try:
                # Call the handler
                result = await reg.handler(**arguments)

                # Convert result to response format
                response = self._format_tool_response(result, reg)

                return [TextContent(
                    type="text",
                    text=json.dumps(response, indent=2),
                )]

            except Exception as e:
                logger.exception(f"UI tool error: {name}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "type": "resource",
                        "resource": {
                            "uri": f"ui://error/{uuid.uuid4().hex[:8]}",
                            "mimeType": "text/html",
                            "text": f"<p style='color: red;'>Error: {str(e)}</p>",
                        },
                    }),
                )]

        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            """List UI resource templates."""
            resources = []

            for uri, template in self._templates.items():
                resources.append(Resource(
                    uri=template.uri,
                    name=template.name,
                    description=template.description,
                    mimeType=template.mime_type,
                ))

            return resources

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a UI resource template."""
            template = self._templates.get(uri)
            if template:
                return template.html

            raise ValueError(f"Unknown resource: {uri}")

    def _format_tool_response(
        self,
        result: UIResource | UIStructuredContent | dict[str, Any],
        reg: UIToolRegistration,
    ) -> dict[str, Any]:
        """Format tool result for MCP response.

        Args:
            result: Handler return value.
            reg: Tool registration.

        Returns:
            Formatted response dictionary.
        """
        # If result is UIResource, convert to MCP format
        if isinstance(result, UIResource):
            return result.to_tool_result()

        # If result is a dict with MCP-UI structure
        if isinstance(result, dict):
            # Check if it's already in resource format
            if result.get("type") == "resource":
                return result

            # Check if it has UI content that needs wrapping
            if "html" in result or "content" in result:
                html = result.get("html") or result.get("content", "")
                if reg.enable_apps_sdk and self.enable_apps_sdk:
                    html = wrap_html_with_adapters(html, {
                        "appsSdk": {"enabled": True}
                    })

                uri = result.get("uri") or f"ui://response/{uuid.uuid4().hex[:8]}"
                return {
                    "type": "resource",
                    "resource": {
                        "uri": uri,
                        "mimeType": result.get("mimeType", "text/html"),
                        "text": html,
                    },
                }

            # Return as structured content with metadata
            template_uri = reg.template_uri or result.get("_meta", {}).get("template")
            if template_uri:
                result.setdefault("_meta", {})
                result["_meta"]["mcp-ui/template"] = template_uri

            return result

        # Fallback: wrap in basic response
        return {
            "type": "resource",
            "resource": {
                "uri": f"ui://response/{uuid.uuid4().hex[:8]}",
                "mimeType": "text/html",
                "text": f"<pre>{json.dumps(result, indent=2)}</pre>",
            },
        }

    def register_ui_tool(
        self,
        name: str,
        handler: UIToolHandler,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        template_uri: str = "",
        return_type: str = "auto",
        enable_apps_sdk: bool | None = None,
    ) -> None:
        """Register a UI tool.

        Args:
            name: Tool name.
            handler: Async handler function.
            description: Tool description.
            input_schema: JSON Schema for inputs.
            template_uri: Default template URI.
            return_type: Expected return type.
            enable_apps_sdk: Enable Apps SDK adapter (default: server setting).

        Example:
            >>> async def my_handler(query: str) -> UIResource:
            ...     html = f"<p>Search results for: {query}</p>"
            ...     return RawHtmlResource.create(html)
            >>>
            >>> server.register_ui_tool(
            ...     name="search",
            ...     handler=my_handler,
            ...     description="Search and display results",
            ...     input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ... )
        """
        self._ui_tools[name] = UIToolRegistration(
            name=name,
            handler=handler,
            description=description or handler.__doc__ or "",
            input_schema=input_schema or {},
            template_uri=template_uri,
            return_type=return_type,
            enable_apps_sdk=enable_apps_sdk if enable_apps_sdk is not None else self.enable_apps_sdk,
        )
        logger.info(f"Registered UI tool: {name}")

    def ui_tool(
        self,
        name: str | None = None,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        template_uri: str = "",
        enable_apps_sdk: bool | None = None,
    ) -> Callable[[UIToolHandler], UIToolHandler]:
        """Decorator for registering UI tools.

        Args:
            name: Tool name (defaults to function name).
            description: Tool description.
            input_schema: JSON Schema for inputs.
            template_uri: Default template URI.
            enable_apps_sdk: Enable Apps SDK adapter.

        Returns:
            Decorator function.

        Example:
            >>> @server.ui_tool("product_list")
            ... async def product_list(category: str = "all") -> UIResource:
            ...     '''Display products in a category.'''
            ...     products = await get_products(category)
            ...     return create_product_list_ui(products)
        """
        def decorator(handler: UIToolHandler) -> UIToolHandler:
            tool_name = name or handler.__name__
            self.register_ui_tool(
                name=tool_name,
                handler=handler,
                description=description or handler.__doc__ or "",
                input_schema=input_schema,
                template_uri=template_uri,
                enable_apps_sdk=enable_apps_sdk,
            )
            return handler
        return decorator

    def register_template(
        self,
        uri: str,
        html: str,
        name: str = "",
        description: str = "",
        mime_type: str = "text/html",
    ) -> None:
        """Register a UI resource template.

        Args:
            uri: Template URI.
            html: HTML content.
            name: Template name.
            description: Template description.
            mime_type: Content MIME type.

        Example:
            >>> server.register_template(
            ...     uri="ui://templates/product-card",
            ...     html=PRODUCT_CARD_HTML,
            ...     name="Product Card",
            ...     description="Display a single product",
            ... )
        """
        self._templates[uri] = UIResourceTemplate(
            uri=uri,
            name=name or uri,
            description=description,
            html=html,
            mime_type=mime_type,
        )
        logger.info(f"Registered template: {uri}")

    async def handle_action(self, action_data: dict[str, Any]) -> dict[str, Any]:
        """Handle a UI action from a component.

        Args:
            action_data: Action data from UI component.

        Returns:
            Action response dictionary.

        Example:
            >>> response = await server.handle_action({
            ...     "type": "tool",
            ...     "payload": {"toolName": "search", "params": {"q": "test"}},
            ... })
        """
        return await self._action_handler.handle(action_data)

    async def run(self):
        """Run the MCP server using stdio transport.

        This is the main entry point for running the server.
        It handles the MCP protocol over stdin/stdout.

        Example:
            >>> server = MCPUIServer()
            >>> # Register tools...
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
        pass


def ui_tool(
    name: str | None = None,
    description: str = "",
    template_uri: str = "",
    enable_apps_sdk: bool = True,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Standalone decorator for marking functions as UI tools.

    Use this decorator to mark functions that return UI resources.
    The decorated function can later be registered with a server.

    Args:
        name: Tool name (defaults to function name).
        description: Tool description.
        template_uri: Default template URI.
        enable_apps_sdk: Enable Apps SDK adapter.

    Returns:
        Decorator function.

    Example:
        >>> @ui_tool(description="Display search results")
        ... async def search_products(query: str) -> UIResource:
        ...     results = await search(query)
        ...     return create_results_ui(results)
        >>>
        >>> # Later, register with server
        >>> server.register_ui_tool(
        ...     name="search",
        ...     handler=search_products,
        ... )
    """
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        # Store metadata on function for later registration
        func._ui_tool_metadata = {  # type: ignore
            "name": name or func.__name__,
            "description": description or func.__doc__ or "",
            "template_uri": template_uri,
            "enable_apps_sdk": enable_apps_sdk,
        }
        return func
    return decorator


class OpenAIAppsWidgetAdapter:
    """Adapter for converting openai_apps widgets to MCP-UI format.

    Provides compatibility layer between existing openai_apps
    widget implementations and the MCP-UI specification.

    Example:
        >>> adapter = OpenAIAppsWidgetAdapter()
        >>>
        >>> # Convert widget payload to MCP-UI resource
        >>> widget_data = ListViewPayload(items=[...]).to_structured_content()
        >>> ui_resource = adapter.convert_widget_to_resource(
        ...     widget_data,
        ...     template_uri="widget://list-view",
        ... )
    """

    def __init__(self, enable_apps_sdk: bool = True):
        """Initialize adapter.

        Args:
            enable_apps_sdk: Enable Apps SDK adapter for converted resources.
        """
        self.enable_apps_sdk = enable_apps_sdk

    def convert_widget_to_resource(
        self,
        widget_data: dict[str, Any],
        template_uri: str = "",
        uri: str | None = None,
    ) -> UIResource:
        """Convert widget structured content to MCP-UI resource.

        Args:
            widget_data: Widget structured content from openai_apps.
            template_uri: Template URI for rendering.
            uri: Optional resource URI.

        Returns:
            UIResource compatible with MCP-UI.
        """
        # Extract _meta if present
        meta = widget_data.pop("_meta", {})
        template = template_uri or meta.get("openai/outputTemplate", "")

        # Determine if we should render HTML or return structured
        if template.startswith("widget://"):
            # Need to render with a template
            html = self._render_widget_html(widget_data, template)
            return RawHtmlResource.create(
                html=html,
                uri=uri,
                enable_apps_sdk=self.enable_apps_sdk,
            )
        else:
            # Return as structured content
            if not uri:
                uri = f"ui://widget/{uuid.uuid4().hex[:8]}"

            content_json = json.dumps({
                **widget_data,
                "_meta": {
                    **meta,
                    "mcp-ui/template": template,
                },
            })

            return UIResource(
                uri=uri,
                mime_type=MimeType.TEXT_HTML,
                content=UIResourceContent.from_text(
                    f'<script type="application/json" id="widget-data">{content_json}</script>'
                ),
            )

    def _render_widget_html(
        self,
        data: dict[str, Any],
        template_uri: str,
    ) -> str:
        """Render widget data with template.

        Args:
            data: Widget data.
            template_uri: Template URI.

        Returns:
            Rendered HTML.
        """
        widget_type = template_uri.replace("widget://", "")
        data_json = json.dumps(data)

        # Generate simple HTML wrapper that renders the data
        return f"""
<!DOCTYPE html>
<html>
<head>
<style>
:root {{
  --bg-color: #ffffff;
  --text-color: #1a1a1a;
  --border-color: #e5e5e5;
  --primary-color: #3b82f6;
}}
[data-theme="dark"] {{
  --bg-color: #1f2937;
  --text-color: #f9fafb;
  --border-color: #374151;
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-color);
  color: var(--text-color);
  padding: 16px;
  margin: 0;
}}
</style>
</head>
<body>
<div id="widget-root"></div>
<script>
const widgetData = {data_json};
const widgetType = '{widget_type}';

// Apply theme
document.body.dataset.theme = window.mcpui?.getTheme() || window.openai?.theme || 'light';

// Simple widget renderers
const renderers = {{
  'list-view': (data) => {{
    const items = data.items || [];
    if (!items.length) return '<p style="text-align:center;opacity:0.6">' + (data.emptyMessage || 'No items') + '</p>';
    return '<div>' + items.map(item => `
      <div style="padding:12px;border-bottom:1px solid var(--border-color)">
        <div style="font-weight:500">${{item.title || ''}}</div>
        <div style="font-size:13px;opacity:0.7">${{item.subtitle || ''}}</div>
      </div>
    `).join('') + '</div>';
  }},
  'status': (data) => {{
    const icons = {{ info: 'i', success: '\\u2713', warning: '!', error: '\\u2717', loading: '...' }};
    return `
      <div style="text-align:center;padding:20px">
        <div style="font-size:40px">${{icons[data.status] || icons.info}}</div>
        <div style="font-size:18px;font-weight:600;margin-top:12px">${{data.title || ''}}</div>
        <div style="opacity:0.7;margin-top:8px">${{data.message || ''}}</div>
      </div>
    `;
  }},
  'form': (data) => {{
    const fields = (data.fields || []).map(f => `
      <div style="margin-bottom:16px">
        <label style="display:block;font-weight:500;margin-bottom:6px">${{f.label || f.name}}</label>
        <input type="${{f.type || 'text'}}" name="${{f.name}}" placeholder="${{f.placeholder || ''}}"
          style="width:100%;padding:10px;border:1px solid var(--border-color);border-radius:6px">
      </div>
    `).join('');
    return `
      <form>
        ${{data.title ? '<div style="font-size:18px;font-weight:600;margin-bottom:16px">' + data.title + '</div>' : ''}}
        ${{fields}}
        <button type="submit" style="background:var(--primary-color);color:white;padding:10px 20px;border:none;border-radius:6px">
          ${{data.submitLabel || 'Submit'}}
        </button>
      </form>
    `;
  }},
}};

const renderer = renderers[widgetType] || ((d) => '<pre>' + JSON.stringify(d, null, 2) + '</pre>');
document.getElementById('widget-root').innerHTML = renderer(widgetData);
</script>
</body>
</html>
"""

    def adapt_widget_handler(
        self,
        handler: Callable[..., Awaitable[dict[str, Any]]],
        template_uri: str = "",
    ) -> UIToolHandler:
        """Adapt an openai_apps widget handler to MCP-UI format.

        Args:
            handler: Original widget handler returning structured content.
            template_uri: Template URI for rendering.

        Returns:
            Adapted handler returning UIResource.

        Example:
            >>> @widget_tool(category=WidgetCategory.LIST)
            ... async def products(category: str) -> dict:
            ...     return {"items": await get_products(category)}
            >>>
            >>> adapted = adapter.adapt_widget_handler(products, "widget://list-view")
            >>> server.register_ui_tool("products", adapted)
        """
        @functools.wraps(handler)
        async def adapted_handler(**kwargs: Any) -> UIResource:
            result = await handler(**kwargs)
            return self.convert_widget_to_resource(result, template_uri)

        return adapted_handler


def create_mcp_ui_server(
    name: str = "mcp-ui-server",
    version: str = "1.0.0",
    enable_apps_sdk: bool = True,
) -> MCPUIServer:
    """Factory function for creating an MCP-UI server.

    Args:
        name: Server name.
        version: Server version.
        enable_apps_sdk: Enable Apps SDK adapter.

    Returns:
        Configured MCPUIServer instance.

    Example:
        >>> server = create_mcp_ui_server("my-server")
        >>>
        >>> @server.ui_tool("hello")
        ... async def hello(name: str = "World") -> UIResource:
        ...     html = f"<h1>Hello, {name}!</h1>"
        ...     return RawHtmlResource.create(html)
        >>>
        >>> asyncio.run(server.run())
    """
    return MCPUIServer(
        name=name,
        version=version,
        enable_apps_sdk=enable_apps_sdk,
    )


def register_openai_apps_widgets(
    server: MCPUIServer,
    widgets: dict[str, tuple[UIToolHandler, str]],
) -> None:
    """Register multiple openai_apps widgets with an MCP-UI server.

    Args:
        server: MCP-UI server instance.
        widgets: Dictionary of {name: (handler, template_uri)}.

    Example:
        >>> widgets = {
        ...     "products": (product_handler, "widget://list-view"),
        ...     "status": (status_handler, "widget://status"),
        ... }
        >>> register_openai_apps_widgets(server, widgets)
    """
    adapter = OpenAIAppsWidgetAdapter(enable_apps_sdk=server.enable_apps_sdk)

    for name, (handler, template_uri) in widgets.items():
        adapted = adapter.adapt_widget_handler(handler, template_uri)
        server.register_ui_tool(
            name=name,
            handler=adapted,
            template_uri=template_uri,
        )
