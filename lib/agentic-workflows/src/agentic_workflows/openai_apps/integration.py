"""Integration adapters for OpenAI Apps SDK with agentic_workflows.

This module provides adapters to expose agentic_workflows tools as
OpenAI Apps SDK widgets:
- OpenAIAppsAdapter: Convert tools to widget-enabled endpoints
- ToolWidgetMapper: Map tool outputs to widget payloads
- WidgetRouter: Route tool calls to appropriate widget handlers

Enables seamless integration between existing agentic_workflows
functionality and the OpenAI Apps SDK widget system.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from .components import (
    CarouselItem,
    CarouselPayload,
    ChartDataset,
    ChartPayload,
    ChartType,
    FormField,
    FormFieldType,
    FormPayload,
    ListItem,
    ListViewPayload,
    StatusPayload,
    StatusType,
)
from .state import InMemoryStateStore, StateStore, WidgetSessionManager
from .widget_server import WidgetHandler, WidgetServer
from .widget_types import (
    StructuredContent,
    WidgetState,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WidgetCategory(str, Enum):
    """Categories for automatic widget type detection."""

    LIST = "list"
    STATUS = "status"
    CHART = "chart"
    FORM = "form"
    CAROUSEL = "carousel"
    CUSTOM = "custom"


@dataclass
class ToolOutputMapping:
    """Mapping configuration for tool output to widget.

    Attributes:
        tool_name: Name of the tool
        widget_category: Category of widget to use
        template_uri: Custom template URI (overrides category default)
        field_mappings: Map tool output fields to widget fields
        transformer: Custom transformation function
        stateful: Whether to maintain state
    """

    tool_name: str
    widget_category: WidgetCategory = WidgetCategory.CUSTOM
    template_uri: str = ""
    field_mappings: dict[str, str] = field(default_factory=dict)
    transformer: Callable[[dict], StructuredContent] | None = None
    stateful: bool = False


class ToolWidgetMapper:
    """Maps tool outputs to widget payloads.

    Provides automatic and configurable mapping from tool outputs
    to appropriate widget payloads based on output structure.

    Example:
        >>> mapper = ToolWidgetMapper()
        >>> mapper.register_mapping(ToolOutputMapping(
        ...     tool_name="get_products",
        ...     widget_category=WidgetCategory.LIST,
        ...     field_mappings={"items": "products", "title": "category"}
        ... ))
        >>> widget_content = mapper.map_output("get_products", tool_output)
    """

    # Default template URIs for each category
    CATEGORY_TEMPLATES = {
        WidgetCategory.LIST: "widget://list-view",
        WidgetCategory.STATUS: "widget://status",
        WidgetCategory.CHART: "widget://chart",
        WidgetCategory.FORM: "widget://form",
        WidgetCategory.CAROUSEL: "widget://carousel",
    }

    def __init__(self):
        """Initialize mapper with default mappings."""
        self._mappings: dict[str, ToolOutputMapping] = {}
        self._auto_detect = True

    def register_mapping(self, mapping: ToolOutputMapping) -> None:
        """Register a tool output mapping.

        Args:
            mapping: Tool output mapping configuration.
        """
        self._mappings[mapping.tool_name] = mapping
        logger.debug(f"Registered mapping: {mapping.tool_name} -> {mapping.widget_category}")

    def enable_auto_detect(self, enabled: bool = True) -> None:
        """Enable/disable automatic widget type detection.

        Args:
            enabled: Whether to enable auto-detection.
        """
        self._auto_detect = enabled

    def get_template_uri(self, tool_name: str) -> str:
        """Get template URI for a tool.

        Args:
            tool_name: Tool name.

        Returns:
            Template URI string.
        """
        mapping = self._mappings.get(tool_name)
        if mapping:
            if mapping.template_uri:
                return mapping.template_uri
            return self.CATEGORY_TEMPLATES.get(mapping.widget_category, "widget://status")
        return "widget://status"

    def map_output(
        self,
        tool_name: str,
        output: dict[str, Any],
    ) -> StructuredContent:
        """Map tool output to widget content.

        Args:
            tool_name: Name of the tool.
            output: Tool output dictionary.

        Returns:
            Widget-compatible structured content.
        """
        mapping = self._mappings.get(tool_name)

        if mapping and mapping.transformer:
            # Use custom transformer
            return mapping.transformer(output)

        if mapping:
            # Apply field mappings
            mapped_output = self._apply_field_mappings(output, mapping.field_mappings)
            return self._create_payload(mapping.widget_category, mapped_output)

        if self._auto_detect:
            # Auto-detect widget type
            category = self._detect_category(output)
            return self._create_payload(category, output)

        # Default: return as-is
        return output

    def _apply_field_mappings(
        self,
        output: dict[str, Any],
        mappings: dict[str, str],
    ) -> dict[str, Any]:
        """Apply field mappings to output."""
        result = output.copy()
        for widget_field, output_field in mappings.items():
            if output_field in output:
                result[widget_field] = output[output_field]
        return result

    def _detect_category(self, output: dict[str, Any]) -> WidgetCategory:
        """Detect widget category from output structure."""
        # Check for list-like outputs
        if "items" in output and isinstance(output["items"], list):
            return WidgetCategory.LIST
        if "results" in output and isinstance(output["results"], list):
            return WidgetCategory.LIST

        # Check for status-like outputs
        if "status" in output or "progress" in output:
            return WidgetCategory.STATUS
        if "state" in output or "message" in output:
            return WidgetCategory.STATUS

        # Check for chart-like outputs
        if "datasets" in output or "data" in output and "labels" in output:
            return WidgetCategory.CHART
        if "chart" in output or "chartType" in output:
            return WidgetCategory.CHART

        # Check for form-like outputs
        if "fields" in output and isinstance(output["fields"], list):
            return WidgetCategory.FORM

        # Check for carousel-like outputs
        if "images" in output or "slides" in output:
            return WidgetCategory.CAROUSEL

        # Default to status
        return WidgetCategory.STATUS

    def _create_payload(
        self,
        category: WidgetCategory,
        output: dict[str, Any],
    ) -> StructuredContent:
        """Create widget payload from output."""
        if category == WidgetCategory.LIST:
            return self._create_list_payload(output)
        elif category == WidgetCategory.STATUS:
            return self._create_status_payload(output)
        elif category == WidgetCategory.CHART:
            return self._create_chart_payload(output)
        elif category == WidgetCategory.FORM:
            return self._create_form_payload(output)
        elif category == WidgetCategory.CAROUSEL:
            return self._create_carousel_payload(output)
        else:
            return output

    def _create_list_payload(self, output: dict) -> StructuredContent:
        """Create list view payload."""
        items_key = "items" if "items" in output else "results"
        raw_items = output.get(items_key, [])

        items = []
        for item in raw_items:
            if isinstance(item, dict):
                items.append(
                    ListItem(
                        id=str(item.get("id", id(item))),
                        title=item.get("title") or item.get("name") or str(item),
                        subtitle=item.get("subtitle") or item.get("description", "")[:100],
                        description=item.get("description", ""),
                        image_url=item.get("image_url") or item.get("imageUrl", ""),
                        badge=item.get("badge", ""),
                    )
                )
            else:
                items.append(ListItem(id=str(id(item)), title=str(item)))

        payload = ListViewPayload(
            items=items,
            title=output.get("title", ""),
            total_count=output.get("total_count") or len(items),
        )
        return payload.to_structured_content()

    def _create_status_payload(self, output: dict) -> StructuredContent:
        """Create status payload."""
        status_map = {
            "success": StatusType.SUCCESS,
            "error": StatusType.ERROR,
            "warning": StatusType.WARNING,
            "loading": StatusType.LOADING,
            "pending": StatusType.LOADING,
            "completed": StatusType.SUCCESS,
            "failed": StatusType.ERROR,
        }

        status_str = output.get("status", "info")
        status = status_map.get(status_str.lower(), StatusType.INFO)

        payload = StatusPayload(
            title=output.get("title") or output.get("message", "Status"),
            status=status,
            message=output.get("message") or output.get("description", ""),
            progress=output.get("progress"),
        )
        return payload.to_structured_content()

    def _create_chart_payload(self, output: dict) -> StructuredContent:
        """Create chart payload."""
        chart_type_map = {
            "line": ChartType.LINE,
            "bar": ChartType.BAR,
            "pie": ChartType.PIE,
            "area": ChartType.AREA,
        }

        chart_type_str = output.get("chartType") or output.get("type", "line")
        chart_type = chart_type_map.get(chart_type_str.lower(), ChartType.LINE)

        datasets = []
        raw_datasets = output.get("datasets", [])
        if not raw_datasets and "data" in output:
            # Single dataset
            raw_datasets = [{"label": "Data", "data": output["data"]}]

        for ds in raw_datasets:
            datasets.append(
                ChartDataset(
                    label=ds.get("label", ""),
                    data=ds.get("data", []),
                    color=ds.get("color", ""),
                )
            )

        payload = ChartPayload(
            chart_type=chart_type,
            datasets=datasets,
            labels=output.get("labels", []),
            title=output.get("title", ""),
        )
        return payload.to_structured_content()

    def _create_form_payload(self, output: dict) -> StructuredContent:
        """Create form payload."""
        fields = []
        for f in output.get("fields", []):
            field_type_map = {
                "text": FormFieldType.TEXT,
                "email": FormFieldType.EMAIL,
                "password": FormFieldType.PASSWORD,
                "number": FormFieldType.NUMBER,
                "textarea": FormFieldType.TEXTAREA,
                "select": FormFieldType.SELECT,
                "checkbox": FormFieldType.CHECKBOX,
            }
            field_type = field_type_map.get(f.get("type", "text").lower(), FormFieldType.TEXT)
            fields.append(
                FormField(
                    name=f.get("name", ""),
                    type=field_type,
                    label=f.get("label", ""),
                    placeholder=f.get("placeholder", ""),
                    required=f.get("required", False),
                    options=f.get("options", []),
                )
            )

        payload = FormPayload(
            fields=fields,
            title=output.get("title", ""),
            description=output.get("description", ""),
            submit_label=output.get("submitLabel", "Submit"),
        )
        return payload.to_structured_content()

    def _create_carousel_payload(self, output: dict) -> StructuredContent:
        """Create carousel payload."""
        items_key = "images" if "images" in output else "slides"
        raw_items = output.get(items_key, [])

        items = []
        for item in raw_items:
            if isinstance(item, dict):
                items.append(
                    CarouselItem(
                        id=str(item.get("id", id(item))),
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        image_url=item.get("url") or item.get("image_url", ""),
                    )
                )
            elif isinstance(item, str):
                items.append(
                    CarouselItem(
                        id=str(id(item)),
                        image_url=item,
                    )
                )

        payload = CarouselPayload(
            items=items,
            title=output.get("title", ""),
        )
        return payload.to_structured_content()


class WidgetRouter:
    """Routes tool calls to widget handlers.

    Manages routing of tool calls to their corresponding widget
    handlers, with support for middleware and error handling.

    Example:
        >>> router = WidgetRouter()
        >>> router.add_route("products", products_handler)
        >>> router.add_middleware(logging_middleware)
        >>> result = await router.route("products", {"category": "electronics"})
    """

    def __init__(self):
        """Initialize router."""
        self._routes: dict[str, WidgetHandler] = {}
        self._middleware: list[Callable] = []
        self._error_handler: Callable | None = None

    def add_route(
        self,
        tool_name: str,
        handler: WidgetHandler,
    ) -> None:
        """Add a route for a tool.

        Args:
            tool_name: Tool name.
            handler: Widget handler function.
        """
        self._routes[tool_name] = handler
        logger.debug(f"Added route: {tool_name}")

    def add_middleware(
        self,
        middleware: Callable[[str, dict, WidgetHandler], Awaitable[StructuredContent]],
    ) -> None:
        """Add middleware for request processing.

        Args:
            middleware: Middleware function.
        """
        self._middleware.append(middleware)

    def set_error_handler(
        self,
        handler: Callable[[str, Exception], StructuredContent],
    ) -> None:
        """Set error handler for route errors.

        Args:
            handler: Error handler function.
        """
        self._error_handler = handler

    async def route(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        state: WidgetState | None = None,
    ) -> StructuredContent:
        """Route a tool call to its handler.

        Args:
            tool_name: Tool name.
            arguments: Tool arguments.
            state: Optional widget state.

        Returns:
            Widget structured content.
        """
        handler = self._routes.get(tool_name)
        if handler is None:
            error_content: StructuredContent = {
                "error": f"No handler for tool: {tool_name}",
                "type": "status",
                "status": "error",
            }
            return error_content

        try:
            # Apply middleware chain
            if self._middleware:

                async def chain(h: WidgetHandler) -> StructuredContent:
                    return await h(arguments, state)

                for mw in reversed(self._middleware):
                    original_chain = chain

                    async def chain(h: WidgetHandler = handler) -> StructuredContent:
                        return await mw(tool_name, arguments, original_chain)

                return await chain(handler)

            # Direct call
            return await handler(arguments, state)

        except Exception as e:
            logger.exception(f"Route error: {tool_name}")
            if self._error_handler:
                return self._error_handler(tool_name, e)
            error_result: StructuredContent = {
                "error": str(e),
                "type": "status",
                "status": "error",
                "title": "Error",
            }
            return error_result


class OpenAIAppsAdapter:
    """Adapter to expose agentic_workflows tools as OpenAI Apps SDK widgets.

    Provides integration between existing agentic_workflows tool
    infrastructure and the OpenAI Apps SDK widget system.

    Example:
        >>> from agentic_workflows import get_skill_registry
        >>>
        >>> adapter = OpenAIAppsAdapter()
        >>> adapter.register_tool("get_products", get_products_handler)
        >>> adapter.set_widget_mapping("get_products", WidgetCategory.LIST)
        >>>
        >>> # Create server with adapter
        >>> server = adapter.create_server()
        >>> await server.run()
    """

    def __init__(
        self,
        server_name: str = "agentic-apps-server",
        state_store: StateStore | None = None,
    ):
        """Initialize adapter.

        Args:
            server_name: MCP server name.
            state_store: Optional state store for sessions.
        """
        self.server_name = server_name
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: dict[str, dict] = {}
        self._tool_descriptions: dict[str, str] = {}
        self._mapper = ToolWidgetMapper()
        self._router = WidgetRouter()
        self._state_store = state_store or InMemoryStateStore()
        self._session_manager = WidgetSessionManager(store=self._state_store)

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool to expose as widget.

        Args:
            name: Tool name.
            handler: Tool handler (sync or async).
            description: Tool description.
            input_schema: JSON Schema for inputs.
        """
        self._tools[name] = handler
        self._tool_descriptions[name] = description or handler.__doc__ or ""
        self._tool_schemas[name] = input_schema or {}

        # Create widget handler wrapper
        async def widget_handler(
            args: dict[str, Any],
            state: WidgetState | None,
            _handler: Callable = handler,
            _name: str = name,
        ) -> StructuredContent:
            # Call original tool
            if asyncio.iscoroutinefunction(_handler):
                result = await _handler(**args)
            else:
                result = _handler(**args)

            # Map to widget payload
            if isinstance(result, dict):
                return self._mapper.map_output(_name, result)
            else:
                return {"result": result, "type": "status", "status": "success"}

        self._router.add_route(name, widget_handler)
        logger.info(f"Registered tool: {name}")

    def set_widget_mapping(
        self,
        tool_name: str,
        category: WidgetCategory,
        field_mappings: dict[str, str] | None = None,
        template_uri: str = "",
        stateful: bool = False,
    ) -> None:
        """Set widget mapping for a tool.

        Args:
            tool_name: Tool name.
            category: Widget category.
            field_mappings: Field mappings (widget_field -> output_field).
            template_uri: Custom template URI.
            stateful: Whether widget is stateful.
        """
        self._mapper.register_mapping(
            ToolOutputMapping(
                tool_name=tool_name,
                widget_category=category,
                field_mappings=field_mappings or {},
                template_uri=template_uri,
                stateful=stateful,
            )
        )

    def set_custom_transformer(
        self,
        tool_name: str,
        transformer: Callable[[dict], StructuredContent],
    ) -> None:
        """Set custom output transformer for a tool.

        Args:
            tool_name: Tool name.
            transformer: Custom transformation function.
        """
        existing = self._mapper._mappings.get(tool_name)
        if existing:
            existing.transformer = transformer
        else:
            self._mapper.register_mapping(
                ToolOutputMapping(
                    tool_name=tool_name,
                    transformer=transformer,
                )
            )

    def create_server(self) -> WidgetServer:
        """Create a WidgetServer with registered tools.

        Returns:
            Configured WidgetServer instance.
        """
        server = WidgetServer(
            name=self.server_name,
            session_manager=self._session_manager,
        )

        # Register all tools as widgets
        for name in self._tools:
            mapping = self._mapper._mappings.get(name)
            template_uri = self._mapper.get_template_uri(name)

            async def handler(
                args: dict[str, Any],
                state: WidgetState | None,
                _name: str = name,
            ) -> StructuredContent:
                return await self._router.route(_name, args, state)

            server.register_widget(
                name=name,
                template_uri=template_uri,
                handler=handler,
                description=self._tool_descriptions.get(name, ""),
                input_schema=self._tool_schemas.get(name),
                stateful=mapping.stateful if mapping else False,
            )

        return server

    async def close(self) -> None:
        """Close adapter resources."""
        await self._session_manager.close()


def create_apps_adapter(
    server_name: str = "agentic-apps-server",
    state_store: StateStore | None = None,
) -> OpenAIAppsAdapter:
    """Factory function for creating OpenAI Apps adapter.

    Args:
        server_name: MCP server name.
        state_store: Optional state store.

    Returns:
        Configured OpenAIAppsAdapter instance.

    Example:
        >>> adapter = create_apps_adapter("my-server")
        >>> adapter.register_tool("search", search_handler)
        >>> adapter.set_widget_mapping("search", WidgetCategory.LIST)
        >>> server = adapter.create_server()
        >>> await server.run()
    """
    return OpenAIAppsAdapter(
        server_name=server_name,
        state_store=state_store,
    )


# Convenience decorators for skill integration


def widget_tool(
    name: str | None = None,
    category: WidgetCategory = WidgetCategory.CUSTOM,
    template_uri: str = "",
    stateful: bool = False,
    description: str = "",
):
    """Decorator to mark a function as a widget tool.

    Args:
        name: Tool name (defaults to function name).
        category: Widget category for auto-mapping.
        template_uri: Custom template URI.
        stateful: Whether widget maintains state.
        description: Tool description.

    Returns:
        Decorator function.

    Example:
        >>> @widget_tool(category=WidgetCategory.LIST)
        ... async def get_products(category: str) -> dict:
        ...     products = await fetch_products(category)
        ...     return {"items": products}
    """

    def decorator(func: Callable) -> Callable:
        # Store metadata on function
        func._widget_metadata = {
            "name": name or func.__name__,
            "category": category,
            "template_uri": template_uri,
            "stateful": stateful,
            "description": description or func.__doc__ or "",
        }
        return func

    return decorator


def register_widget_tools(
    adapter: OpenAIAppsAdapter,
    *tools: Callable,
) -> None:
    """Register multiple widget tools with an adapter.

    Args:
        adapter: OpenAI Apps adapter.
        tools: Tool functions decorated with @widget_tool.

    Example:
        >>> @widget_tool(category=WidgetCategory.LIST)
        ... async def products(): ...
        >>>
        >>> @widget_tool(category=WidgetCategory.STATUS)
        ... async def status(): ...
        >>>
        >>> adapter = create_apps_adapter()
        >>> register_widget_tools(adapter, products, status)
    """
    for tool in tools:
        metadata = getattr(tool, "_widget_metadata", None)
        if metadata is None:
            # Not decorated, register with defaults
            adapter.register_tool(
                name=tool.__name__,
                handler=tool,
            )
        else:
            adapter.register_tool(
                name=metadata["name"],
                handler=tool,
                description=metadata["description"],
            )
            if metadata["category"] != WidgetCategory.CUSTOM or metadata["template_uri"]:
                adapter.set_widget_mapping(
                    tool_name=metadata["name"],
                    category=metadata["category"],
                    template_uri=metadata["template_uri"],
                    stateful=metadata["stateful"],
                )
