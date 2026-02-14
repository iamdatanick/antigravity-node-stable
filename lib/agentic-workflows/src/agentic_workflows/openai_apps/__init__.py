"""OpenAI Apps SDK Integration Module for Agentic Workflows.

This module provides integration with the OpenAI Apps SDK for creating
rich UI widgets that render alongside assistant messages. Based on the
Model Context Protocol (MCP) patterns.

Features:
---------
- **Widget Types**: Data structures for widget templates, payloads, and state
- **Widget Server**: MCP server for exposing widgets via tools and resources
- **Components**: Pre-built widget components (ListView, Carousel, Form, Status, Chart)
- **State Management**: Session state tracking with multiple backends
- **Integration**: Adapters for exposing agentic_workflows tools as widgets

Quick Start:
------------
```python
from agentic_workflows.openai_apps import (
    create_apps_server,
    ListViewPayload,
    ListItem,
    WidgetMeta,
)

# Create a widget server
server = create_apps_server("my-widgets")

# Register a widget tool
@server.widget("products", "widget://list-view")
async def products(args, state):
    items = [
        ListItem(id="1", title="Product A", subtitle="$29.99"),
        ListItem(id="2", title="Product B", subtitle="$49.99"),
    ]
    return ListViewPayload(items=items, title="Products").to_structured_content()

# Run the server
await server.run()
```

Integration with Agentic Workflows:
-----------------------------------
```python
from agentic_workflows.openai_apps import (
    OpenAIAppsAdapter,
    WidgetCategory,
    widget_tool,
)

# Create adapter
adapter = OpenAIAppsAdapter("my-server")

# Register existing tool
@widget_tool(category=WidgetCategory.LIST)
async def search_products(query: str) -> dict:
    results = await product_search(query)
    return {"items": results}

adapter.register_tool("search", search_products)

# Create and run server
server = adapter.create_server()
await server.run()
```

State Management:
-----------------
```python
from agentic_workflows.openai_apps import (
    WidgetSessionManager,
    RedisStateStore,
    StateStoreConfig,
)

# Production setup with Redis
config = StateStoreConfig(prefix="myapp:", default_ttl=7200)
store = RedisStateStore(config, host="redis.example.com")

manager = WidgetSessionManager(store=store)

# Create session
session_id = await manager.create_session(
    initial_state={"cart": {"items": []}},
    ttl=3600,
)

# Update state
await manager.merge_state(session_id, {"cart": {"total": 99.99}})
```
"""

__version__ = "1.0.0"

# Widget Types
# Widget Components
from .components import (
    CAROUSEL_TEMPLATE,
    CHART_TEMPLATE,
    FORM_TEMPLATE,
    LIST_VIEW_TEMPLATE,
    STATUS_TEMPLATE,
    # Templates
    WIDGET_TEMPLATES,
    CarouselItem,
    CarouselItemType,
    # Carousel
    CarouselPayload,
    ChartDataset,
    # Chart
    ChartPayload,
    ChartType,
    FormField,
    FormFieldType,
    # Form
    FormPayload,
    ListItem,
    ListStyle,
    # List View
    ListViewPayload,
    # Status
    StatusPayload,
    StatusStep,
    StatusType,
    get_template,
    register_template,
)

# Integration
from .integration import (
    # Adapter
    OpenAIAppsAdapter,
    ToolOutputMapping,
    # Mapper
    ToolWidgetMapper,
    WidgetCategory,
    # Router
    WidgetRouter,
    # Factory
    create_apps_adapter,
    register_widget_tools,
    # Decorators
    widget_tool,
)

# State Management
from .state import (
    D1StateStore,
    InMemoryStateStore,
    RedisStateStore,
    # Stores
    StateStore,
    StateStoreConfig,
    # Managers
    WidgetSessionManager,
)

# Widget Server
from .widget_server import (
    WidgetHandler,
    WidgetRegistration,
    WidgetServer,
    WidgetServerBuilder,
    create_widget_server,
)
from .widget_types import (
    # Type aliases
    StructuredContent,
    # Enums
    WidgetDisplayMode,
    WidgetMeta,
    WidgetMetaDict,
    WidgetPayload,
    WidgetState,
    # Core types
    WidgetTemplate,
    WidgetTheme,
)


def create_apps_server(
    name: str = "openai-apps-server",
    version: str = "1.0.0",
    state_store: StateStore | None = None,
    enable_cleanup: bool = True,
) -> WidgetServer:
    """Factory function to create an OpenAI Apps SDK-compatible widget server.

    This is the main entry point for creating a widget server that integrates
    with the OpenAI Apps SDK. The server exposes widgets as MCP tools and
    resources.

    Args:
        name: Server name for MCP identification.
        version: Server version string.
        state_store: Custom state store for widget sessions (default: InMemoryStateStore).
        enable_cleanup: Enable background cleanup of expired sessions.

    Returns:
        Configured WidgetServer ready for widget registration.

    Example:
        Basic usage:
        ```python
        from agentic_workflows.openai_apps import create_apps_server

        server = create_apps_server("my-widgets")

        @server.widget("hello", "widget://status")
        async def hello(args, state):
            return {
                "type": "status",
                "status": "success",
                "title": "Hello!",
                "message": f"Hello, {args.get('name', 'World')}!",
            }

        # Run with: python -m agentic_workflows.openai_apps.server
        if __name__ == "__main__":
            import asyncio
            asyncio.run(server.run())
        ```

        With custom state store:
        ```python
        from agentic_workflows.openai_apps import (
            create_apps_server,
            RedisStateStore,
            StateStoreConfig,
        )

        config = StateStoreConfig(prefix="widgets:", default_ttl=3600)
        store = RedisStateStore(config, host="localhost")

        server = create_apps_server(
            name="stateful-widgets",
            state_store=store,
        )

        @server.widget("cart", "widget://form", stateful=True)
        async def shopping_cart(args, state):
            items = state.get("items", []) if state else []
            # ... handle cart operations
            return FormPayload(...).to_structured_content()
        ```

        Using the builder pattern:
        ```python
        from agentic_workflows.openai_apps import WidgetServerBuilder

        server = (
            WidgetServerBuilder("my-server")
            .with_widget("list", "widget://list-view", list_handler)
            .with_stateful_widget("form", "widget://form", form_handler)
            .build()
        )
        ```
    """
    store = state_store or InMemoryStateStore()
    session_manager = WidgetSessionManager(store=store)

    # Start cleanup task for in-memory store
    if enable_cleanup and isinstance(store, InMemoryStateStore):
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(store.start_cleanup())
        except RuntimeError:
            pass  # No event loop, will start on first use

    return WidgetServer(
        name=name,
        version=version,
        session_manager=session_manager,
    )


__all__ = [
    # Version
    "__version__",
    # Factory
    "create_apps_server",
    # Widget Types
    "WidgetTemplate",
    "WidgetPayload",
    "WidgetMeta",
    "WidgetState",
    "WidgetDisplayMode",
    "WidgetTheme",
    "StructuredContent",
    "WidgetMetaDict",
    # State Management
    "WidgetSessionManager",
    "StateStore",
    "StateStoreConfig",
    "InMemoryStateStore",
    "RedisStateStore",
    "D1StateStore",
    # Components - List
    "ListViewPayload",
    "ListItem",
    "ListStyle",
    # Components - Carousel
    "CarouselPayload",
    "CarouselItem",
    "CarouselItemType",
    # Components - Form
    "FormPayload",
    "FormField",
    "FormFieldType",
    # Components - Status
    "StatusPayload",
    "StatusStep",
    "StatusType",
    # Components - Chart
    "ChartPayload",
    "ChartDataset",
    "ChartType",
    # Templates
    "WIDGET_TEMPLATES",
    "LIST_VIEW_TEMPLATE",
    "CAROUSEL_TEMPLATE",
    "FORM_TEMPLATE",
    "STATUS_TEMPLATE",
    "CHART_TEMPLATE",
    "get_template",
    "register_template",
    # Server
    "WidgetServer",
    "WidgetServerBuilder",
    "WidgetRegistration",
    "WidgetHandler",
    "create_widget_server",
    # Integration
    "OpenAIAppsAdapter",
    "ToolWidgetMapper",
    "ToolOutputMapping",
    "WidgetCategory",
    "WidgetRouter",
    "create_apps_adapter",
    "widget_tool",
    "register_widget_tools",
]
