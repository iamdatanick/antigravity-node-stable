"""MCP-UI Integration Module for Agentic Workflows.

This module provides integration with the MCP-UI SDK for creating
rich UI resources in MCP tools. Based on the specification at
https://github.com/MCP-UI-Org/mcp-ui.

Features:
---------
- **UI Types**: Data structures for UI resources, actions, and metadata
- **Resource Creation**: Factory functions for creating UI resources
- **Action Handlers**: Processing UI actions from components
- **Component Library**: Built-in HTML components for common UI patterns
- **MCP Server Integration**: MCPUIServer with @ui_tool decorator

Quick Start:
------------
```python
from agentic_workflows.mcp_ui import (
    MCPUIServer,
    RawHtmlResource,
    UIMetadata,
    FrameSize,
)

# Create an MCP-UI server
server = MCPUIServer(name="my-ui-server")

# Register a UI tool
@server.ui_tool("greeting")
async def greeting(name: str = "World") -> UIResource:
    '''Display a greeting message.'''
    html = f'''
    <div style="text-align: center; padding: 20px;">
        <h1>Hello, {name}!</h1>
        <p>Welcome to MCP-UI</p>
    </div>
    '''
    return RawHtmlResource.create(
        html=html,
        metadata=UIMetadata(
            preferred_frame_size=FrameSize(width=400, height=200),
            accessible_description=f"Greeting for {name}",
        ),
    )

# Run the server
if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run())
```

Creating Different Resource Types:
----------------------------------
```python
from agentic_workflows.mcp_ui import (
    create_ui_resource,
    RawHtmlResource,
    ExternalUrlResource,
    RemoteDomResource,
)

# Raw HTML (rendered inline via srcDoc)
html_resource = RawHtmlResource.create(
    html="<p>Hello World!</p>",
    uri="ui://hello/1",
)

# External URL (embedded via iframe src)
url_resource = ExternalUrlResource.create(
    url="https://example.com/widget",
    uri="ui://external/1",
)

# Remote DOM (sandboxed script execution)
dom_resource = RemoteDomResource.create(
    script="document.body.innerHTML = '<h1>Dynamic!</h1>';",
    framework="react",
    uri="ui://dynamic/1",
)

# Using create_ui_resource (mcp-ui-server API)
resource = create_ui_resource({
    "uri": "ui://greeting/1",
    "content": {"type": "rawHtml", "htmlString": "<p>Hello!</p>"},
    "encoding": "text",
})
```

Handling UI Actions:
--------------------
```python
from agentic_workflows.mcp_ui import (
    UIActionHandler,
    ui_action_result_tool_call,
    ui_action_result_prompt,
)

# Create action handler
handler = UIActionHandler()

@handler.on_tool_call
async def handle_tool(tool_name: str, params: dict) -> Any:
    return await execute_tool(tool_name, params)

@handler.on_prompt
async def handle_prompt(prompt: str) -> Any:
    return await send_to_llm(prompt)

# Process incoming action
result = await handler.handle({
    "type": "tool",
    "payload": {"toolName": "search", "params": {"q": "test"}},
})

# Create action results for UI responses
action = ui_action_result_tool_call("search", {"query": "laptops"})
```

Using Components:
-----------------
```python
from agentic_workflows.mcp_ui import (
    create_component_resource,
    ButtonConfig,
    TableConfig,
    TableColumn,
    ListConfig,
    ListItem,
)

# Create a button
button = create_component_resource(
    "button",
    ButtonConfig(
        label="Click Me",
        variant=ButtonVariant.PRIMARY,
        action={"type": "tool", "toolName": "click_handler", "params": {}},
    ),
)

# Create a table
table = create_component_resource(
    "table",
    TableConfig(
        columns=[
            TableColumn(key="name", header="Name"),
            TableColumn(key="price", header="Price"),
        ],
        data=[
            {"name": "Product A", "price": "$29.99"},
            {"name": "Product B", "price": "$49.99"},
        ],
    ),
)

# Create a list
list_ui = create_component_resource(
    "list",
    ListConfig(
        items=[
            ListItem(id="1", title="Item 1", subtitle="Description"),
            ListItem(id="2", title="Item 2", subtitle="Description"),
        ],
        title="My Items",
        selectable=True,
    ),
)
```

OpenAI Apps SDK Compatibility:
------------------------------
```python
from agentic_workflows.mcp_ui import (
    OpenAIAppsWidgetAdapter,
    register_openai_apps_widgets,
)
from agentic_workflows.openai_apps import ListViewPayload, ListItem

# Adapter for converting openai_apps widgets
adapter = OpenAIAppsWidgetAdapter()

# Convert widget to MCP-UI resource
widget_data = ListViewPayload(
    items=[ListItem(id="1", title="Product")],
).to_structured_content()
resource = adapter.convert_widget_to_resource(widget_data, "widget://list-view")

# Adapt existing widget handler
@widget_tool(category=WidgetCategory.LIST)
async def products(category: str) -> dict:
    return {"items": await get_products(category)}

adapted = adapter.adapt_widget_handler(products, "widget://list-view")
server.register_ui_tool("products", adapted)
```
"""

__version__ = "1.0.0"

# UI Types
# UI Actions
from .actions import (
    IntentHandler,
    LinkHandler,
    NotifyHandler,
    PromptHandler,
    # Handler types
    ToolCallHandler,
    # Handler
    UIActionHandler,
    UIActionHandlerConfig,
    create_action_handler,
    ui_action_result_intent,
    ui_action_result_link,
    ui_action_result_notification,
    ui_action_result_prompt,
    # Action result factories
    ui_action_result_tool_call,
)

# Components
from .components import (
    # CSS
    BASE_CSS,
    # Component classes
    Button,
    # Component configs
    ButtonConfig,
    ButtonSize,
    # Enums
    ButtonVariant,
    Card,
    CardConfig,
    Chart,
    ChartConfig,
    ChartDataset,
    Component,
    Input,
    InputConfig,
    InputType,
    List,
    ListConfig,
    ListItem,
    Select,
    SelectConfig,
    SelectOption,
    Table,
    TableColumn,
    TableConfig,
    # Factory functions
    create_component_html,
    create_component_resource,
    generate_remote_dom_script,
)

# Integration
from .integration import (
    # Server
    MCPUIServer,
    # Adapter
    OpenAIAppsWidgetAdapter,
    UIResourceTemplate,
    UIToolHandler,
    # Registration types
    UIToolRegistration,
    create_mcp_ui_server,
    register_openai_apps_widgets,
    # Tool decorator
    ui_tool,
)

# Resource Creation
from .resource import (
    UI_METADATA_PREFIX,
    AdapterConfig,
    # Configuration
    AppsSdkConfig,
    ExternalUrlResource,
    # Resource factories
    RawHtmlResource,
    RemoteDomResource,
    ResourceContent,
    # Convenience functions
    create_html_resource,
    create_remote_dom_resource,
    # Main factory
    create_ui_resource,
    create_url_resource,
    get_apps_sdk_adapter_script,
    # Adapter utilities
    wrap_html_with_adapters,
)
from .ui_types import (
    ContentType,
    FrameSize,
    # Enums
    MimeType,
    UIActionIntentPayload,
    UIActionLinkPayload,
    UIActionNotifyPayload,
    # Action payloads
    UIActionPayload,
    UIActionPromptPayload,
    # Action result
    UIActionResult,
    UIActionToolPayload,
    UIActionType,
    UIMetadata,
    # Core types
    UIResource,
    UIResourceContent,
    # Type aliases
    UIStructuredContent,
)

__all__ = [
    # Version
    "__version__",
    # UI Types - Core
    "UIResource",
    "UIResourceContent",
    "UIMetadata",
    "FrameSize",
    # UI Types - Enums
    "MimeType",
    "ContentType",
    "UIActionType",
    # UI Types - Action Payloads
    "UIActionPayload",
    "UIActionToolPayload",
    "UIActionIntentPayload",
    "UIActionPromptPayload",
    "UIActionNotifyPayload",
    "UIActionLinkPayload",
    "UIActionResult",
    # UI Types - Aliases
    "UIStructuredContent",
    # Resource Creation - Main
    "create_ui_resource",
    # Resource Creation - Factories
    "RawHtmlResource",
    "ExternalUrlResource",
    "RemoteDomResource",
    # Resource Creation - Convenience
    "create_html_resource",
    "create_url_resource",
    "create_remote_dom_resource",
    # Resource Creation - Adapters
    "wrap_html_with_adapters",
    "get_apps_sdk_adapter_script",
    # Resource Creation - Config
    "AppsSdkConfig",
    "AdapterConfig",
    "ResourceContent",
    "UI_METADATA_PREFIX",
    # Actions - Factories
    "ui_action_result_tool_call",
    "ui_action_result_prompt",
    "ui_action_result_link",
    "ui_action_result_intent",
    "ui_action_result_notification",
    # Actions - Handler
    "UIActionHandler",
    "UIActionHandlerConfig",
    "create_action_handler",
    # Actions - Types
    "ToolCallHandler",
    "IntentHandler",
    "PromptHandler",
    "NotifyHandler",
    "LinkHandler",
    # Components - Enums
    "ButtonVariant",
    "ButtonSize",
    "InputType",
    # Components - Configs
    "ButtonConfig",
    "InputConfig",
    "SelectConfig",
    "SelectOption",
    "TableConfig",
    "TableColumn",
    "ChartConfig",
    "ChartDataset",
    "ListConfig",
    "ListItem",
    "CardConfig",
    # Components - Classes
    "Button",
    "Input",
    "Select",
    "Table",
    "Chart",
    "List",
    "Card",
    "Component",
    # Components - Functions
    "create_component_html",
    "create_component_resource",
    "generate_remote_dom_script",
    "BASE_CSS",
    # Integration - Server
    "MCPUIServer",
    "create_mcp_ui_server",
    # Integration - Types
    "UIToolRegistration",
    "UIResourceTemplate",
    "UIToolHandler",
    # Integration - Decorator
    "ui_tool",
    # Integration - Adapter
    "OpenAIAppsWidgetAdapter",
    "register_openai_apps_widgets",
]
