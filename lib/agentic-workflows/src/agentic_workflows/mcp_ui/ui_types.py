"""MCP-UI Type Definitions.

This module provides type definitions for the MCP-UI SDK based on the
specification at https://github.com/MCP-UI-Org/mcp-ui.

Types include:
- UIResource: Core payload structure for UI resources
- MimeType: Supported MIME types for rendering
- ContentType: Content type variants (rawHtml, externalUrl, remoteDom)
- UIActionType: Action types for UI callbacks
- UIActionResult: Result types for each action type
- UIMetadata: Metadata for UI rendering configuration

Based on MCP-UI SDK specification for rich UI components in MCP tools.
"""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Union, Literal, TypeVar, Generic


class MimeType(str, Enum):
    """MIME types supported by MCP-UI.

    These MIME types determine how the UI resource is rendered:
    - TEXT_HTML: Rendered as inline iframe content via srcDoc
    - TEXT_URI_LIST: Rendered as external iframe via src attribute
    - REMOTE_DOM: Rendered via sandboxed script execution with Remote DOM
    """

    TEXT_HTML = "text/html"
    TEXT_URI_LIST = "text/uri-list"
    REMOTE_DOM_REACT = "application/vnd.mcp-ui.remote-dom+javascript; framework=react"
    REMOTE_DOM_WEBCOMPONENTS = "application/vnd.mcp-ui.remote-dom+javascript; framework=webcomponents"

    @classmethod
    def remote_dom(cls, framework: Literal["react", "webcomponents"] = "react") -> str:
        """Get Remote DOM MIME type for specified framework.

        Args:
            framework: Either 'react' or 'webcomponents'.

        Returns:
            Remote DOM MIME type string.
        """
        return f"application/vnd.mcp-ui.remote-dom+javascript; framework={framework}"


class ContentType(str, Enum):
    """Content type variants for MCP-UI resources.

    Determines how content is provided and rendered:
    - RAW_HTML: Self-contained HTML snippets rendered inline
    - EXTERNAL_URL: External application embedded via iframe src
    - REMOTE_DOM: Dynamic, host-styled components via script execution
    """

    RAW_HTML = "rawHtml"
    EXTERNAL_URL = "externalUrl"
    REMOTE_DOM = "remoteDom"


class UIActionType(str, Enum):
    """UI action types for onUIAction callback.

    These represent the different types of actions that can be
    triggered from within the UI and handled by the host application.
    """

    TOOL = "tool"
    INTENT = "intent"
    PROMPT = "prompt"
    NOTIFY = "notify"
    LINK = "link"


@dataclass
class UIActionToolPayload:
    """Payload for tool action type.

    Attributes:
        tool_name: Name of the tool to invoke.
        params: Parameters to pass to the tool.
    """
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "toolName": self.tool_name,
            "params": self.params,
        }


@dataclass
class UIActionIntentPayload:
    """Payload for intent action type.

    Attributes:
        intent: Intent identifier.
        params: Parameters for the intent.
    """
    intent: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "intent": self.intent,
            "params": self.params,
        }


@dataclass
class UIActionPromptPayload:
    """Payload for prompt action type.

    Attributes:
        prompt: Prompt text to send to the LLM.
    """
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {"prompt": self.prompt}


@dataclass
class UIActionNotifyPayload:
    """Payload for notify action type.

    Attributes:
        message: Notification message to display.
    """
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {"message": self.message}


@dataclass
class UIActionLinkPayload:
    """Payload for link action type.

    Attributes:
        url: URL to navigate to.
    """
    url: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {"url": self.url}


# Union type for all action payloads
UIActionPayload = Union[
    UIActionToolPayload,
    UIActionIntentPayload,
    UIActionPromptPayload,
    UIActionNotifyPayload,
    UIActionLinkPayload,
]


@dataclass
class UIActionResult:
    """Result from a UI action.

    Represents an action triggered from within the UI via the
    onUIAction callback mechanism.

    Attributes:
        type: The action type (tool, intent, prompt, notify, link).
        payload: Action-specific payload data.
        message_id: Optional message ID for async response correlation.
    """
    type: UIActionType
    payload: UIActionPayload
    message_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        result: dict[str, Any] = {
            "type": self.type.value,
            "payload": self.payload.to_dict(),
        }
        if self.message_id:
            result["messageId"] = self.message_id
        return result

    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        params: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> "UIActionResult":
        """Create a tool call action result.

        Args:
            tool_name: Name of the tool to call.
            params: Parameters for the tool.
            message_id: Optional message ID for response correlation.

        Returns:
            UIActionResult for tool action.
        """
        return cls(
            type=UIActionType.TOOL,
            payload=UIActionToolPayload(tool_name, params or {}),
            message_id=message_id,
        )

    @classmethod
    def intent(
        cls,
        intent: str,
        params: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> "UIActionResult":
        """Create an intent action result.

        Args:
            intent: Intent identifier.
            params: Parameters for the intent.
            message_id: Optional message ID for response correlation.

        Returns:
            UIActionResult for intent action.
        """
        return cls(
            type=UIActionType.INTENT,
            payload=UIActionIntentPayload(intent, params or {}),
            message_id=message_id,
        )

    @classmethod
    def prompt(
        cls,
        prompt: str,
        message_id: str | None = None,
    ) -> "UIActionResult":
        """Create a prompt action result.

        Args:
            prompt: Prompt text for the LLM.
            message_id: Optional message ID for response correlation.

        Returns:
            UIActionResult for prompt action.
        """
        return cls(
            type=UIActionType.PROMPT,
            payload=UIActionPromptPayload(prompt),
            message_id=message_id,
        )

    @classmethod
    def notification(
        cls,
        message: str,
        message_id: str | None = None,
    ) -> "UIActionResult":
        """Create a notification action result.

        Args:
            message: Notification message.
            message_id: Optional message ID for response correlation.

        Returns:
            UIActionResult for notify action.
        """
        return cls(
            type=UIActionType.NOTIFY,
            payload=UIActionNotifyPayload(message),
            message_id=message_id,
        )

    @classmethod
    def link(
        cls,
        url: str,
        message_id: str | None = None,
    ) -> "UIActionResult":
        """Create a link action result.

        Args:
            url: URL to navigate to.
            message_id: Optional message ID for response correlation.

        Returns:
            UIActionResult for link action.
        """
        return cls(
            type=UIActionType.LINK,
            payload=UIActionLinkPayload(url),
            message_id=message_id,
        )


@dataclass
class FrameSize:
    """Preferred frame size for UI rendering.

    Attributes:
        width: Width in pixels or CSS units.
        height: Height in pixels or CSS units.
    """
    width: int | str | None = None
    height: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        if self.width is not None:
            result["width"] = self.width
        if self.height is not None:
            result["height"] = self.height
        return result


@dataclass
class UIMetadata:
    """Metadata for MCP-UI resource rendering.

    Contains rendering configuration and initial state that controls
    how the UI resource is displayed and initialized.

    Attributes:
        preferred_frame_size: Preferred iframe dimensions.
        initial_render_data: Initial data passed to the widget.
        auto_resize_iframe: Enable automatic iframe resizing.
        sandbox_permissions: Additional sandbox permissions.
        accessible_description: Accessibility description.
        session_id: Session ID for stateful widgets.
        additional: Additional custom metadata fields.
    """
    preferred_frame_size: FrameSize | None = None
    initial_render_data: dict[str, Any] = field(default_factory=dict)
    auto_resize_iframe: bool | dict[str, bool] = True
    sandbox_permissions: list[str] = field(default_factory=list)
    accessible_description: str = ""
    session_id: str = ""
    additional: dict[str, Any] = field(default_factory=dict)

    # Prefix for metadata keys in the resource
    UI_METADATA_PREFIX = "mcp-ui/"

    def __post_init__(self):
        """Generate session ID if not provided."""
        if not self.session_id:
            self.session_id = f"ui_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format with UI_METADATA_PREFIX.

        Returns:
            Dictionary with prefixed metadata keys.
        """
        prefix = self.UI_METADATA_PREFIX
        result: dict[str, Any] = {}

        if self.preferred_frame_size:
            result[f"{prefix}preferredFrameSize"] = self.preferred_frame_size.to_dict()

        if self.initial_render_data:
            result[f"{prefix}initialRenderData"] = self.initial_render_data

        if self.auto_resize_iframe != True:  # Only include if not default
            if isinstance(self.auto_resize_iframe, dict):
                result[f"{prefix}autoResizeIframe"] = self.auto_resize_iframe
            else:
                result[f"{prefix}autoResizeIframe"] = self.auto_resize_iframe

        if self.sandbox_permissions:
            result[f"{prefix}sandboxPermissions"] = self.sandbox_permissions

        if self.accessible_description:
            result[f"{prefix}accessibleDescription"] = self.accessible_description

        if self.session_id:
            result[f"{prefix}sessionId"] = self.session_id

        # Add any additional custom metadata
        for key, value in self.additional.items():
            if not key.startswith(prefix):
                key = f"{prefix}{key}"
            result[key] = value

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UIMetadata":
        """Create UIMetadata from dictionary.

        Args:
            data: Dictionary with prefixed or unprefixed keys.

        Returns:
            UIMetadata instance.
        """
        prefix = cls.UI_METADATA_PREFIX

        # Extract values, handling both prefixed and unprefixed keys
        def get_value(key: str) -> Any:
            return data.get(f"{prefix}{key}") or data.get(key)

        frame_size = None
        frame_size_data = get_value("preferredFrameSize")
        if frame_size_data:
            frame_size = FrameSize(
                width=frame_size_data.get("width"),
                height=frame_size_data.get("height"),
            )

        return cls(
            preferred_frame_size=frame_size,
            initial_render_data=get_value("initialRenderData") or {},
            auto_resize_iframe=get_value("autoResizeIframe") or True,
            sandbox_permissions=get_value("sandboxPermissions") or [],
            accessible_description=get_value("accessibleDescription") or "",
            session_id=get_value("sessionId") or "",
        )


@dataclass
class UIResourceContent:
    """Content container for UI resource.

    Holds either text or blob content for the resource.

    Attributes:
        text: Text content (for text encoding).
        blob: Base64-encoded blob content (for blob encoding).
    """
    text: str | None = None
    blob: str | None = None

    @property
    def is_blob(self) -> bool:
        """Check if content is blob-encoded."""
        return self.blob is not None

    @property
    def content(self) -> str:
        """Get the content (text or decoded blob)."""
        if self.text is not None:
            return self.text
        if self.blob is not None:
            return base64.b64decode(self.blob).decode("utf-8")
        return ""

    @classmethod
    def from_text(cls, text: str) -> "UIResourceContent":
        """Create content from text.

        Args:
            text: Text content.

        Returns:
            UIResourceContent with text.
        """
        return cls(text=text)

    @classmethod
    def from_blob(cls, data: bytes) -> "UIResourceContent":
        """Create content from binary data.

        Args:
            data: Binary data to encode.

        Returns:
            UIResourceContent with base64-encoded blob.
        """
        return cls(blob=base64.b64encode(data).decode("ascii"))

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        if self.text is not None:
            return {"text": self.text}
        if self.blob is not None:
            return {"blob": self.blob}
        return {}


@dataclass
class UIResource:
    """MCP-UI Resource payload.

    The core payload structure returned from MCP-UI server to client.
    Contains the resource URI, MIME type, and content.

    Attributes:
        uri: Resource URI (e.g., "ui://component/id").
        mime_type: Content MIME type determining rendering approach.
        content: Text or blob content.
        metadata: Optional rendering metadata.

    Example:
        >>> resource = UIResource(
        ...     uri="ui://greeting/1",
        ...     mime_type=MimeType.TEXT_HTML,
        ...     content=UIResourceContent.from_text("<p>Hello!</p>"),
        ... )
        >>> payload = resource.to_mcp_resource()
    """
    uri: str
    mime_type: MimeType | str
    content: UIResourceContent
    metadata: UIMetadata | None = None

    def __post_init__(self):
        """Validate resource configuration."""
        if not self.uri:
            raise ValueError("UIResource uri is required")
        if not self.content.text and not self.content.blob:
            raise ValueError("UIResource must have text or blob content")

    @property
    def mime_type_str(self) -> str:
        """Get MIME type as string."""
        if isinstance(self.mime_type, MimeType):
            return self.mime_type.value
        return self.mime_type

    def to_mcp_resource(self) -> dict[str, Any]:
        """Convert to MCP resource format.

        Returns:
            Dictionary in MCP resource format.
        """
        resource: dict[str, Any] = {
            "uri": self.uri,
            "mimeType": self.mime_type_str,
        }
        resource.update(self.content.to_dict())

        if self.metadata:
            resource.update(self.metadata.to_dict())

        return resource

    def to_tool_result(self) -> dict[str, Any]:
        """Convert to MCP tool result format.

        Returns:
            Dictionary in MCP tool result format with type='resource'.
        """
        return {
            "type": "resource",
            "resource": self.to_mcp_resource(),
        }


# Type for supported content types in resource creation
ContentSpec = Union[
    tuple[Literal["rawHtml"], str],  # (type, htmlString)
    tuple[Literal["externalUrl"], str],  # (type, iframeUrl)
    tuple[Literal["remoteDom"], str, Literal["react", "webcomponents"]],  # (type, script, framework)
]


# Type alias for structured UI content
UIStructuredContent = dict[str, Any]
