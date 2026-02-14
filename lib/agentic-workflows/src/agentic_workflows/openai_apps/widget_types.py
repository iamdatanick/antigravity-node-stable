"""Widget type definitions for OpenAI Apps SDK integration.

This module defines the core data structures for widgets:
- WidgetTemplate: Template configuration with URI and HTML
- WidgetPayload: Base class for widget data payloads
- WidgetMeta: Metadata for widget rendering and session management
- WidgetState: Stateful widget state tracking

Based on OpenAI Apps SDK patterns for rich UI components.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar


class WidgetDisplayMode(str, Enum):
    """Widget display mode options."""

    COMPACT = "compact"
    EXPANDED = "expanded"
    FULLSCREEN = "fullscreen"


class WidgetTheme(str, Enum):
    """Widget theme options."""

    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


@dataclass
class WidgetTemplate:
    """Widget template configuration.

    Defines how a widget is rendered by specifying its template URI,
    HTML content, and MIME type.

    Attributes:
        uri: Resource URI for the widget template (e.g., "widget://list-view")
        html: The HTML/JS/CSS content for the widget
        mime_type: Content MIME type (default: "text/html")
        name: Human-readable widget name
        description: Widget description for documentation
        version: Template version string

    Example:
        >>> template = WidgetTemplate(
        ...     uri="widget://list-view",
        ...     html="<div id='list-root'></div><script>...</script>",
        ...     name="ListView",
        ...     description="Display lists of items with selection"
        ... )
    """

    uri: str
    html: str
    mime_type: str = "text/html"
    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    def __post_init__(self):
        """Validate template configuration."""
        if not self.uri:
            raise ValueError("Widget template URI is required")
        if not self.html:
            raise ValueError("Widget template HTML is required")

    def to_resource(self) -> dict[str, Any]:
        """Convert to MCP resource format.

        Returns:
            Dictionary suitable for MCP resource response.
        """
        return {
            "uri": self.uri,
            "name": self.name or self.uri,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class WidgetMeta:
    """Widget metadata for rendering and session management.

    Contains the metadata annotations (_meta) that control how widgets
    are rendered and maintain state across conversation turns.

    Attributes:
        output_template: URI of the widget template to use for rendering
        tool_invocation: Information about the tool call that produced this widget
        widget_accessible: Accessibility description for screen readers
        widget_session_id: Unique session ID for stateful widgets
        display_mode: Current display mode (compact, expanded, fullscreen)
        theme: Widget theme (light, dark, system)
        additional: Additional custom metadata fields

    Example:
        >>> meta = WidgetMeta(
        ...     output_template="widget://list-view",
        ...     widget_session_id="sess_abc123",
        ...     widget_accessible="Product list with 5 items"
        ... )
        >>> meta.to_dict()
        {'openai/outputTemplate': 'widget://list-view', 'widgetSessionId': 'sess_abc123', ...}
    """

    output_template: str
    tool_invocation: dict[str, Any] | None = None
    widget_accessible: str = ""
    widget_session_id: str = ""
    display_mode: WidgetDisplayMode = WidgetDisplayMode.COMPACT
    theme: WidgetTheme = WidgetTheme.SYSTEM
    additional: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate session ID if not provided."""
        if not self.widget_session_id:
            self.widget_session_id = f"sess_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to _meta dictionary format.

        Returns:
            Dictionary with OpenAI Apps SDK metadata format.
        """
        meta = {
            "openai/outputTemplate": self.output_template,
            "widgetSessionId": self.widget_session_id,
        }

        if self.tool_invocation:
            meta["toolInvocation"] = self.tool_invocation

        if self.widget_accessible:
            meta["widgetAccessible"] = self.widget_accessible

        if self.display_mode != WidgetDisplayMode.COMPACT:
            meta["displayMode"] = self.display_mode.value

        if self.theme != WidgetTheme.SYSTEM:
            meta["theme"] = self.theme.value

        # Merge additional metadata
        meta.update(self.additional)

        return meta


T = TypeVar("T")


class WidgetPayload(ABC, Generic[T]):
    """Base class for widget data payloads.

    Widget payloads contain the structured content that gets passed
    to widget templates for rendering. Subclasses define specific
    payload shapes for different widget types.

    Type Parameters:
        T: The type of data contained in this payload.

    Example:
        >>> class ProductListPayload(WidgetPayload[list[dict]]):
        ...     def __init__(self, products: list[dict]):
        ...         self.products = products
        ...
        ...     def to_structured_content(self) -> dict[str, Any]:
        ...         return {"products": self.products, "count": len(self.products)}
    """

    @abstractmethod
    def to_structured_content(self) -> dict[str, Any]:
        """Convert payload to structured content format.

        Returns:
            Dictionary with widget-specific data structure.
        """
        pass

    def with_meta(self, meta: WidgetMeta) -> dict[str, Any]:
        """Combine payload with metadata for tool result.

        Args:
            meta: Widget metadata for rendering.

        Returns:
            Complete structured content with _meta field.
        """
        content = self.to_structured_content()
        content["_meta"] = meta.to_dict()
        return content


@dataclass
class WidgetState:
    """Stateful widget state container.

    Tracks widget state across conversation turns, enabling
    interactive widgets that maintain state (e.g., shopping carts,
    forms with partial input).

    Attributes:
        session_id: Unique session identifier
        state_data: Current state data dictionary
        version: State version for optimistic locking
        created_at: State creation timestamp
        updated_at: Last update timestamp
        expires_at: Optional expiration timestamp
        metadata: Additional state metadata

    Example:
        >>> state = WidgetState(session_id="sess_abc123")
        >>> state.update({"items": [{"id": 1, "qty": 2}]})
        >>> state.merge({"items": [{"id": 1, "qty": 3}]})  # Updates qty
    """

    session_id: str
    state_data: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, new_data: dict[str, Any]) -> WidgetState:
        """Replace state data completely.

        Args:
            new_data: New state data to set.

        Returns:
            Self for chaining.
        """
        self.state_data = new_data
        self.version += 1
        self.updated_at = datetime.utcnow()
        return self

    def merge(self, partial_data: dict[str, Any]) -> WidgetState:
        """Merge partial data into current state.

        Performs a shallow merge of partial_data into state_data.
        For deep merging, use merge_deep().

        Args:
            partial_data: Partial state data to merge.

        Returns:
            Self for chaining.
        """
        self.state_data.update(partial_data)
        self.version += 1
        self.updated_at = datetime.utcnow()
        return self

    def merge_deep(self, partial_data: dict[str, Any]) -> WidgetState:
        """Deep merge partial data into current state.

        Recursively merges nested dictionaries.

        Args:
            partial_data: Partial state data to merge.

        Returns:
            Self for chaining.
        """
        self.state_data = self._deep_merge(self.state_data, partial_data)
        self.version += 1
        self.updated_at = datetime.utcnow()
        return self

    def _deep_merge(self, base: dict, update: dict) -> dict:
        """Recursively merge update into base."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state data.

        Args:
            key: State key to retrieve.
            default: Default value if key not found.

        Returns:
            Value from state or default.
        """
        return self.state_data.get(key, default)

    def set(self, key: str, value: Any) -> WidgetState:
        """Set a single value in state data.

        Args:
            key: State key to set.
            value: Value to set.

        Returns:
            Self for chaining.
        """
        self.state_data[key] = value
        self.version += 1
        self.updated_at = datetime.utcnow()
        return self

    def is_expired(self) -> bool:
        """Check if state has expired.

        Returns:
            True if state has an expiration and it has passed.
        """
        if self.expires_at is None:
            return False
        return datetime.utcnow() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of state.
        """
        return {
            "session_id": self.session_id,
            "state_data": self.state_data,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WidgetState:
        """Create state from dictionary.

        Args:
            data: Dictionary representation of state.

        Returns:
            WidgetState instance.
        """
        return cls(
            session_id=data["session_id"],
            state_data=data.get("state_data", {}),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.utcnow(),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            metadata=data.get("metadata", {}),
        )


# Type aliases for convenience
StructuredContent = dict[str, Any]
WidgetMetaDict = dict[str, Any]
