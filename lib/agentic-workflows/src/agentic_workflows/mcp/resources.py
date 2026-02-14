"""MCP Resources Module.

Provides resource management for MCP protocol following spec 2024-11-05.

Usage:
    from agentic_workflows.mcp.resources import ResourceManager, Resource

    manager = ResourceManager()
    manager.register(Resource(
        uri="file:///docs/readme.md",
        name="README",
        mimeType="text/markdown",
    ))
    resources = await manager.list()
    content = await manager.read("file:///docs/readme.md")
"""

from __future__ import annotations

import asyncio
import builtins
import logging
import mimetypes
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ResourceContent:
    """Content of a resource."""

    uri: str
    mimeType: str | None = None
    text: str | None = None
    blob: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        result = {"uri": self.uri}
        if self.mimeType:
            result["mimeType"] = self.mimeType
        if self.text is not None:
            result["text"] = self.text
        if self.blob is not None:
            import base64

            result["blob"] = base64.b64encode(self.blob).decode()
        return result


@dataclass
class Resource:
    """MCP Resource definition.

    Represents a resource that can be accessed via MCP protocol.

    Attributes:
        uri: Unique resource identifier (URI format).
        name: Human-readable name.
        description: Optional description.
        mimeType: MIME type of the resource.
        annotations: Optional annotations for client hints.
    """

    uri: str
    name: str
    description: str | None = None
    mimeType: str | None = None
    annotations: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        result = {
            "uri": self.uri,
            "name": self.name,
        }
        if self.description:
            result["description"] = self.description
        if self.mimeType:
            result["mimeType"] = self.mimeType
        if self.annotations:
            result["annotations"] = self.annotations
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Resource:
        """Create from MCP format."""
        return cls(
            uri=data["uri"],
            name=data["name"],
            description=data.get("description"),
            mimeType=data.get("mimeType"),
            annotations=data.get("annotations"),
        )

    @classmethod
    def from_file(cls, path: str | Path, base_uri: str = "file://") -> Resource:
        """Create resource from a file path.

        Args:
            path: File path.
            base_uri: Base URI prefix.

        Returns:
            Resource instance.
        """
        file_path = Path(path)
        mime_type, _ = mimetypes.guess_type(str(file_path))

        return cls(
            uri=f"{base_uri}{file_path.absolute()}",
            name=file_path.name,
            mimeType=mime_type,
        )


@dataclass
class ResourceTemplate:
    """MCP Resource Template for dynamic resource patterns.

    Example:
        template = ResourceTemplate(
            uriTemplate="file:///{path}",
            name="File Resource",
            description="Access files by path",
        )
    """

    uriTemplate: str
    name: str
    description: str | None = None
    mimeType: str | None = None
    annotations: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        result = {
            "uriTemplate": self.uriTemplate,
            "name": self.name,
        }
        if self.description:
            result["description"] = self.description
        if self.mimeType:
            result["mimeType"] = self.mimeType
        if self.annotations:
            result["annotations"] = self.annotations
        return result

    def matches(self, uri: str) -> dict[str, str] | None:
        """Check if URI matches template and extract variables.

        Args:
            uri: URI to match.

        Returns:
            Dict of extracted variables or None if no match.
        """
        # Convert template to regex
        pattern = self.uriTemplate
        variables = []

        # Find all {variable} placeholders
        for match in re.finditer(r"\{(\w+)\}", pattern):
            var_name = match.group(1)
            variables.append(var_name)
            pattern = pattern.replace(match.group(0), r"(.+)")

        regex = re.compile(f"^{pattern}$")
        match = regex.match(uri)

        if match:
            return dict(zip(variables, match.groups()))
        return None


# Type for resource handlers
ResourceHandler = Callable[[str], ResourceContent | None]


class ResourceManager:
    """Manages MCP resources and templates.

    Example:
        manager = ResourceManager()

        # Register static resources
        manager.register(Resource(
            uri="file:///config.json",
            name="Configuration",
            mimeType="application/json",
        ))

        # Register templates for dynamic resources
        manager.register_template(ResourceTemplate(
            uriTemplate="memory:///{key}",
            name="Memory Store",
        ))

        # Register handler for reading
        manager.set_handler("file://", file_handler)

        # List and read
        resources = await manager.list()
        content = await manager.read("file:///config.json")
    """

    def __init__(self):
        """Initialize resource manager."""
        self._resources: dict[str, Resource] = {}
        self._templates: list[ResourceTemplate] = []
        self._handlers: dict[str, ResourceHandler] = {}
        self._subscriptions: dict[str, list[Callable]] = {}

    def register(self, resource: Resource) -> None:
        """Register a static resource.

        Args:
            resource: Resource to register.
        """
        self._resources[resource.uri] = resource
        logger.debug(f"Registered resource: {resource.uri}")

    def unregister(self, uri: str) -> bool:
        """Unregister a resource.

        Args:
            uri: Resource URI.

        Returns:
            True if removed, False if not found.
        """
        if uri in self._resources:
            del self._resources[uri]
            return True
        return False

    def register_template(self, template: ResourceTemplate) -> None:
        """Register a resource template.

        Args:
            template: Template to register.
        """
        self._templates.append(template)
        logger.debug(f"Registered template: {template.uriTemplate}")

    def set_handler(self, uri_prefix: str, handler: ResourceHandler) -> None:
        """Set handler for reading resources with given prefix.

        Args:
            uri_prefix: URI prefix to match.
            handler: Handler function.
        """
        self._handlers[uri_prefix] = handler

    async def list(self, cursor: str | None = None) -> dict[str, Any]:
        """List available resources.

        Args:
            cursor: Pagination cursor.

        Returns:
            MCP resources/list response.
        """
        resources = [r.to_dict() for r in self._resources.values()]

        return {
            "resources": resources,
        }

    async def list_templates(self) -> dict[str, Any]:
        """List resource templates.

        Returns:
            MCP resources/templates/list response.
        """
        templates = [t.to_dict() for t in self._templates]

        return {
            "resourceTemplates": templates,
        }

    async def read(self, uri: str) -> dict[str, Any]:
        """Read resource content.

        Args:
            uri: Resource URI.

        Returns:
            MCP resources/read response.

        Raises:
            ValueError: If resource not found or no handler.
        """
        # Find matching handler
        handler = None
        for prefix, h in self._handlers.items():
            if uri.startswith(prefix):
                handler = h
                break

        if not handler:
            # Try built-in file handler
            if uri.startswith("file://"):
                content = await self._read_file(uri)
            else:
                raise ValueError(f"No handler for resource: {uri}")
        else:
            content = handler(uri)

        if content is None:
            raise ValueError(f"Resource not found: {uri}")

        return {
            "contents": [content.to_dict()],
        }

    async def _read_file(self, uri: str) -> ResourceContent:
        """Built-in file reader.

        Args:
            uri: File URI.

        Returns:
            ResourceContent with file contents.
        """
        # Parse file URI
        parsed = urlparse(uri)
        file_path = Path(parsed.path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(str(file_path))

        # Read as text or binary
        if mime_type and mime_type.startswith("text/"):
            content = file_path.read_text(encoding="utf-8")
            return ResourceContent(uri=uri, mimeType=mime_type, text=content)
        else:
            content = file_path.read_bytes()
            return ResourceContent(uri=uri, mimeType=mime_type, blob=content)

    async def subscribe(self, uri: str, callback: Callable) -> bool:
        """Subscribe to resource updates.

        Args:
            uri: Resource URI.
            callback: Callback for updates.

        Returns:
            True if subscribed.
        """
        if uri not in self._subscriptions:
            self._subscriptions[uri] = []
        self._subscriptions[uri].append(callback)
        return True

    async def unsubscribe(self, uri: str, callback: Callable) -> bool:
        """Unsubscribe from resource updates.

        Args:
            uri: Resource URI.
            callback: Callback to remove.

        Returns:
            True if unsubscribed.
        """
        if uri in self._subscriptions:
            try:
                self._subscriptions[uri].remove(callback)
                return True
            except ValueError:
                pass
        return False

    async def notify_update(self, uri: str) -> None:
        """Notify subscribers of resource update.

        Args:
            uri: Updated resource URI.
        """
        if uri in self._subscriptions:
            for callback in self._subscriptions[uri]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(uri)
                    else:
                        callback(uri)
                except Exception as e:
                    logger.error(f"Subscription callback error: {e}")

    def get(self, uri: str) -> Resource | None:
        """Get a resource by URI.

        Args:
            uri: Resource URI.

        Returns:
            Resource or None.
        """
        return self._resources.get(uri)

    def list_uris(self) -> builtins.list[str]:
        """Get list of registered resource URIs.

        Returns:
            List of URIs.
        """
        return list(self._resources.keys())


__all__ = [
    "Resource",
    "ResourceContent",
    "ResourceTemplate",
    "ResourceHandler",
    "ResourceManager",
]
