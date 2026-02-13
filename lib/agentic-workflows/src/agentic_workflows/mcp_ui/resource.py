"""MCP-UI Resource Creation.

This module provides factory functions for creating MCP-UI resources
matching the mcp-ui-server Python package API.

Functions include:
- create_ui_resource(): Main factory function for all resource types
- RawHtmlResource: Factory for inline HTML resources
- ExternalUrlResource: Factory for external iframe resources
- RemoteDomResource: Factory for Remote DOM script resources

Based on the mcp-ui-server package specification.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from .ui_types import (
    ContentType,
    FrameSize,
    MimeType,
    UIMetadata,
    UIResource,
    UIResourceContent,
)

# Metadata key prefix
UI_METADATA_PREFIX = "mcp-ui/"


@dataclass
class AppsSdkConfig:
    """Configuration for Apps SDK adapter.

    Attributes:
        enabled: Whether the adapter is enabled.
        intent_handling: How to handle intents ('ignore', 'prompt', 'error').
    """

    enabled: bool = False
    intent_handling: Literal["ignore", "prompt", "error"] = "prompt"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "enabled": self.enabled,
            "intentHandling": self.intent_handling,
        }


@dataclass
class AdapterConfig:
    """Configuration for resource adapters.

    Attributes:
        apps_sdk: Configuration for Apps SDK adapter (ChatGPT compatibility).
    """

    apps_sdk: AppsSdkConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        if self.apps_sdk:
            result["appsSdk"] = self.apps_sdk.to_dict()
        return result


@dataclass
class ResourceContent:
    """Content specification for resource creation.

    One of the type-specific fields should be set.

    Attributes:
        type: Content type (rawHtml, externalUrl, remoteDom).
        html_string: HTML content for rawHtml type.
        iframe_url: URL for externalUrl type.
        script: JavaScript for remoteDom type.
        framework: Framework for remoteDom type (react or webcomponents).
    """

    type: ContentType
    html_string: str = ""
    iframe_url: str = ""
    script: str = ""
    framework: Literal["react", "webcomponents"] = "react"

    @classmethod
    def raw_html(cls, html_string: str) -> ResourceContent:
        """Create raw HTML content.

        Args:
            html_string: HTML content string.

        Returns:
            ResourceContent for raw HTML.
        """
        return cls(type=ContentType.RAW_HTML, html_string=html_string)

    @classmethod
    def external_url(cls, iframe_url: str) -> ResourceContent:
        """Create external URL content.

        Args:
            iframe_url: URL to embed in iframe.

        Returns:
            ResourceContent for external URL.
        """
        return cls(type=ContentType.EXTERNAL_URL, iframe_url=iframe_url)

    @classmethod
    def remote_dom(
        cls,
        script: str,
        framework: Literal["react", "webcomponents"] = "react",
    ) -> ResourceContent:
        """Create Remote DOM content.

        Args:
            script: JavaScript content for Remote DOM.
            framework: Target framework (react or webcomponents).

        Returns:
            ResourceContent for Remote DOM.
        """
        return cls(
            type=ContentType.REMOTE_DOM,
            script=script,
            framework=framework,
        )


def create_ui_resource(
    config: dict[str, Any],
) -> UIResource:
    """Create a UI resource from configuration.

    Main factory function matching the mcp-ui-server Python package API.

    Args:
        config: Configuration dictionary with:
            - uri: Resource URI (e.g., "ui://greeting/1")
            - content: Content configuration with type and content
            - encoding: Content encoding ("text" or "blob")
            - metadata: Optional metadata configuration
            - adapters: Optional adapter configurations

    Returns:
        UIResource instance.

    Example:
        >>> resource = create_ui_resource({
        ...     "uri": "ui://greeting/1",
        ...     "content": {"type": "rawHtml", "htmlString": "<p>Hello!</p>"},
        ...     "encoding": "text",
        ... })
    """
    uri = config.get("uri", "")
    if not uri:
        uri = f"ui://{uuid.uuid4().hex[:8]}"

    content_config = config.get("content", {})
    encoding = config.get("encoding", "text")
    metadata_config = config.get("metadata", {})
    adapters_config = config.get("adapters", {})

    # Determine content type and value
    content_type = content_config.get("type", "rawHtml")
    content_value = ""
    mime_type: MimeType | str = MimeType.TEXT_HTML

    if content_type == "rawHtml":
        content_value = content_config.get("htmlString", "")
        mime_type = MimeType.TEXT_HTML

        # Apply adapters if configured
        if adapters_config.get("appsSdk", {}).get("enabled"):
            content_value = wrap_html_with_adapters(
                content_value,
                adapters_config,
            )

    elif content_type == "externalUrl":
        content_value = content_config.get("iframeUrl", "")
        mime_type = MimeType.TEXT_URI_LIST

    elif content_type == "remoteDom":
        content_value = content_config.get("script", "")
        framework = content_config.get("framework", "react")
        mime_type = MimeType.remote_dom(framework)

    # Create content based on encoding
    if encoding == "blob":
        content = UIResourceContent.from_blob(content_value.encode("utf-8"))
    else:
        content = UIResourceContent.from_text(content_value)

    # Create metadata
    metadata = None
    if metadata_config:
        frame_size = None
        if "preferredFrameSize" in metadata_config:
            fs = metadata_config["preferredFrameSize"]
            frame_size = FrameSize(
                width=fs.get("width"),
                height=fs.get("height"),
            )

        metadata = UIMetadata(
            preferred_frame_size=frame_size,
            initial_render_data=metadata_config.get("initialRenderData", {}),
            auto_resize_iframe=metadata_config.get("autoResizeIframe", True),
            accessible_description=metadata_config.get("accessibleDescription", ""),
        )

    return UIResource(
        uri=uri,
        mime_type=mime_type,
        content=content,
        metadata=metadata,
    )


def wrap_html_with_adapters(
    html_content: str,
    adapters: dict[str, Any],
) -> str:
    """Wrap HTML content with adapter scripts.

    Adds adapter scripts to HTML for compatibility with various hosts
    like ChatGPT via the Apps SDK.

    Args:
        html_content: Original HTML content.
        adapters: Adapter configuration dictionary.

    Returns:
        HTML with adapter scripts injected.
    """
    apps_sdk = adapters.get("appsSdk", {})
    if not apps_sdk.get("enabled"):
        return html_content

    adapter_script = get_apps_sdk_adapter_script(apps_sdk.get("config", {}))

    # Inject script before closing </body> or at end
    if "</body>" in html_content:
        return html_content.replace(
            "</body>",
            f"<script>{adapter_script}</script></body>",
        )
    elif "</html>" in html_content:
        return html_content.replace(
            "</html>",
            f"<script>{adapter_script}</script></html>",
        )
    else:
        return html_content + f"\n<script>{adapter_script}</script>"


def get_apps_sdk_adapter_script(config: dict[str, Any] | None = None) -> str:
    """Get the Apps SDK adapter script.

    This script bridges MCP-UI postMessage protocol to the
    OpenAI Apps SDK API for ChatGPT compatibility.

    Args:
        config: Optional adapter configuration.

    Returns:
        JavaScript adapter script.
    """
    config = config or {}
    intent_handling = config.get("intentHandling", "prompt")

    return f"""
// MCP-UI Apps SDK Adapter
(function() {{
  const config = {{ intentHandling: '{intent_handling}' }};

  // Bridge MCP-UI postMessage to Apps SDK
  const mcpui = {{
    callTool: function(toolName, params) {{
      if (window.openai && window.openai.callTool) {{
        return window.openai.callTool(toolName, params);
      }}
      window.parent.postMessage({{
        type: 'mcp-ui:action',
        action: {{ type: 'tool', payload: {{ toolName, params }} }}
      }}, '*');
    }},

    sendPrompt: function(prompt) {{
      if (window.openai && window.openai.sendPrompt) {{
        return window.openai.sendPrompt(prompt);
      }}
      window.parent.postMessage({{
        type: 'mcp-ui:action',
        action: {{ type: 'prompt', payload: {{ prompt }} }}
      }}, '*');
    }},

    sendIntent: function(intent, params) {{
      if (config.intentHandling === 'prompt') {{
        // Convert intent to prompt
        const prompt = `Intent: ${{intent}}\\nParams: ${{JSON.stringify(params)}}`;
        return mcpui.sendPrompt(prompt);
      }} else if (config.intentHandling === 'ignore') {{
        console.log('Intent ignored:', intent, params);
        return;
      }}
      window.parent.postMessage({{
        type: 'mcp-ui:action',
        action: {{ type: 'intent', payload: {{ intent, params }} }}
      }}, '*');
    }},

    notify: function(message) {{
      if (window.openai && window.openai.showToast) {{
        return window.openai.showToast(message);
      }}
      window.parent.postMessage({{
        type: 'mcp-ui:action',
        action: {{ type: 'notify', payload: {{ message }} }}
      }}, '*');
    }},

    openLink: function(url) {{
      if (window.openai && window.openai.openLink) {{
        return window.openai.openLink(url);
      }}
      window.parent.postMessage({{
        type: 'mcp-ui:action',
        action: {{ type: 'link', payload: {{ url }} }}
      }}, '*');
    }},

    getRenderData: function() {{
      if (window.openai && window.openai.toolOutput) {{
        return window.openai.toolOutput;
      }}
      return window.__MCP_UI_RENDER_DATA__ || {{}};
    }},

    getTheme: function() {{
      if (window.openai && window.openai.theme) {{
        return window.openai.theme;
      }}
      return window.__MCP_UI_THEME__ || 'light';
    }}
  }};

  // Expose globally
  window.mcpui = mcpui;

  // Also expose as openai for backward compatibility
  if (!window.openai) {{
    window.openai = {{
      callTool: mcpui.callTool,
      toolOutput: mcpui.getRenderData(),
      theme: mcpui.getTheme(),
    }};
  }}

  // Listen for messages from parent
  window.addEventListener('message', function(event) {{
    if (event.data && event.data.type === 'mcp-ui:renderData') {{
      window.__MCP_UI_RENDER_DATA__ = event.data.data;
      window.openai.toolOutput = event.data.data;
      window.dispatchEvent(new CustomEvent('mcp-ui:dataUpdate', {{
        detail: event.data.data
      }}));
    }}
    if (event.data && event.data.type === 'mcp-ui:theme') {{
      window.__MCP_UI_THEME__ = event.data.theme;
      window.openai.theme = event.data.theme;
      document.body.dataset.theme = event.data.theme;
    }}
  }});
}})();
"""


class RawHtmlResource:
    """Factory for creating raw HTML UI resources.

    Provides a convenient API for creating inline HTML resources
    that are rendered via srcDoc in an iframe.

    Example:
        >>> resource = RawHtmlResource.create(
        ...     uri="ui://hello/1",
        ...     html="<p>Hello, World!</p>",
        ...     title="Greeting",
        ... )
    """

    @staticmethod
    def create(
        html: str,
        uri: str | None = None,
        encoding: Literal["text", "blob"] = "text",
        metadata: UIMetadata | None = None,
        enable_apps_sdk: bool = False,
        apps_sdk_config: dict[str, Any] | None = None,
    ) -> UIResource:
        """Create a raw HTML resource.

        Args:
            html: HTML content string.
            uri: Resource URI (auto-generated if not provided).
            encoding: Content encoding type.
            metadata: Optional rendering metadata.
            enable_apps_sdk: Enable Apps SDK adapter.
            apps_sdk_config: Apps SDK configuration.

        Returns:
            UIResource with raw HTML content.
        """
        if not uri:
            uri = f"ui://html/{uuid.uuid4().hex[:8]}"

        # Apply adapters if enabled
        if enable_apps_sdk:
            html = wrap_html_with_adapters(
                html,
                {
                    "appsSdk": {
                        "enabled": True,
                        "config": apps_sdk_config or {},
                    }
                },
            )

        # Create content
        if encoding == "blob":
            content = UIResourceContent.from_blob(html.encode("utf-8"))
        else:
            content = UIResourceContent.from_text(html)

        return UIResource(
            uri=uri,
            mime_type=MimeType.TEXT_HTML,
            content=content,
            metadata=metadata,
        )


class ExternalUrlResource:
    """Factory for creating external URL UI resources.

    Creates resources that embed external applications via iframe src.

    Example:
        >>> resource = ExternalUrlResource.create(
        ...     url="https://example.com/widget",
        ...     uri="ui://external/1",
        ... )
    """

    @staticmethod
    def create(
        url: str,
        uri: str | None = None,
        metadata: UIMetadata | None = None,
    ) -> UIResource:
        """Create an external URL resource.

        Args:
            url: URL to embed in iframe.
            uri: Resource URI (auto-generated if not provided).
            metadata: Optional rendering metadata.

        Returns:
            UIResource with external URL content.

        Note:
            Only the first valid HTTP(S) URL is processed.
            The system warns if additional URLs are present.
        """
        if not uri:
            uri = f"ui://external/{uuid.uuid4().hex[:8]}"

        # Validate and normalize URL
        if not url.startswith(("http://", "https://")):
            raise ValueError("External URL must start with http:// or https://")

        # text/uri-list format: one URL per line
        content = UIResourceContent.from_text(url)

        return UIResource(
            uri=uri,
            mime_type=MimeType.TEXT_URI_LIST,
            content=content,
            metadata=metadata,
        )


class RemoteDomResource:
    """Factory for creating Remote DOM UI resources.

    Creates resources that use sandboxed script execution with
    Remote DOM for dynamic, host-styled components.

    Example:
        >>> script = '''
        ... const root = document.getElementById('root');
        ... root.innerHTML = '<h1>Dynamic Content</h1>';
        ... '''
        >>> resource = RemoteDomResource.create(
        ...     script=script,
        ...     framework="react",
        ...     uri="ui://dynamic/1",
        ... )
    """

    @staticmethod
    def create(
        script: str,
        framework: Literal["react", "webcomponents"] = "react",
        uri: str | None = None,
        encoding: Literal["text", "blob"] = "text",
        metadata: UIMetadata | None = None,
    ) -> UIResource:
        """Create a Remote DOM resource.

        Args:
            script: JavaScript content for Remote DOM execution.
            framework: Target framework (react or webcomponents).
            uri: Resource URI (auto-generated if not provided).
            encoding: Content encoding type.
            metadata: Optional rendering metadata.

        Returns:
            UIResource with Remote DOM script content.
        """
        if not uri:
            uri = f"ui://remote-dom/{uuid.uuid4().hex[:8]}"

        # Create content
        if encoding == "blob":
            content = UIResourceContent.from_blob(script.encode("utf-8"))
        else:
            content = UIResourceContent.from_text(script)

        mime_type = MimeType.remote_dom(framework)

        return UIResource(
            uri=uri,
            mime_type=mime_type,
            content=content,
            metadata=metadata,
        )


def create_html_resource(
    html: str,
    uri: str | None = None,
    **kwargs: Any,
) -> UIResource:
    """Convenience function for creating raw HTML resources.

    Args:
        html: HTML content.
        uri: Optional resource URI.
        **kwargs: Additional arguments passed to RawHtmlResource.create.

    Returns:
        UIResource with raw HTML.
    """
    return RawHtmlResource.create(html=html, uri=uri, **kwargs)


def create_url_resource(
    url: str,
    uri: str | None = None,
    **kwargs: Any,
) -> UIResource:
    """Convenience function for creating external URL resources.

    Args:
        url: External URL to embed.
        uri: Optional resource URI.
        **kwargs: Additional arguments passed to ExternalUrlResource.create.

    Returns:
        UIResource with external URL.
    """
    return ExternalUrlResource.create(url=url, uri=uri, **kwargs)


def create_remote_dom_resource(
    script: str,
    framework: Literal["react", "webcomponents"] = "react",
    uri: str | None = None,
    **kwargs: Any,
) -> UIResource:
    """Convenience function for creating Remote DOM resources.

    Args:
        script: JavaScript for Remote DOM.
        framework: Target framework.
        uri: Optional resource URI.
        **kwargs: Additional arguments passed to RemoteDomResource.create.

    Returns:
        UIResource with Remote DOM script.
    """
    return RemoteDomResource.create(
        script=script,
        framework=framework,
        uri=uri,
        **kwargs,
    )
