"""MCP-UI Action Handlers.

This module provides functions and handlers for processing UI actions
triggered from MCP-UI components.

Functions include:
- ui_action_result_tool_call(): Create tool call action result
- ui_action_result_prompt(): Create prompt action result
- ui_action_result_link(): Create link action result
- ui_action_result_intent(): Create intent action result
- ui_action_result_notification(): Create notification action result
- UIActionHandler: Handler class for processing UI actions

Based on MCP-UI SDK onUIAction callback specification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from .ui_types import (
    UIActionType,
    UIActionResult,
    UIActionPayload,
    UIActionToolPayload,
    UIActionIntentPayload,
    UIActionPromptPayload,
    UIActionNotifyPayload,
    UIActionLinkPayload,
)

logger = logging.getLogger(__name__)


# Type aliases for action handlers
ToolCallHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]
IntentHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]
PromptHandler = Callable[[str], Awaitable[Any]]
NotifyHandler = Callable[[str], Awaitable[None]]
LinkHandler = Callable[[str], Awaitable[None]]


def ui_action_result_tool_call(
    tool_name: str,
    params: dict[str, Any] | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Create a tool call action result.

    Creates an action result that requests the host to invoke
    a specific tool with the given parameters.

    Args:
        tool_name: Name of the tool to invoke.
        params: Parameters to pass to the tool.
        message_id: Optional message ID for async response correlation.

    Returns:
        Action result dictionary ready for JSON serialization.

    Example:
        >>> result = ui_action_result_tool_call(
        ...     "search_products",
        ...     {"query": "laptop", "category": "electronics"},
        ... )
        >>> # Returns: {"type": "tool", "payload": {"toolName": "search_products", ...}}
    """
    return UIActionResult.tool_call(
        tool_name=tool_name,
        params=params,
        message_id=message_id,
    ).to_dict()


def ui_action_result_prompt(
    prompt: str,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Create a prompt action result.

    Creates an action result that sends a prompt to the LLM
    for processing.

    Args:
        prompt: Prompt text to send to the LLM.
        message_id: Optional message ID for async response correlation.

    Returns:
        Action result dictionary ready for JSON serialization.

    Example:
        >>> result = ui_action_result_prompt(
        ...     "Explain the search results in more detail",
        ... )
        >>> # Returns: {"type": "prompt", "payload": {"prompt": "..."}}
    """
    return UIActionResult.prompt(
        prompt=prompt,
        message_id=message_id,
    ).to_dict()


def ui_action_result_link(
    url: str,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Create a link action result.

    Creates an action result that requests navigation to a URL.
    Note: Link support varies across host implementations.

    Args:
        url: URL to navigate to.
        message_id: Optional message ID for async response correlation.

    Returns:
        Action result dictionary ready for JSON serialization.

    Example:
        >>> result = ui_action_result_link(
        ...     "https://example.com/product/123",
        ... )
        >>> # Returns: {"type": "link", "payload": {"url": "..."}}
    """
    return UIActionResult.link(
        url=url,
        message_id=message_id,
    ).to_dict()


def ui_action_result_intent(
    intent: str,
    params: dict[str, Any] | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Create an intent action result.

    Creates an action result that triggers a named intent with
    optional parameters. Intents are typically translated to
    prompts by the Apps SDK adapter.

    Args:
        intent: Intent identifier (e.g., "add_to_cart", "checkout").
        params: Parameters for the intent.
        message_id: Optional message ID for async response correlation.

    Returns:
        Action result dictionary ready for JSON serialization.

    Example:
        >>> result = ui_action_result_intent(
        ...     "add_to_cart",
        ...     {"product_id": "123", "quantity": 2},
        ... )
        >>> # Returns: {"type": "intent", "payload": {"intent": "...", "params": {...}}}
    """
    return UIActionResult.intent(
        intent=intent,
        params=params,
        message_id=message_id,
    ).to_dict()


def ui_action_result_notification(
    message: str,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Create a notification action result.

    Creates an action result that displays a notification message
    to the user (e.g., toast notification).

    Args:
        message: Notification message to display.
        message_id: Optional message ID for async response correlation.

    Returns:
        Action result dictionary ready for JSON serialization.

    Example:
        >>> result = ui_action_result_notification(
        ...     "Item added to cart successfully!",
        ... )
        >>> # Returns: {"type": "notify", "payload": {"message": "..."}}
    """
    return UIActionResult.notification(
        message=message,
        message_id=message_id,
    ).to_dict()


@dataclass
class UIActionHandlerConfig:
    """Configuration for UIActionHandler.

    Attributes:
        allow_tool_calls: Allow tool call actions.
        allow_intents: Allow intent actions.
        allow_prompts: Allow prompt actions.
        allow_links: Allow link actions.
        allow_notifications: Allow notification actions.
        intent_to_prompt: Convert intents to prompts automatically.
        log_actions: Log all incoming actions.
    """
    allow_tool_calls: bool = True
    allow_intents: bool = True
    allow_prompts: bool = True
    allow_links: bool = True
    allow_notifications: bool = True
    intent_to_prompt: bool = True
    log_actions: bool = True


class UIActionHandler:
    """Handler for processing UI actions from MCP-UI components.

    Provides routing of UI actions to appropriate handlers based
    on action type, with support for async responses and middleware.

    Example:
        >>> handler = UIActionHandler()
        >>>
        >>> @handler.on_tool_call
        ... async def handle_tool(tool_name: str, params: dict) -> Any:
        ...     return await execute_tool(tool_name, params)
        >>>
        >>> @handler.on_prompt
        ... async def handle_prompt(prompt: str) -> Any:
        ...     return await send_to_llm(prompt)
        >>>
        >>> # Process an action
        >>> result = await handler.handle({
        ...     "type": "tool",
        ...     "payload": {"toolName": "search", "params": {}},
        ... })
    """

    def __init__(self, config: UIActionHandlerConfig | None = None):
        """Initialize action handler.

        Args:
            config: Handler configuration.
        """
        self.config = config or UIActionHandlerConfig()
        self._tool_handlers: list[ToolCallHandler] = []
        self._intent_handlers: list[IntentHandler] = []
        self._prompt_handlers: list[PromptHandler] = []
        self._notify_handlers: list[NotifyHandler] = []
        self._link_handlers: list[LinkHandler] = []
        self._pending_responses: dict[str, asyncio.Future] = {}

    def on_tool_call(self, handler: ToolCallHandler) -> ToolCallHandler:
        """Decorator to register a tool call handler.

        Args:
            handler: Async function that handles tool calls.

        Returns:
            The registered handler.

        Example:
            >>> @handler.on_tool_call
            ... async def handle_tool(tool_name: str, params: dict) -> Any:
            ...     if tool_name == "search":
            ...         return await search(params.get("query"))
        """
        self._tool_handlers.append(handler)
        return handler

    def on_intent(self, handler: IntentHandler) -> IntentHandler:
        """Decorator to register an intent handler.

        Args:
            handler: Async function that handles intents.

        Returns:
            The registered handler.

        Example:
            >>> @handler.on_intent
            ... async def handle_intent(intent: str, params: dict) -> Any:
            ...     if intent == "add_to_cart":
            ...         return await add_item(params.get("product_id"))
        """
        self._intent_handlers.append(handler)
        return handler

    def on_prompt(self, handler: PromptHandler) -> PromptHandler:
        """Decorator to register a prompt handler.

        Args:
            handler: Async function that handles prompts.

        Returns:
            The registered handler.

        Example:
            >>> @handler.on_prompt
            ... async def handle_prompt(prompt: str) -> Any:
            ...     return await send_to_llm(prompt)
        """
        self._prompt_handlers.append(handler)
        return handler

    def on_notify(self, handler: NotifyHandler) -> NotifyHandler:
        """Decorator to register a notification handler.

        Args:
            handler: Async function that handles notifications.

        Returns:
            The registered handler.

        Example:
            >>> @handler.on_notify
            ... async def handle_notify(message: str) -> None:
            ...     await show_toast(message)
        """
        self._notify_handlers.append(handler)
        return handler

    def on_link(self, handler: LinkHandler) -> LinkHandler:
        """Decorator to register a link handler.

        Args:
            handler: Async function that handles link navigation.

        Returns:
            The registered handler.

        Example:
            >>> @handler.on_link
            ... async def handle_link(url: str) -> None:
            ...     await open_browser(url)
        """
        self._link_handlers.append(handler)
        return handler

    async def handle(self, action_data: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming UI action.

        Parses the action data, validates it against configuration,
        and routes to appropriate handlers.

        Args:
            action_data: Action dictionary from UI component.

        Returns:
            Response dictionary with status and optional result.

        Example:
            >>> result = await handler.handle({
            ...     "type": "tool",
            ...     "payload": {"toolName": "search", "params": {"q": "test"}},
            ...     "messageId": "msg_123",
            ... })
        """
        action_type = action_data.get("type")
        payload = action_data.get("payload", {})
        message_id = action_data.get("messageId")

        if self.config.log_actions:
            logger.info(f"UI Action: type={action_type}, messageId={message_id}")

        try:
            result = await self._dispatch_action(action_type, payload)
            response = {
                "status": "success",
                "result": result,
            }
        except PermissionError as e:
            response = {
                "status": "error",
                "error": str(e),
                "code": "permission_denied",
            }
        except Exception as e:
            logger.exception(f"Action handler error: {action_type}")
            response = {
                "status": "error",
                "error": str(e),
                "code": "handler_error",
            }

        # Include message ID for correlation
        if message_id:
            response["messageId"] = message_id

        return response

    async def _dispatch_action(
        self,
        action_type: str | None,
        payload: dict[str, Any],
    ) -> Any:
        """Dispatch action to appropriate handlers."""
        if action_type == UIActionType.TOOL.value:
            if not self.config.allow_tool_calls:
                raise PermissionError("Tool calls are not allowed")
            return await self._handle_tool_call(payload)

        elif action_type == UIActionType.INTENT.value:
            if not self.config.allow_intents:
                raise PermissionError("Intents are not allowed")
            return await self._handle_intent(payload)

        elif action_type == UIActionType.PROMPT.value:
            if not self.config.allow_prompts:
                raise PermissionError("Prompts are not allowed")
            return await self._handle_prompt(payload)

        elif action_type == UIActionType.NOTIFY.value:
            if not self.config.allow_notifications:
                raise PermissionError("Notifications are not allowed")
            return await self._handle_notify(payload)

        elif action_type == UIActionType.LINK.value:
            if not self.config.allow_links:
                raise PermissionError("Links are not allowed")
            return await self._handle_link(payload)

        else:
            raise ValueError(f"Unknown action type: {action_type}")

    async def _handle_tool_call(self, payload: dict[str, Any]) -> Any:
        """Handle tool call action."""
        tool_name = payload.get("toolName", "")
        params = payload.get("params", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        results = []
        for handler in self._tool_handlers:
            result = await handler(tool_name, params)
            results.append(result)

        # Return first result or None
        return results[0] if results else None

    async def _handle_intent(self, payload: dict[str, Any]) -> Any:
        """Handle intent action."""
        intent = payload.get("intent", "")
        params = payload.get("params", {})

        if not intent:
            raise ValueError("Intent is required")

        # Convert to prompt if configured
        if self.config.intent_to_prompt and not self._intent_handlers:
            prompt = f"User intent: {intent}"
            if params:
                prompt += f"\nParameters: {json.dumps(params)}"
            return await self._handle_prompt({"prompt": prompt})

        results = []
        for handler in self._intent_handlers:
            result = await handler(intent, params)
            results.append(result)

        return results[0] if results else None

    async def _handle_prompt(self, payload: dict[str, Any]) -> Any:
        """Handle prompt action."""
        prompt = payload.get("prompt", "")

        if not prompt:
            raise ValueError("Prompt is required")

        results = []
        for handler in self._prompt_handlers:
            result = await handler(prompt)
            results.append(result)

        return results[0] if results else None

    async def _handle_notify(self, payload: dict[str, Any]) -> None:
        """Handle notification action."""
        message = payload.get("message", "")

        if not message:
            return

        for handler in self._notify_handlers:
            await handler(message)

    async def _handle_link(self, payload: dict[str, Any]) -> None:
        """Handle link action."""
        url = payload.get("url", "")

        if not url:
            raise ValueError("URL is required")

        for handler in self._link_handlers:
            await handler(url)

    def create_response_future(self, message_id: str | None = None) -> tuple[str, asyncio.Future]:
        """Create a future for awaiting async response.

        Use this when you need to wait for a response from the UI
        after sending an action.

        Args:
            message_id: Optional message ID (auto-generated if not provided).

        Returns:
            Tuple of (message_id, future) for response correlation.

        Example:
            >>> message_id, future = handler.create_response_future()
            >>> # Send action with message_id to UI
            >>> response = await asyncio.wait_for(future, timeout=30)
        """
        if not message_id:
            message_id = f"msg_{uuid.uuid4().hex[:12]}"

        future = asyncio.get_event_loop().create_future()
        self._pending_responses[message_id] = future
        return message_id, future

    def resolve_response(self, message_id: str, result: Any) -> bool:
        """Resolve a pending response future.

        Call this when receiving a response from the UI.

        Args:
            message_id: Message ID to resolve.
            result: Result to resolve with.

        Returns:
            True if a pending future was found and resolved.
        """
        future = self._pending_responses.pop(message_id, None)
        if future and not future.done():
            future.set_result(result)
            return True
        return False

    def reject_response(self, message_id: str, error: Exception) -> bool:
        """Reject a pending response future.

        Call this when an error occurs while processing a response.

        Args:
            message_id: Message ID to reject.
            error: Exception to reject with.

        Returns:
            True if a pending future was found and rejected.
        """
        future = self._pending_responses.pop(message_id, None)
        if future and not future.done():
            future.set_exception(error)
            return True
        return False


def create_action_handler(
    allow_tool_calls: bool = True,
    allow_intents: bool = True,
    allow_prompts: bool = True,
    allow_links: bool = True,
    allow_notifications: bool = True,
    intent_to_prompt: bool = True,
) -> UIActionHandler:
    """Factory function for creating an action handler.

    Args:
        allow_tool_calls: Allow tool call actions.
        allow_intents: Allow intent actions.
        allow_prompts: Allow prompt actions.
        allow_links: Allow link actions.
        allow_notifications: Allow notification actions.
        intent_to_prompt: Convert intents to prompts.

    Returns:
        Configured UIActionHandler instance.

    Example:
        >>> handler = create_action_handler(
        ...     allow_links=False,  # Disable link navigation
        ...     intent_to_prompt=True,  # Convert intents to prompts
        ... )
    """
    config = UIActionHandlerConfig(
        allow_tool_calls=allow_tool_calls,
        allow_intents=allow_intents,
        allow_prompts=allow_prompts,
        allow_links=allow_links,
        allow_notifications=allow_notifications,
        intent_to_prompt=intent_to_prompt,
    )
    return UIActionHandler(config)
