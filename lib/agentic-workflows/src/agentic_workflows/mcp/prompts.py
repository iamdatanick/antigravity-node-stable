"""MCP Prompts Module.

Provides prompt management for MCP protocol.

Usage:
    from agentic_workflows.mcp.prompts import PromptManager, Prompt

    manager = PromptManager()
    manager.register(Prompt(
        name="code-review",
        description="Review code for issues",
        arguments=[
            PromptArgument(name="file", description="File to review", required=True),
        ],
    ))
    messages = await manager.get("code-review", {"file": "main.py"})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class PromptArgument:
    """Argument definition for a prompt.

    Attributes:
        name: Argument name.
        description: Human-readable description.
        required: Whether argument is required.
    """

    name: str
    description: Optional[str] = None
    required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format."""
        result = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.required:
            result["required"] = self.required
        return result


@dataclass
class PromptMessage:
    """A message in a prompt.

    Attributes:
        role: Message role (user, assistant).
        content: Message content.
    """

    role: str
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format."""
        content = self.content
        if isinstance(content, str):
            content = {"type": "text", "text": content}
        return {
            "role": self.role,
            "content": content,
        }


@dataclass
class Prompt:
    """MCP Prompt definition.

    Represents a reusable prompt template.

    Example:
        prompt = Prompt(
            name="code-review",
            description="Review code for bugs and style issues",
            arguments=[
                PromptArgument(name="file", description="File to review", required=True),
                PromptArgument(name="focus", description="Areas to focus on"),
            ],
        )
    """

    name: str
    description: Optional[str] = None
    arguments: List[PromptArgument] = field(default_factory=list)

    # Template for generating messages
    template: Optional[str] = None
    messages: Optional[List[PromptMessage]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format."""
        result = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.arguments:
            result["arguments"] = [a.to_dict() for a in self.arguments]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prompt":
        """Create from MCP format."""
        arguments = [
            PromptArgument(
                name=a["name"],
                description=a.get("description"),
                required=a.get("required", False),
            )
            for a in data.get("arguments", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description"),
            arguments=arguments,
        )

    def generate_messages(self, arguments: Dict[str, Any]) -> List[PromptMessage]:
        """Generate prompt messages from arguments.

        Args:
            arguments: Argument values.

        Returns:
            List of prompt messages.
        """
        if self.messages:
            # Use predefined messages with variable substitution
            result = []
            for msg in self.messages:
                content = msg.content
                if isinstance(content, str):
                    # Substitute {variable} placeholders
                    for key, value in arguments.items():
                        content = content.replace(f"{{{key}}}", str(value))
                result.append(PromptMessage(role=msg.role, content=content))
            return result

        elif self.template:
            # Use template string
            text = self.template
            for key, value in arguments.items():
                text = text.replace(f"{{{key}}}", str(value))
            return [PromptMessage(role="user", content=text)]

        else:
            # Generate default message
            text_parts = [f"{self.name}:"]
            for key, value in arguments.items():
                text_parts.append(f"- {key}: {value}")
            return [PromptMessage(role="user", content="\n".join(text_parts))]


# Type for prompt handlers
PromptHandler = Callable[[str, Dict[str, Any]], List[PromptMessage]]


class PromptManager:
    """Manages MCP prompts.

    Example:
        manager = PromptManager()

        # Register a prompt
        manager.register(Prompt(
            name="summarize",
            description="Summarize text",
            arguments=[
                PromptArgument(name="text", required=True),
                PromptArgument(name="max_length"),
            ],
            template="Please summarize the following text in {max_length} words or less:\\n\\n{text}",
        ))

        # Get prompt messages
        messages = await manager.get("summarize", {
            "text": "Long text here...",
            "max_length": "100",
        })
    """

    def __init__(self):
        """Initialize prompt manager."""
        self._prompts: Dict[str, Prompt] = {}
        self._handlers: Dict[str, PromptHandler] = {}

    def register(self, prompt: Prompt) -> None:
        """Register a prompt.

        Args:
            prompt: Prompt to register.
        """
        self._prompts[prompt.name] = prompt
        logger.debug(f"Registered prompt: {prompt.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a prompt.

        Args:
            name: Prompt name.

        Returns:
            True if removed, False if not found.
        """
        if name in self._prompts:
            del self._prompts[name]
            return True
        return False

    def set_handler(self, name: str, handler: PromptHandler) -> None:
        """Set custom handler for a prompt.

        Args:
            name: Prompt name.
            handler: Handler function.
        """
        self._handlers[name] = handler

    async def list(self, cursor: Optional[str] = None) -> Dict[str, Any]:
        """List available prompts.

        Args:
            cursor: Pagination cursor.

        Returns:
            MCP prompts/list response.
        """
        prompts = [p.to_dict() for p in self._prompts.values()]

        return {
            "prompts": prompts,
        }

    async def get(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get prompt messages.

        Args:
            name: Prompt name.
            arguments: Argument values.

        Returns:
            MCP prompts/get response.

        Raises:
            ValueError: If prompt not found.
        """
        prompt = self._prompts.get(name)
        if not prompt:
            raise ValueError(f"Prompt not found: {name}")

        arguments = arguments or {}

        # Validate required arguments
        for arg in prompt.arguments:
            if arg.required and arg.name not in arguments:
                raise ValueError(f"Missing required argument: {arg.name}")

        # Use custom handler if available
        if name in self._handlers:
            messages = self._handlers[name](name, arguments)
        else:
            messages = prompt.generate_messages(arguments)

        return {
            "description": prompt.description,
            "messages": [m.to_dict() for m in messages],
        }

    def get_prompt(self, name: str) -> Optional[Prompt]:
        """Get a prompt by name.

        Args:
            name: Prompt name.

        Returns:
            Prompt or None.
        """
        return self._prompts.get(name)

    def list_names(self) -> List[str]:
        """Get list of registered prompt names.

        Returns:
            List of names.
        """
        return list(self._prompts.keys())


# Built-in prompts
BUILTIN_PROMPTS: Dict[str, Prompt] = {
    "code-review": Prompt(
        name="code-review",
        description="Review code for bugs, security issues, and style",
        arguments=[
            PromptArgument(name="code", description="Code to review", required=True),
            PromptArgument(name="language", description="Programming language"),
            PromptArgument(name="focus", description="Areas to focus on"),
        ],
        template="""Please review the following {language} code for bugs, security issues, and style improvements.

{focus}

```{language}
{code}
```

Provide specific feedback with line numbers where applicable.""",
    ),
    "explain": Prompt(
        name="explain",
        description="Explain code or concept",
        arguments=[
            PromptArgument(name="topic", description="Topic to explain", required=True),
            PromptArgument(name="level", description="Expertise level (beginner/intermediate/expert)"),
        ],
        template="Please explain {topic} at a {level} level.",
    ),
    "refactor": Prompt(
        name="refactor",
        description="Suggest refactoring improvements",
        arguments=[
            PromptArgument(name="code", description="Code to refactor", required=True),
            PromptArgument(name="goal", description="Refactoring goal"),
        ],
        template="""Please suggest refactoring improvements for this code.

Goal: {goal}

```
{code}
```

Provide the refactored code with explanations for each change.""",
    ),
    "debug": Prompt(
        name="debug",
        description="Help debug an issue",
        arguments=[
            PromptArgument(name="error", description="Error message or issue", required=True),
            PromptArgument(name="code", description="Relevant code"),
            PromptArgument(name="context", description="Additional context"),
        ],
        template="""Please help debug this issue.

Error/Issue: {error}

{context}

```
{code}
```

Provide possible causes and solutions.""",
    ),
}


__all__ = [
    "Prompt",
    "PromptArgument",
    "PromptMessage",
    "PromptHandler",
    "PromptManager",
    "BUILTIN_PROMPTS",
]
