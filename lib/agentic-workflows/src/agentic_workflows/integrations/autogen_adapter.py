"""AutoGen Adapter for Agentic Workflows.

Provides integration with Microsoft AutoGen framework.

Usage:
    from agentic_workflows.integrations.autogen_adapter import AutoGenAdapter, AutoGenAgent

    adapter = AutoGenAdapter()
    autogen_agent = adapter.to_autogen(our_agent)
    result = adapter.run_conversation([agent1, agent2], message)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class AutoGenAgentConfig:
    """Configuration for AutoGen agent."""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    seed: Optional[int] = None
    timeout: int = 600
    max_consecutive_auto_reply: int = 10


@dataclass
class AutoGenAgent:
    """AutoGen-compatible agent definition.

    Maps to autogen.ConversableAgent interface.
    """

    name: str
    system_message: str = ""
    llm_config: Optional[Dict[str, Any]] = None
    human_input_mode: str = "NEVER"  # ALWAYS, TERMINATE, NEVER
    is_termination_msg: Optional[Callable[[Dict], bool]] = None
    code_execution_config: Optional[Dict[str, Any]] = None
    function_map: Optional[Dict[str, Callable]] = None

    def __post_init__(self):
        """Set defaults."""
        if self.llm_config is None:
            self.llm_config = {
                "model": "gpt-4o",
                "temperature": 0.7,
            }


class AutoGenAdapter:
    """Adapter for Microsoft AutoGen framework.

    Example:
        adapter = AutoGenAdapter()

        # Convert our agent to AutoGen agent
        autogen_agent = adapter.to_autogen(our_agent)

        # Create a group chat
        chat = adapter.create_group_chat([agent1, agent2])

        # Run conversation
        result = adapter.run_conversation(chat, "Hello")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter.

        Args:
            config: Optional LLM configuration.
        """
        self.config = config or {}
        self._autogen_available = self._check_autogen()

    def _check_autogen(self) -> bool:
        """Check if AutoGen is available."""
        try:
            import autogen
            return True
        except ImportError:
            logger.warning("autogen package not installed")
            return False

    def to_autogen(self, agent: Any) -> AutoGenAgent:
        """Convert agent to AutoGen format.

        Args:
            agent: Our agent definition.

        Returns:
            AutoGenAgent compatible with AutoGen.
        """
        name = getattr(agent, "name", "agent")
        instructions = getattr(agent, "instructions", "")

        # Build LLM config
        llm_config = {
            "model": getattr(agent, "model", "gpt-4o"),
            "temperature": 0.7,
        }

        return AutoGenAgent(
            name=name,
            system_message=instructions,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

    def from_autogen(self, autogen_agent: Any) -> Dict[str, Any]:
        """Convert AutoGen agent to our format.

        Args:
            autogen_agent: AutoGen agent instance.

        Returns:
            Dictionary in our agent format.
        """
        return {
            "name": getattr(autogen_agent, "name", "agent"),
            "instructions": getattr(autogen_agent, "system_message", ""),
            "model": autogen_agent.llm_config.get("model", "gpt-4o") if autogen_agent.llm_config else "gpt-4o",
        }

    def create_assistant(
        self,
        name: str,
        system_message: str,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create an AutoGen AssistantAgent.

        Args:
            name: Agent name.
            system_message: System prompt.
            llm_config: Optional LLM configuration.

        Returns:
            AutoGen AssistantAgent.
        """
        if not self._autogen_available:
            raise RuntimeError("autogen package not installed")

        from autogen import AssistantAgent

        return AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config or self.config,
        )

    def create_user_proxy(
        self,
        name: str = "user_proxy",
        human_input_mode: str = "NEVER",
        code_execution_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create an AutoGen UserProxyAgent.

        Args:
            name: Agent name.
            human_input_mode: Input mode (ALWAYS/TERMINATE/NEVER).
            code_execution_config: Optional code execution config.

        Returns:
            AutoGen UserProxyAgent.
        """
        if not self._autogen_available:
            raise RuntimeError("autogen package not installed")

        from autogen import UserProxyAgent

        return UserProxyAgent(
            name=name,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config or {"work_dir": "workspace"},
        )

    def create_group_chat(
        self,
        agents: List[Any],
        max_round: int = 10,
        speaker_selection_method: str = "auto",
    ) -> Any:
        """Create an AutoGen GroupChat.

        Args:
            agents: List of agents.
            max_round: Maximum conversation rounds.
            speaker_selection_method: How to select next speaker.

        Returns:
            AutoGen GroupChat.
        """
        if not self._autogen_available:
            raise RuntimeError("autogen package not installed")

        from autogen import GroupChat, GroupChatManager

        # Convert agents if needed
        autogen_agents = []
        for agent in agents:
            if isinstance(agent, AutoGenAgent):
                autogen_agents.append(self.create_assistant(
                    name=agent.name,
                    system_message=agent.system_message,
                    llm_config=agent.llm_config,
                ))
            elif hasattr(agent, "name") and hasattr(agent, "llm_config"):
                # Already an AutoGen agent
                autogen_agents.append(agent)
            else:
                # Convert from our format
                ag = self.to_autogen(agent)
                autogen_agents.append(self.create_assistant(
                    name=ag.name,
                    system_message=ag.system_message,
                    llm_config=ag.llm_config,
                ))

        # Create group chat
        group_chat = GroupChat(
            agents=autogen_agents,
            messages=[],
            max_round=max_round,
            speaker_selection_method=speaker_selection_method,
        )

        # Create manager
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.config,
        )

        return manager

    def run_conversation(
        self,
        agent: Any,
        recipient: Any,
        message: str,
        max_turns: int = 10,
    ) -> str:
        """Run a conversation between agents.

        Args:
            agent: Initiating agent.
            recipient: Receiving agent.
            message: Initial message.
            max_turns: Maximum conversation turns.

        Returns:
            Conversation result.
        """
        if not self._autogen_available:
            raise RuntimeError("autogen package not installed")

        chat_result = agent.initiate_chat(
            recipient,
            message=message,
            max_turns=max_turns,
        )

        return chat_result.summary if hasattr(chat_result, "summary") else str(chat_result)


__all__ = [
    "AutoGenAgentConfig",
    "AutoGenAgent",
    "AutoGenAdapter",
]
