"""LangChain specialist agent for LLM orchestration.

Handles chain execution, tool integration, and prompt management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability


@dataclass
class LangChainConfig(SpecialistConfig):
    """LangChain-specific configuration."""

    default_model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    verbose: bool = False
    memory_type: str = "buffer"  # buffer, summary, conversation
    custom_tools: list[dict[str, Any]] = field(default_factory=list)


class LangChainAgent(SpecialistAgent):
    """Specialist agent for LangChain operations.

    Capabilities:
    - Chain execution
    - Tool integration
    - Memory management
    - Prompt templating
    """

    def __init__(self, config: LangChainConfig | None = None, **kwargs):
        self.lc_config = config or LangChainConfig()
        super().__init__(config=self.lc_config, **kwargs)

        self._llm = None
        self._memory = None
        self._tools: dict[str, Any] = {}
        self._chains: dict[str, Any] = {}

        self.register_handler("run_chain", self._run_chain)
        self.register_handler("create_chain", self._create_chain)
        self.register_handler("register_tool", self._register_tool)
        self.register_handler("run_agent", self._run_agent)
        self.register_handler("get_memory", self._get_memory)
        self.register_handler("clear_memory", self._clear_memory)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.LLM_ORCHESTRATION,
            SpecialistCapability.CHAIN_EXECUTION,
            SpecialistCapability.TOOL_INTEGRATION,
        ]

    @property
    def service_name(self) -> str:
        return "LangChain"

    async def _connect(self) -> None:
        """Initialize LangChain components."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain.memory import ConversationBufferMemory

            self._llm = ChatOpenAI(
                model=self.lc_config.default_model,
                temperature=self.lc_config.temperature,
                max_tokens=self.lc_config.max_tokens,
            )

            self._memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
            )

        except ImportError:
            self.logger.warning("langchain not installed")

    async def _disconnect(self) -> None:
        """Cleanup LangChain resources."""
        self._llm = None
        self._memory = None
        self._tools.clear()
        self._chains.clear()

    async def _health_check(self) -> bool:
        """Check LangChain health."""
        return self._llm is not None

    async def _run_chain(
        self,
        chain_name: str,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a registered chain.

        Args:
            chain_name: Name of the chain to run.
            inputs: Chain inputs.

        Returns:
            Chain output.
        """
        if chain_name not in self._chains:
            return {"error": f"Chain not found: {chain_name}"}

        try:
            chain = self._chains[chain_name]
            result = await chain.ainvoke(inputs)
            return {"output": result}
        except Exception as e:
            return {"error": str(e)}

    async def _create_chain(
        self,
        chain_name: str,
        chain_type: str = "llm",
        prompt_template: str | None = None,
        tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new chain.

        Args:
            chain_name: Name for the chain.
            chain_type: Type of chain (llm, sequential, agent).
            prompt_template: Optional prompt template.
            tools: Tool names to include.

        Returns:
            Chain creation result.
        """
        if self._llm is None:
            return {"error": "LLM not initialized"}

        try:
            if chain_type == "llm":
                from langchain.chains import LLMChain
                from langchain.prompts import PromptTemplate

                if prompt_template:
                    prompt = PromptTemplate.from_template(prompt_template)
                    chain = LLMChain(llm=self._llm, prompt=prompt)
                else:
                    chain = self._llm

            elif chain_type == "agent":
                from langchain.agents import create_openai_functions_agent, AgentExecutor
                from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

                agent_tools = [self._tools[t] for t in (tools or []) if t in self._tools]

                prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt_template or "You are a helpful assistant."),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])

                agent = create_openai_functions_agent(self._llm, agent_tools, prompt)
                chain = AgentExecutor(agent=agent, tools=agent_tools, memory=self._memory)

            else:
                return {"error": f"Unknown chain type: {chain_type}"}

            self._chains[chain_name] = chain
            return {"chain": chain_name, "created": True, "type": chain_type}

        except Exception as e:
            return {"error": str(e)}

    async def _register_tool(
        self,
        name: str,
        description: str,
        func: Callable | None = None,
        args_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a tool for agent use.

        Args:
            name: Tool name.
            description: Tool description.
            func: Tool function.
            args_schema: Argument schema.

        Returns:
            Registration result.
        """
        try:
            from langchain.tools import Tool, StructuredTool

            if func:
                if args_schema:
                    from pydantic import create_model

                    fields = {k: (v["type"], ...) for k, v in args_schema.items()}
                    ArgsModel = create_model(f"{name}Args", **fields)
                    tool = StructuredTool.from_function(
                        func=func,
                        name=name,
                        description=description,
                        args_schema=ArgsModel,
                    )
                else:
                    tool = Tool(name=name, description=description, func=func)

                self._tools[name] = tool
                return {"tool": name, "registered": True}
            else:
                return {"error": "No function provided"}

        except Exception as e:
            return {"error": str(e)}

    async def _run_agent(
        self,
        input_text: str,
        tools: list[str] | None = None,
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """Run an agent with tools.

        Args:
            input_text: User input.
            tools: Tool names to use.
            max_iterations: Maximum iterations.

        Returns:
            Agent output.
        """
        if self._llm is None:
            return {"error": "LLM not initialized"}

        try:
            from langchain.agents import create_openai_functions_agent, AgentExecutor
            from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

            agent_tools = [self._tools[t] for t in (tools or list(self._tools.keys())) if t in self._tools]

            if not agent_tools:
                # Direct LLM call if no tools
                result = await self._llm.ainvoke(input_text)
                return {"output": result.content}

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent = create_openai_functions_agent(self._llm, agent_tools, prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=agent_tools,
                memory=self._memory,
                max_iterations=max_iterations,
                verbose=self.lc_config.verbose,
            )

            result = await executor.ainvoke({"input": input_text})
            return {"output": result.get("output", "")}

        except Exception as e:
            return {"error": str(e)}

    async def _get_memory(self) -> dict[str, Any]:
        """Get current conversation memory.

        Returns:
            Memory contents.
        """
        if self._memory is None:
            return {"error": "Memory not initialized"}

        return {
            "messages": self._memory.chat_memory.messages if hasattr(self._memory, "chat_memory") else [],
        }

    async def _clear_memory(self) -> dict[str, Any]:
        """Clear conversation memory.

        Returns:
            Confirmation.
        """
        if self._memory is None:
            return {"error": "Memory not initialized"}

        self._memory.clear()
        return {"cleared": True}
