"""LlamaIndex Adapter for Agentic Workflows.

Provides integration with LlamaIndex for RAG and agents.

Usage:
    from agentic_workflows.integrations.llama_index import LlamaIndexAdapter, LlamaIndexAgent

    adapter = LlamaIndexAdapter()
    agent = adapter.create_agent(tools, index)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class LlamaIndexConfig:
    """Configuration for LlamaIndex."""

    model: str = "claude-3-sonnet-20240229"
    embed_model: str = "text-embedding-3-small"
    temperature: float = 0.1
    chunk_size: int = 1024
    chunk_overlap: int = 20


@dataclass
class LlamaIndexAgent:
    """LlamaIndex agent definition."""

    name: str
    system_prompt: str = ""
    tools: List[Any] = field(default_factory=list)
    config: LlamaIndexConfig = field(default_factory=LlamaIndexConfig)


class LlamaIndexAdapter:
    """Adapter for LlamaIndex framework.

    Example:
        adapter = LlamaIndexAdapter()

        # Create index from documents
        index = adapter.create_index(documents)

        # Create query engine
        engine = adapter.create_query_engine(index)

        # Create agent with tools
        agent = adapter.create_agent(
            tools=[query_tool],
            system_prompt="You are a helpful assistant.",
        )
    """

    def __init__(self, config: Optional[LlamaIndexConfig] = None):
        """Initialize adapter.

        Args:
            config: Optional configuration.
        """
        self.config = config or LlamaIndexConfig()
        self._llama_available = self._check_llama_index()

    def _check_llama_index(self) -> bool:
        """Check if LlamaIndex is available."""
        try:
            import llama_index
            return True
        except ImportError:
            logger.warning("llama-index package not installed")
            return False

    def configure(
        self,
        model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        """Configure LlamaIndex settings.

        Args:
            model: LLM model name.
            embed_model: Embedding model name.
        """
        if not self._llama_available:
            raise RuntimeError("llama-index package not installed")

        from llama_index.core import Settings

        if model:
            self.config.model = model
            # Set LLM based on model type
            if "claude" in model.lower():
                from llama_index.llms.anthropic import Anthropic
                Settings.llm = Anthropic(model=model)
            else:
                from llama_index.llms.openai import OpenAI
                Settings.llm = OpenAI(model=model)

        if embed_model:
            self.config.embed_model = embed_model
            from llama_index.embeddings.openai import OpenAIEmbedding
            Settings.embed_model = OpenAIEmbedding(model=embed_model)

        Settings.chunk_size = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap

    def load_documents(
        self,
        path: Union[str, Path],
        recursive: bool = True,
    ) -> List[Any]:
        """Load documents from a directory.

        Args:
            path: Directory path.
            recursive: Whether to load recursively.

        Returns:
            List of Document objects.
        """
        if not self._llama_available:
            raise RuntimeError("llama-index package not installed")

        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(
            input_dir=str(path),
            recursive=recursive,
        )
        return reader.load_data()

    def create_index(
        self,
        documents: List[Any],
        index_type: str = "vector",
    ) -> Any:
        """Create an index from documents.

        Args:
            documents: List of documents.
            index_type: Type of index (vector, list, tree).

        Returns:
            LlamaIndex index.
        """
        if not self._llama_available:
            raise RuntimeError("llama-index package not installed")

        from llama_index.core import (
            VectorStoreIndex,
            ListIndex,
            TreeIndex,
        )

        if index_type == "vector":
            return VectorStoreIndex.from_documents(documents)
        elif index_type == "list":
            return ListIndex.from_documents(documents)
        elif index_type == "tree":
            return TreeIndex.from_documents(documents)
        else:
            return VectorStoreIndex.from_documents(documents)

    def create_query_engine(
        self,
        index: Any,
        similarity_top_k: int = 3,
        response_mode: str = "compact",
    ) -> Any:
        """Create a query engine from an index.

        Args:
            index: LlamaIndex index.
            similarity_top_k: Number of similar docs to retrieve.
            response_mode: Response synthesis mode.

        Returns:
            Query engine.
        """
        if not self._llama_available:
            raise RuntimeError("llama-index package not installed")

        return index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode,
        )

    def create_tool(
        self,
        query_engine: Any,
        name: str,
        description: str,
    ) -> Any:
        """Create a tool from a query engine.

        Args:
            query_engine: Query engine.
            name: Tool name.
            description: Tool description.

        Returns:
            QueryEngineTool.
        """
        if not self._llama_available:
            raise RuntimeError("llama-index package not installed")

        from llama_index.core.tools import QueryEngineTool

        return QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=name,
            description=description,
        )

    def create_agent(
        self,
        tools: List[Any],
        system_prompt: str = "",
        verbose: bool = False,
    ) -> Any:
        """Create a LlamaIndex agent.

        Args:
            tools: List of tools.
            system_prompt: System prompt.
            verbose: Enable verbose output.

        Returns:
            LlamaIndex agent.
        """
        if not self._llama_available:
            raise RuntimeError("llama-index package not installed")

        from llama_index.core.agent import ReActAgent

        return ReActAgent.from_tools(
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose,
        )

    def to_llamaindex_agent(self, agent: Any) -> LlamaIndexAgent:
        """Convert our agent to LlamaIndex format.

        Args:
            agent: Our agent definition.

        Returns:
            LlamaIndexAgent.
        """
        name = getattr(agent, "name", "agent")
        instructions = getattr(agent, "instructions", "")
        tools = getattr(agent, "tools", [])

        return LlamaIndexAgent(
            name=name,
            system_prompt=instructions,
            tools=tools,
            config=self.config,
        )

    async def query(
        self,
        agent: Any,
        query: str,
    ) -> str:
        """Query an agent.

        Args:
            agent: LlamaIndex agent.
            query: Query string.

        Returns:
            Response string.
        """
        response = agent.chat(query)
        return str(response)


__all__ = [
    "LlamaIndexConfig",
    "LlamaIndexAgent",
    "LlamaIndexAdapter",
]
