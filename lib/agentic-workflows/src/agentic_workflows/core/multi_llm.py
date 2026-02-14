"""Multi-LLM Router with real Anthropic SDK integration.

Routes tasks to appropriate models based on complexity, cost, and capability.
Supports hierarchical review where stronger models validate weaker models.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Optional anthropic import - allows module to load without SDK
try:
    import anthropic
    from anthropic import AsyncAnthropic
    from anthropic.types import ContentBlock, Message, MessageParam

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None
    Message = None
    MessageParam = dict
    ContentBlock = None


class LLMProvider(Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    LOCAL = "local"


class ModelTier(Enum):
    """Model tiers by capability."""

    FAST = "fast"  # Haiku - quick tasks, classification
    BALANCED = "balanced"  # Sonnet - most tasks
    POWERFUL = "powerful"  # Opus - complex reasoning, review
    THINKING = "thinking"  # Extended thinking models


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: LLMProvider
    model_id: str
    tier: ModelTier
    max_tokens: int = 4096
    supports_tools: bool = True
    supports_vision: bool = True
    supports_caching: bool = True
    supports_extended_thinking: bool = False
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    # Extended thinking config
    thinking_budget_tokens: int = 10000


# Pre-configured Anthropic models
ANTHROPIC_MODELS = {
    "haiku": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-5-haiku-20241022",
        tier=ModelTier.FAST,
        max_tokens=4096,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.005,
    ),
    "sonnet": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        tier=ModelTier.BALANCED,
        max_tokens=8192,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "opus": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-opus-4-20250514",
        tier=ModelTier.POWERFUL,
        max_tokens=8192,
        supports_extended_thinking=True,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
}


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    model_config: ModelConfig
    reasoning: str
    estimated_tokens: int
    use_extended_thinking: bool = False
    use_caching: bool = False
    review_required: bool = False
    reviewer_model: str | None = None


@dataclass
class LLMResponse:
    """Unified response from any LLM."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    thinking_content: str | None = None
    tool_calls: list[dict] | None = None
    stop_reason: str | None = None
    cached_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_content": self.thinking_content,
            "tool_calls": self.tool_calls,
            "stop_reason": self.stop_reason,
            "cached_tokens": self.cached_tokens,
            "latency_ms": self.latency_ms,
        }


class MultiLLMRouter:
    """Routes requests to appropriate LLMs based on task requirements.

    Features:
    - Automatic model selection based on task complexity
    - Hierarchical review (stronger models check weaker ones)
    - Cost optimization
    - Extended thinking for complex reasoning
    - Prompt caching for efficiency
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        default_tier: ModelTier = ModelTier.BALANCED,
        enable_review: bool = True,
        enable_hierarchical_review: bool = True,
        review_threshold: float = 0.7,  # Confidence below this triggers review
    ):
        """Initialize the multi-LLM router.

        Args:
            anthropic_api_key: Anthropic API key (uses env var if not provided)
            default_tier: Default model tier for unclassified tasks
            enable_review: Enable hierarchical review
            enable_hierarchical_review: Alias for enable_review
            review_threshold: Confidence threshold for triggering review
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The 'anthropic' package is required for MultiLLMRouter. "
                "Install it with: pip install anthropic"
            )

        self.anthropic = AsyncAnthropic(api_key=anthropic_api_key)
        self.default_tier = default_tier
        self.enable_review = enable_review and enable_hierarchical_review
        self.review_threshold = review_threshold

        self.models = ANTHROPIC_MODELS.copy()

        # Usage tracking
        self._usage: dict[str, dict] = {}

        # Cache for routing decisions
        self._routing_cache: dict[str, RoutingDecision] = {}

    def classify_task_complexity(self, task: str, context: dict | None = None) -> ModelTier:
        """Classify task complexity to determine appropriate model.

        Args:
            task: The task description
            context: Optional context about the task

        Returns:
            Recommended model tier
        """
        task_lower = task.lower()

        # Fast tier indicators
        fast_indicators = [
            "classify",
            "categorize",
            "extract",
            "summarize briefly",
            "yes or no",
            "true or false",
            "simple",
            "quick",
            "format",
            "convert",
            "translate short",
        ]

        # Powerful tier indicators
        powerful_indicators = [
            "analyze deeply",
            "complex",
            "multi-step",
            "research",
            "design",
            "architect",
            "review code",
            "security audit",
            "debug",
            "optimize",
            "refactor large",
            "explain thoroughly",
            "creative writing",
            "long-form",
            "comprehensive",
        ]

        # Extended thinking indicators
        thinking_indicators = [
            "prove",
            "derive",
            "mathematical",
            "formal verification",
            "complex reasoning",
            "step by step logic",
            "theorem",
            "multi-constraint optimization",
            "game theory",
        ]

        # Check indicators
        if any(ind in task_lower for ind in thinking_indicators):
            return ModelTier.THINKING

        if any(ind in task_lower for ind in powerful_indicators):
            return ModelTier.POWERFUL

        if any(ind in task_lower for ind in fast_indicators):
            return ModelTier.FAST

        # Check task length as complexity proxy
        if len(task) > 2000:
            return ModelTier.POWERFUL
        elif len(task) < 200:
            return ModelTier.FAST

        return self.default_tier

    def get_routing_decision(
        self,
        task: str,
        context: dict | None = None,
        force_tier: ModelTier | None = None,
        require_tools: bool = False,
        require_vision: bool = False,
    ) -> RoutingDecision:
        """Get routing decision for a task.

        Args:
            task: Task description
            context: Optional context
            force_tier: Force a specific tier
            require_tools: Task requires tool use
            require_vision: Task requires vision

        Returns:
            Routing decision with model config
        """
        tier = force_tier or self.classify_task_complexity(task, context)

        # Select model based on tier
        if tier == ModelTier.FAST:
            model_key = "haiku"
        elif tier == ModelTier.THINKING:
            model_key = "opus"  # Opus supports extended thinking
        elif tier == ModelTier.POWERFUL:
            model_key = "opus"
        else:
            model_key = "sonnet"

        model_config = self.models[model_key]

        # Determine if extended thinking should be used
        use_thinking = tier == ModelTier.THINKING and model_config.supports_extended_thinking

        # Determine if review is needed
        review_required = self.enable_review and tier in (ModelTier.FAST, ModelTier.BALANCED)

        # Estimate tokens
        estimated_tokens = len(task.split()) * 2  # Rough estimate

        reasoning = f"Selected {model_key} ({tier.value}) based on task complexity analysis"
        if use_thinking:
            reasoning += " with extended thinking enabled"
        if review_required:
            reasoning += " with opus review"

        return RoutingDecision(
            model_config=model_config,
            reasoning=reasoning,
            estimated_tokens=estimated_tokens,
            use_extended_thinking=use_thinking,
            use_caching=model_config.supports_caching,
            review_required=review_required,
            reviewer_model="opus" if review_required else None,
        )

    async def call(
        self,
        messages: list[MessageParam],
        system: str | None = None,
        task_description: str | None = None,
        tools: list[dict] | None = None,
        force_tier: ModelTier | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call the appropriate LLM based on routing.

        Args:
            messages: Conversation messages
            system: System prompt
            task_description: Description for routing (uses first message if not provided)
            tools: Tools available to the model
            force_tier: Force a specific model tier
            temperature: Sampling temperature
            max_tokens: Maximum output tokens

        Returns:
            LLM response
        """
        # Determine task for routing
        if task_description:
            task = task_description
        elif messages:
            first_msg = messages[0]
            if isinstance(first_msg.get("content"), str):
                task = first_msg["content"]
            else:
                task = str(first_msg.get("content", ""))
        else:
            task = ""

        # Get routing decision
        decision = self.get_routing_decision(
            task=task,
            force_tier=force_tier,
            require_tools=bool(tools),
        )

        model_config = decision.model_config

        # Build API call parameters
        params: dict[str, Any] = {
            "model": model_config.model_id,
            "messages": messages,
            "max_tokens": max_tokens or model_config.max_tokens,
        }

        if system:
            # Use caching for system prompts when supported
            if decision.use_caching:
                params["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                params["system"] = system

        if tools:
            params["tools"] = tools

        # Extended thinking
        if decision.use_extended_thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": model_config.thinking_budget_tokens,
            }
            # Temperature must be 1 for extended thinking
            params["temperature"] = 1
        else:
            params["temperature"] = temperature

        # Make API call
        start_time = time.time()
        response = await self.anthropic.messages.create(**params)
        latency_ms = (time.time() - start_time) * 1000

        # Extract content
        content = ""
        thinking_content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "thinking":
                thinking_content = block.thinking
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        # Track usage
        self._track_usage(
            model_config.model_id,
            response.usage.input_tokens,
            response.usage.output_tokens,
            getattr(response.usage, "cache_read_input_tokens", 0),
        )

        result = LLMResponse(
            content=content,
            model=model_config.model_id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            thinking_content=thinking_content,
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=response.stop_reason,
            cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
            latency_ms=latency_ms,
        )

        # Hierarchical review if needed
        if decision.review_required and self.enable_review:
            result = await self._review_response(
                original_task=task,
                response=result,
                reviewer_model=decision.reviewer_model,
            )

        return result

    async def call_with_tools(
        self,
        messages: list[MessageParam],
        tools: list[dict],
        tool_executor: Callable[[str, dict], Any],
        system: str | None = None,
        max_iterations: int = 10,
        force_tier: ModelTier | None = None,
    ) -> LLMResponse:
        """Call LLM with automatic tool execution loop.

        Args:
            messages: Conversation messages
            tools: Available tools
            tool_executor: Function to execute tools
            system: System prompt
            max_iterations: Maximum tool iterations
            force_tier: Force model tier

        Returns:
            Final LLM response
        """
        current_messages = list(messages)

        for iteration in range(max_iterations):
            response = await self.call(
                messages=current_messages,
                system=system,
                tools=tools,
                force_tier=force_tier,
            )

            # Check if we're done
            if response.stop_reason != "tool_use" or not response.tool_calls:
                return response

            # Execute tools
            tool_results = []
            for tool_call in response.tool_calls:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: tool_executor(tool_call["name"], tool_call["input"])
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call["id"],
                            "content": str(result),
                        }
                    )
                except Exception as e:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call["id"],
                            "content": f"Error: {str(e)}",
                            "is_error": True,
                        }
                    )

            # Add assistant response and tool results
            current_messages.append(
                {
                    "role": "assistant",
                    "content": response.tool_calls,
                }
            )
            current_messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

        return response

    async def stream(
        self,
        messages: list[MessageParam],
        system: str | None = None,
        task_description: str | None = None,
        force_tier: ModelTier | None = None,
    ) -> AsyncIterator[str]:
        """Stream response from LLM.

        Args:
            messages: Conversation messages
            system: System prompt
            task_description: Description for routing
            force_tier: Force model tier

        Yields:
            Response text chunks
        """
        # Get routing decision
        task = task_description or str(messages[0].get("content", "") if messages else "")
        decision = self.get_routing_decision(task=task, force_tier=force_tier)

        params: dict[str, Any] = {
            "model": decision.model_config.model_id,
            "messages": messages,
            "max_tokens": decision.model_config.max_tokens,
        }

        if system:
            params["system"] = system

        async with self.anthropic.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    async def _review_response(
        self,
        original_task: str,
        response: LLMResponse,
        reviewer_model: str | None,
    ) -> LLMResponse:
        """Have a stronger model review and potentially improve a response.

        Args:
            original_task: The original task
            response: Response to review
            reviewer_model: Model to use for review

        Returns:
            Reviewed/improved response
        """
        if not reviewer_model:
            return response

        review_prompt = f"""You are reviewing another AI's response.

ORIGINAL TASK:
{original_task}

RESPONSE TO REVIEW:
{response.content}

Please evaluate this response:
1. Is it correct and complete?
2. Are there any errors or omissions?
3. Could it be improved?

If the response is good, respond with: "APPROVED: [brief reason]"
If it needs changes, provide the corrected/improved response.
"""

        review_response = await self.call(
            messages=[{"role": "user", "content": review_prompt}],
            force_tier=ModelTier.POWERFUL,
        )

        # Check if approved or needs changes
        if review_response.content.startswith("APPROVED:"):
            # Original is fine
            return response
        else:
            # Return improved response
            return LLMResponse(
                content=review_response.content,
                model=f"{response.model}+{review_response.model}",
                input_tokens=response.input_tokens + review_response.input_tokens,
                output_tokens=response.output_tokens + review_response.output_tokens,
                thinking_content=review_response.thinking_content,
                stop_reason=review_response.stop_reason,
                latency_ms=response.latency_ms + review_response.latency_ms,
            )

    def _track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
    ) -> None:
        """Track usage statistics."""
        if model not in self._usage:
            self._usage[model] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
            }

        self._usage[model]["calls"] += 1
        self._usage[model]["input_tokens"] += input_tokens
        self._usage[model]["output_tokens"] += output_tokens
        self._usage[model]["cached_tokens"] += cached_tokens

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        total_cost = 0.0

        for model_key, usage in self._usage.items():
            # Find model config
            for config in self.models.values():
                if config.model_id == model_key:
                    cost = (usage["input_tokens"] / 1000) * config.cost_per_1k_input + (
                        usage["output_tokens"] / 1000
                    ) * config.cost_per_1k_output
                    total_cost += cost
                    break

        return {
            "by_model": self._usage,
            "total_cost_usd": total_cost,
        }
