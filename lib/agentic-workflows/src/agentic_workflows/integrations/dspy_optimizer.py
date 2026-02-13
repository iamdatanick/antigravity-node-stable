"""DSPy Optimizer for Agentic Workflows.

Provides integration with DSPy for prompt optimization.

Usage:
    from agentic_workflows.integrations.dspy_optimizer import DSPyOptimizer, DSPyModule

    optimizer = DSPyOptimizer()
    optimized = optimizer.compile(module, trainset)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DSPySignature:
    """DSPy signature definition.

    Maps to dspy.Signature interface.
    """

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    instructions: str = ""

    def to_string(self) -> str:
        """Convert to DSPy signature string format."""
        inputs = ", ".join(self.inputs)
        outputs = ", ".join(self.outputs)
        return f"{inputs} -> {outputs}"


@dataclass
class DSPyModule:
    """DSPy module definition.

    Represents a DSPy program/module for optimization.
    """

    name: str
    signature: DSPySignature
    forward_fn: Optional[Callable] = None
    predictor_type: str = "Predict"  # Predict, ChainOfThought, ProgramOfThought


@dataclass
class OptimizationConfig:
    """Configuration for DSPy optimization."""

    metric: Optional[Callable] = None
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    max_rounds: int = 1
    num_threads: int = 1
    teacher_model: Optional[str] = None


class DSPyOptimizer:
    """DSPy optimization integration.

    Example:
        optimizer = DSPyOptimizer()

        # Define a signature
        sig = DSPySignature(
            inputs=["question"],
            outputs=["answer"],
            instructions="Answer questions accurately",
        )

        # Create and optimize module
        module = DSPyModule(name="qa", signature=sig)
        optimized = optimizer.compile(module, trainset, metric=exact_match)
    """

    def __init__(self, model: Optional[str] = None):
        """Initialize optimizer.

        Args:
            model: Model to use (e.g., "claude-3-sonnet").
        """
        self.model = model
        self._dspy_available = self._check_dspy()

    def _check_dspy(self) -> bool:
        """Check if DSPy is available."""
        try:
            import dspy
            return True
        except ImportError:
            logger.warning("dspy package not installed")
            return False

    def configure(
        self,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 4096,
    ) -> None:
        """Configure DSPy with a language model.

        Args:
            model: Model name.
            max_tokens: Maximum tokens.
        """
        if not self._dspy_available:
            raise RuntimeError("dspy package not installed")

        import dspy

        # Configure LM - using Anthropic
        if "claude" in model.lower():
            lm = dspy.Claude(model=model, max_tokens=max_tokens)
        else:
            lm = dspy.OpenAI(model=model, max_tokens=max_tokens)

        dspy.configure(lm=lm)

    def create_signature(
        self,
        inputs: List[str],
        outputs: List[str],
        instructions: str = "",
    ) -> Any:
        """Create a DSPy signature.

        Args:
            inputs: Input field names.
            outputs: Output field names.
            instructions: Optional instructions.

        Returns:
            DSPy Signature class.
        """
        if not self._dspy_available:
            raise RuntimeError("dspy package not installed")

        import dspy

        # Create signature dynamically
        sig_str = f"{', '.join(inputs)} -> {', '.join(outputs)}"
        sig_class = dspy.Signature(sig_str)

        if instructions:
            sig_class.__doc__ = instructions

        return sig_class

    def create_predictor(
        self,
        signature: Any,
        predictor_type: str = "Predict",
    ) -> Any:
        """Create a DSPy predictor.

        Args:
            signature: DSPy Signature.
            predictor_type: Type of predictor.

        Returns:
            DSPy predictor module.
        """
        if not self._dspy_available:
            raise RuntimeError("dspy package not installed")

        import dspy

        if predictor_type == "ChainOfThought":
            return dspy.ChainOfThought(signature)
        elif predictor_type == "ProgramOfThought":
            return dspy.ProgramOfThought(signature)
        else:
            return dspy.Predict(signature)

    def compile(
        self,
        module: Any,
        trainset: List[Any],
        metric: Optional[Callable] = None,
        config: Optional[OptimizationConfig] = None,
    ) -> Any:
        """Compile/optimize a DSPy module.

        Args:
            module: DSPy module to optimize.
            trainset: Training examples.
            metric: Evaluation metric function.
            config: Optimization configuration.

        Returns:
            Optimized module.
        """
        if not self._dspy_available:
            raise RuntimeError("dspy package not installed")

        import dspy
        from dspy.teleprompt import BootstrapFewShot

        config = config or OptimizationConfig()

        # Create optimizer
        optimizer = BootstrapFewShot(
            metric=metric or (lambda x, y: True),
            max_bootstrapped_demos=config.max_bootstrapped_demos,
            max_labeled_demos=config.max_labeled_demos,
            max_rounds=config.max_rounds,
        )

        # Compile
        return optimizer.compile(module, trainset=trainset)

    def evaluate(
        self,
        module: Any,
        devset: List[Any],
        metric: Callable,
    ) -> Tuple[float, List[Any]]:
        """Evaluate a module on a dataset.

        Args:
            module: DSPy module.
            devset: Development set.
            metric: Evaluation metric.

        Returns:
            Tuple of (score, results).
        """
        if not self._dspy_available:
            raise RuntimeError("dspy package not installed")

        import dspy

        evaluator = dspy.Evaluate(
            devset=devset,
            metric=metric,
            num_threads=1,
            display_progress=True,
        )

        score = evaluator(module)
        return score, []

    def to_prompt(self, module: Any) -> str:
        """Extract optimized prompt from module.

        Args:
            module: Optimized DSPy module.

        Returns:
            Compiled prompt string.
        """
        if not self._dspy_available:
            raise RuntimeError("dspy package not installed")

        # Get the compiled prompt/demos
        if hasattr(module, "dump_state"):
            state = module.dump_state()
            return str(state)
        return str(module)


__all__ = [
    "DSPySignature",
    "DSPyModule",
    "OptimizationConfig",
    "DSPyOptimizer",
]
