"""
Base schemas and abstract contracts for agentic harnesses.
Tracks tool calls, tool success rates, reasoning traces, and execution results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable


@dataclass
class HarnessResult:
    """Detailed telemetry captured from agentic harness execution."""
    task_id: str = ""
    prompt: str = ""
    final_text: str = ""
    reasoning_text: str = ""
    tools_called: int = 0
    tools_succeeded: int = 0
    tools_failed: int = 0
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    success_rate: float = 0.0
    is_completed: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tools_called": self.tools_called,
            "tools_succeeded": self.tools_succeeded,
            "tools_failed": self.tools_failed,
            "success_rate": round(self.success_rate, 4),
            "is_completed": self.is_completed,
            "error_message": self.error_message,
        }


class BaseHarness:
    """Abstract base class for all CLI tool-calling harnesses."""
    name: str = "base"

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 3,
        task_id: str = "",
    ) -> HarnessResult:
        """Executes the agentic interaction loop and records tool telemetry."""
        raise NotImplementedError
