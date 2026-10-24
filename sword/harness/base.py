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

    def get_system_prompt(self) -> str:
        """Returns the authentic harness system prompt loaded from the cloned repository."""
        from sword.harness.prompts import get_harness_system_prompt
        return get_harness_system_prompt(self.name)

    def format_prompt(
        self,
        problem: str,
        max_prompt_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Combines authentic harness system prompt and tool guidelines with user problem.
        Applies a token/character budget check so the user task and tool declarations are never truncated.
        """
        sys_prompt = self.get_system_prompt()
        if not sys_prompt:
            return problem

        task_block = f"\n\nTask:\n{problem}"
        budget = max_prompt_chars
        if budget is None and max_tokens is not None:
            budget = max(256, int(max_tokens) * 3)

        if budget is not None and len(sys_prompt) + len(task_block) > budget:
            avail = max(128, budget - len(task_block) - 32)
            if len(sys_prompt) > avail:
                head_len = avail // 2
                tail_len = avail - head_len
                sys_prompt = sys_prompt[:head_len] + "\n...[trimmed]...\n" + sys_prompt[-tail_len:]

        return f"{sys_prompt}{task_block}"

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 3,
        task_id: str = "",
    ) -> HarnessResult:
        """Executes the agentic interaction loop and records tool telemetry."""
        raise NotImplementedError
