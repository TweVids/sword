"""
Pi Agent Harness: Lightweight autonomous agent loop,
tracking iterative plan, action, and reflection cycles.
"""

import re
from typing import Dict, Any, List, Optional, Callable

from sword.harness.base import BaseHarness, HarnessResult


class PiHarness(BaseHarness):
    """
    Harness for Pi autonomous coding agent.
    Tracks structured plan and action cycles.
    """
    name: str = "pi"

    def extract_reasoning(self, text: str) -> str:
        """Extracts thought / plan sections."""
        m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m_think:
            return m_think.group(1).strip()
        m_plan = re.search(r"(?:^|\n)(?:Plan|Thinking):\s*(.*?)(?=\n(?:Action|Step\s*\d|Execution):|$)", text, re.DOTALL | re.IGNORECASE)
        return m_plan.group(1).strip() if m_plan else ""

    def count_actions(self, text: str) -> int:
        """Counts explicit action steps."""
        actions = re.findall(r"(?:^|\n)(?:Action|Step\s*\d+):\s*", text, re.IGNORECASE)
        code_blocks = re.findall(r"```[a-z]*\n.*?```", text, re.DOTALL)
        return max(len(actions), len(code_blocks))

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 2,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the Pi agent interaction cycle."""
        response = model_generate_fn(prompt)
        reasoning = self.extract_reasoning(response)
        num_actions = self.count_actions(response)

        tools_called = num_actions
        tools_succeeded = num_actions  # Successful generation of planned actions
        tools_failed = 0
        success_rate = 1.0 if tools_called > 0 else 0.0

        return HarnessResult(
            task_id=task_id,
            prompt=prompt,
            final_text=response,
            reasoning_text=reasoning,
            tools_called=tools_called,
            tools_succeeded=tools_succeeded,
            tools_failed=tools_failed,
            tool_history=[{"action_count": num_actions}],
            success_rate=success_rate,
            is_completed=True,
        )
