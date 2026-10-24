"""
Pi Agent Harness: Autonomous iterative agent loop,
tracking multi-turn plan, action, and reflection cycles.
"""

import re
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult


class PiHarness(BaseHarness):
    """
    Harness for Pi autonomous coding agent.
    Tracks structured plan, action, observation, and reflection cycles across multiple turns.
    """
    name: str = "pi"

    def extract_reasoning(self, text: str) -> str:
        """Extracts thought / plan sections."""
        m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m_think:
            return m_think.group(1).strip()
        m_chan = re.search(r"<\|channel>thought(.*?)(?:<channel\||<\|end_channel>|$)", text, re.DOTALL)
        if m_chan:
            return m_chan.group(1).strip()
        m_plan = re.search(r"(?:^|\n)(?:Plan|Thinking):\s*(.*?)(?=\n(?:Action|Step\s*\d|Execution):|$)", text, re.DOTALL | re.IGNORECASE)
        return m_plan.group(1).strip() if m_plan else ""

    def parse_actions(self, text: str) -> List[Dict[str, Any]]:
        """Extracts explicit action steps or code actions."""
        actions = []

        # Pattern: Action: command or code
        for match in re.finditer(r"(?:^|\n)Action:\s*(.+?)(?=\n(?:Plan|Observation|Reflection|Action|$))", text, re.DOTALL | re.IGNORECASE):
            act_text = match.group(1).strip()
            if act_text:
                actions.append({"type": "action_directive", "content": act_text})

        # Code actions: ```bash ... ``` or ```python ... ```
        for match in re.finditer(r"```([a-z0-9_-]+)\n(.*?)```", text, re.DOTALL):
            actions.append({"type": match.group(1), "content": match.group(2).strip()})

        return actions

    def execute_action(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates action content."""
        content = action.get("content", "").strip()
        if not content:
            return False, "Empty action block."
        return True, "Action recorded and executed (emulated environment)."

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 2,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the multi-turn Pi agent interaction cycle."""
        curr_prompt = prompt
        tools_called = 0
        tools_succeeded = 0
        tools_failed = 0
        history: List[Dict[str, Any]] = []
        final_text = ""
        reasoning_text = ""

        for turn in range(max(1, max_turns)):
            response = model_generate_fn(curr_prompt)
            final_text = response
            r = self.extract_reasoning(response)
            if r:
                reasoning_text = r

            actions = self.parse_actions(response)
            if not actions:
                break

            turn_feedbacks = []
            for act in actions:
                tools_called += 1
                ok, msg = self.execute_action(act)
                if ok:
                    tools_succeeded += 1
                else:
                    tools_failed += 1

                turn_feedbacks.append(f"Observation: {msg}")
                history.append({"turn": turn, "action": act, "success": ok, "output": msg})

            if turn >= max_turns - 1:
                break

            feedback_str = "\n".join(turn_feedbacks)
            curr_prompt = (
                f"{curr_prompt}\n\nModel:\n{response}\n\n"
                f"{feedback_str}\nProceed to complete task or state Final Answer:"
            )

        success_rate = (tools_succeeded / tools_called) if tools_called > 0 else 0.0

        return HarnessResult(
            task_id=task_id,
            prompt=prompt,
            final_text=final_text,
            reasoning_text=reasoning_text,
            tools_called=tools_called,
            tools_succeeded=tools_succeeded,
            tools_failed=tools_failed,
            tool_history=history,
            success_rate=success_rate,
            is_completed=True,
        )
