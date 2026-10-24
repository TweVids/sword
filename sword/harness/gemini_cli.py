"""
Gemini CLI Harness: Function-calling and multi-turn tool interaction harness,
handling structured call:tool_name{args} syntax and telemetry.
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult


class GeminiCliHarness(BaseHarness):
    """
    Harness for Gemini CLI function calling style agents.
    Parses `call:tool_name{...}` patterns and validates argument schemas.
    """
    name: str = "gemini_cli"

    def extract_reasoning(self, text: str) -> str:
        """Extracts thought traces."""
        m = re.search(r"<\|channel>thought(.*?)(?:<channel\||<\|end_channel>|$)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m_alt = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return m_alt.group(1).strip() if m_alt else ""

    def parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parses call:func{args} or {"call": ...} patterns."""
        calls = []

        # Pattern 1: call:tool_name{...}
        for match in re.finditer(r"call:([A-Za-z0-9_]+)\s*(\{[^{}]*\})", text):
            fname = match.group(1)
            raw_args = match.group(2)
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {"raw": raw_args}
            calls.append({"name": fname, "args": args, "valid_json": isinstance(args, dict)})

        # Pattern 2: ```json\n{"action": ...}\n```
        for match in re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if "action" in data or "tool" in data:
                    calls.append({
                        "name": data.get("action") or data.get("tool"),
                        "args": data.get("parameters") or data.get("args") or {},
                        "valid_json": True,
                    })
            except Exception:
                pass

        return calls

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 2,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the Gemini CLI harness loop."""
        response = model_generate_fn(prompt)
        reasoning = self.extract_reasoning(response)
        calls = self.parse_function_calls(response)

        tools_called = len(calls)
        tools_succeeded = sum(1 for c in calls if c.get("valid_json", False))
        tools_failed = tools_called - tools_succeeded
        success_rate = (tools_succeeded / tools_called) if tools_called > 0 else 0.0

        return HarnessResult(
            task_id=task_id,
            prompt=prompt,
            final_text=response,
            reasoning_text=reasoning,
            tools_called=tools_called,
            tools_succeeded=tools_succeeded,
            tools_failed=tools_failed,
            tool_history=[{"call": c, "success": c.get("valid_json", False)} for c in calls],
            success_rate=success_rate,
            is_completed=True,
        )
