"""
Gemini CLI Harness: Function-calling and multi-turn tool interaction harness,
handling structured call:tool_name{args} syntax and telemetry with environment feedback.
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult


class GeminiCliHarness(BaseHarness):
    """
    Harness for Gemini CLI function calling style agents.
    Parses `call:tool_name{...}` patterns, validates argument schemas,
    dispatches tools, and feeds results back into a multi-turn interaction loop.
    """
    name: str = "gemini_cli"

    def __init__(self, allow_exec: bool = False):
        self.allow_exec = allow_exec

    def extract_reasoning(self, text: str) -> str:
        """Extracts thought traces."""
        m = re.search(r"<\|channel>thought(.*?)(?:<channel\||<\|end_channel>|$)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m_alt = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return m_alt.group(1).strip() if m_alt else ""

    def parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parses call:func{args} or {"action": ...} patterns."""
        calls = []

        # Pattern 1: call:tool_name{...}
        for match in re.finditer(r"call:([A-Za-z0-9_]+)\s*(\{[^{}]*\})", text):
            fname = match.group(1)
            raw_args = match.group(2)
            try:
                args = json.loads(raw_args)
                valid_json = isinstance(args, dict)
            except Exception:
                args = {"raw": raw_args}
                valid_json = False
            calls.append({"name": fname, "args": args, "valid_json": valid_json})

        # Pattern 2: ```json\n{"action": ...}\n```
        for match in re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and ("action" in data or "tool" in data):
                    calls.append({
                        "name": data.get("action") or data.get("tool"),
                        "args": data.get("parameters") or data.get("args") or {},
                        "valid_json": True,
                    })
            except Exception:
                pass

        return calls

    def execute_call(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates and executes function call."""
        if not call.get("valid_json", False):
            return False, "Malformed JSON arguments in function call."

        fname = str(call.get("name", "")).strip().lower()
        args = call.get("args", {})

        # Standard built-in tool validations
        if fname in ("calculator", "math_eval"):
            expr = str(args.get("expression") or args.get("expr") or "").strip()
            if not expr:
                return False, "Missing expression argument."
            try:
                # Safe evaluation of basic arithmetic
                if re.match(r"^[\d\s\+\-\*\/\(\)\.\%]+$", expr):
                    result = eval(expr, {"__builtins__": None}, {})
                    return True, str(result)
                return True, "Expression parsed (emulated)"
            except Exception as e:
                return False, f"Evaluation error: {e}"

        elif fname in ("bash", "shell", "run_command"):
            cmd = str(args.get("command") or args.get("cmd") or "").strip()
            if not cmd:
                return False, "Missing command argument."
            return True, f"Command '{cmd[:40]}' accepted."

        elif fname in ("read_file", "search", "web_search"):
            return True, f"Function {fname} executed successfully."

        return True, f"Tool {fname} executed."

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 2,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the multi-turn Gemini CLI harness loop."""
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

            calls = self.parse_function_calls(response)
            if not calls:
                break

            turn_feedbacks = []
            for c in calls:
                tools_called += 1
                ok, output = self.execute_call(c)
                if ok:
                    tools_succeeded += 1
                else:
                    tools_failed += 1

                tag = "[Tool Output]" if ok else "[Tool Error]"
                turn_feedbacks.append(f"{tag} {c.get('name')}: {output}")
                history.append({"turn": turn, "call": c, "success": ok, "output": output})

            if turn >= max_turns - 1:
                break

            feedback_str = "\n".join(turn_feedbacks)
            curr_prompt = (
                f"{curr_prompt}\n\nModel:\n{response}\n\n"
                f"Environment Feedback:\n{feedback_str}\nProceed to complete the response:"
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
