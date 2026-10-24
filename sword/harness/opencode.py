"""
OpenCode CLI Harness: Tracks bash, python, and file tool invocations,
capturing tool execution results, success rates, and reasoning traces.
"""

import re
import json
import subprocess
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult


class OpenCodeHarness(BaseHarness):
    """
    OpenCode agentic loop adapter.
    Parses tool commands, executes safe sandboxed sub-processes or emulations,
    and returns tool execution telemetry.
    """
    name: str = "opencode"

    def __init__(self, timeout_sec: float = 10.0, allow_exec: bool = False):
        self.timeout_sec = timeout_sec
        self.allow_exec = allow_exec

    def extract_reasoning(self, text: str) -> str:
        """Extracts thought / reasoning channel contents."""
        m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m_think:
            return m_think.group(1).strip()
        m_channel = re.search(r"<\|channel>thought(.*?)(?:<channel\||<\|end_channel>|$)", text, re.DOTALL)
        if m_channel:
            return m_channel.group(1).strip()
        return ""

    def parse_tools(self, text: str) -> List[Dict[str, Any]]:
        """Finds structured tool calls or code execution blocks in model output."""
        calls = []

        # 1. JSON tool_call blocks
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
            raw = match.group(1).strip()
            try:
                data = json.loads(raw)
                calls.append({"type": "tool_call", "payload": data, "raw": raw})
            except Exception:
                calls.append({"type": "tool_call", "payload": {"command": raw}, "raw": raw})

        # 2. Bash / Python code blocks intended for execution
        for match in re.finditer(r"```(bash|sh|python)\n(.*?)```", text, re.DOTALL):
            lang = match.group(1)
            code = match.group(2).strip()
            calls.append({"type": lang, "payload": {"code": code}, "raw": code})

        return calls

    def execute_tool(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        """Safely runs or validates tool invocation."""
        ctype = call.get("type", "")
        payload = call.get("payload", {})

        if not self.allow_exec:
            # Emulated / safe validation without spawning real host shell
            raw = call.get("raw", "")
            if len(raw) > 0 and not any(bad in raw for bad in ("rm -rf /", ":(){ :|:& };:")):
                return True, "Executed successfully (emulated sandbox)"
            return False, "Tool execution rejected by policy"

        try:
            if ctype in ("bash", "sh"):
                cmd = payload.get("code") or payload.get("command", "")
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_sec)
                ok = res.returncode == 0
                out = res.stdout if ok else res.stderr
                return ok, out[:500]
            elif ctype == "python":
                code = payload.get("code", "")
                res = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=self.timeout_sec)
                ok = res.returncode == 0
                out = res.stdout if ok else res.stderr
                return ok, out[:500]
            else:
                return True, "Tool recognized"
        except Exception as e:
            return False, str(e)[:200]

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 3,
        task_id: str = "",
    ) -> HarnessResult:
        """Executes multi-turn tool interaction loop."""
        curr_prompt = prompt
        tools_called = 0
        tools_succeeded = 0
        tools_failed = 0
        history: List[Dict[str, Any]] = []
        final_text = ""
        reasoning_text = ""

        for turn in range(max_turns):
            response = model_generate_fn(curr_prompt)
            final_text = response
            r_chunk = self.extract_reasoning(response)
            if r_chunk:
                reasoning_text = r_chunk

            calls = self.parse_tools(response)
            if not calls:
                # No more tools called, task finished
                break

            for c in calls:
                tools_called += 1
                ok, output = self.execute_tool(c)
                if ok:
                    tools_succeeded += 1
                else:
                    tools_failed += 1

                history.append({
                    "turn": turn,
                    "call": c,
                    "success": ok,
                    "output": output,
                })

            # Append tool outputs for next turn
            tool_feedback = "\n".join([f"[Tool Result]: {h['output']}" for h in history[-len(calls):]])
            curr_prompt = f"{curr_prompt}\n\nModel:\n{response}\n\nEnvironment:\n{tool_feedback}\n\nProceed:"

        success_rate = (tools_succeeded / tools_called) if tools_called > 0 else 1.0

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
