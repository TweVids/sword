"""
OpenCode CLI Harness: Tracks bash, python, and file tool invocations,
capturing tool execution results, success rates, and reasoning traces.
"""

import re
import ast
import json
import shlex
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
        raw = call.get("raw", "")

        # Reject dangerous shell instructions
        dangerous_patterns = ("rm -rf /", ":(){ :|:& };:", "dd if=/dev/zero", "mkfs.")
        if any(bad in raw for bad in dangerous_patterns):
            return False, "Tool execution rejected: dangerous command pattern detected."

        if not self.allow_exec:
            # Emulated / safe validation without spawning real host shell
            if ctype == "python":
                code = payload.get("code") or raw
                try:
                    ast.parse(code)
                    return True, "Executed successfully (valid Python syntax)"
                except SyntaxError as e:
                    return False, f"SyntaxError: {e.msg} at line {e.lineno}"
            elif ctype in ("bash", "sh"):
                cmd = payload.get("code") or payload.get("command") or raw
                try:
                    shlex.split(cmd)
                    if len(cmd.strip()) > 0:
                        return True, "Executed successfully (valid shell syntax)"
                    return False, "Empty shell command"
                except Exception as e:
                    return False, f"Shell parse error: {str(e)}"
            elif ctype == "tool_call":
                if isinstance(payload, dict) and ("command" in payload or "action" in payload or "name" in payload):
                    return True, "Executed successfully (valid tool payload)"
                return False, "Malformed tool_call payload"
            return len(raw.strip()) > 0, "Executed successfully (emulated)"

        # Live execution when allow_exec is True
        try:
            if ctype in ("bash", "sh"):
                cmd = payload.get("code") or payload.get("command") or raw
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_sec)
                ok = res.returncode == 0
                out = res.stdout if ok else res.stderr
                return ok, (out[:500] if out else "Success (exit code 0)")
            elif ctype == "python":
                code = payload.get("code") or raw
                res = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=self.timeout_sec)
                ok = res.returncode == 0
                out = res.stdout if ok else res.stderr
                return ok, (out[:500] if out else "Success (exit code 0)")
            else:
                return True, "Tool recognized"
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {self.timeout_sec}s"
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

        for turn in range(max(1, max_turns)):
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

            if turn < max_turns - 1:
                # Append tool outputs for next turn
                recent_calls = history[-len(calls):]
                feedbacks = []
                for h in recent_calls:
                    tag = "[Tool Result]" if h["success"] else "[Tool Error]"
                    feedbacks.append(f"{tag}: {h['output']}")
                tool_feedback = "\n".join(feedbacks)
                curr_prompt = f"{curr_prompt}\n\nModel:\n{response}\n\nEnvironment:\n{tool_feedback}\n\nProceed:"

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
