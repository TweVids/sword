"""
Gemini CLI Harness: Function-calling and multi-turn tool interaction harness.
Handles call:tool_name{args} and structured JSON tool invocations with real execution
in an isolated WorkspaceSandbox environment.
"""

import re
import ast
import json
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult
from sword.harness.sandbox import WorkspaceSandbox


class GeminiCliHarness(BaseHarness):
    """
    Harness for Gemini CLI function calling style agents.
    Parses function calls, verifies parameter schemas, executes tools in a real sandbox,
    and returns genuine observation telemetry.
    """
    name: str = "gemini_cli"

    def __init__(self, timeout_sec: float = 10.0, workspace_dir: Optional[str] = None):
        self.timeout_sec = timeout_sec
        self.workspace_dir = workspace_dir

    def extract_reasoning(self, text: str) -> str:
        """Extracts thought traces."""
        m = re.search(r"<\|channel>thought(.*?)(?:<channel\||<\|end_channel>|$)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m_alt = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return m_alt.group(1).strip() if m_alt else ""

    def parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parses call:func{args} or ```json {"action": ...}``` patterns."""
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

    def execute_call_in_sandbox(self, call: Dict[str, Any], sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
        """Validates function schema and executes real tool in sandbox."""
        if not call.get("valid_json", False):
            return False, "Malformed JSON arguments in function call."

        fname = str(call.get("name", "")).strip().lower()
        args = call.get("args", {})
        if not isinstance(args, dict):
            return False, "Function call arguments must be a JSON dictionary."

        if fname in ("calculator", "math_eval", "eval_math"):
            expr = str(args.get("expression") or args.get("expr") or "").strip()
            if not expr:
                return False, "Missing required 'expression' parameter."
            if not re.match(r"^[\d\s\+\-\*\/\(\)\.\%]+$", expr):
                return False, f"Invalid characters in arithmetic expression: {expr}"
            try:
                # Safe AST-based evaluation of arithmetic
                node = ast.parse(expr, mode="eval")
                for sub in ast.walk(node):
                    if not isinstance(sub, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num, ast.operator, ast.unaryop)):
                        return False, "Forbidden AST node in arithmetic expression."
                result = eval(expr, {"__builtins__": {}}, {})
                return True, str(result)
            except Exception as e:
                return False, f"Evaluation error: {e}"

        elif fname in ("bash", "run_bash", "shell", "run_command"):
            cmd = str(args.get("command") or args.get("cmd") or "").strip()
            if not cmd:
                return False, "Missing required 'command' parameter."
            return sandbox.run_bash(cmd)

        elif fname in ("python", "run_python", "python_eval"):
            code = str(args.get("code") or args.get("script") or "").strip()
            if not code:
                return False, "Missing required 'code' parameter."
            return sandbox.run_python(code)

        elif fname in ("read_file", "read"):
            path = str(args.get("path") or args.get("file") or "").strip()
            if not path:
                return False, "Missing required 'path' parameter."
            return sandbox.read_file(path)

        elif fname in ("write_file", "write"):
            path = str(args.get("path") or args.get("file") or "").strip()
            content = str(args.get("content") or "")
            if not path:
                return False, "Missing required 'path' parameter."
            return sandbox.write_file(path, content)

        return False, f"Unknown Gemini CLI tool: {fname}"

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 2,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the authentic Gemini CLI multi-turn function calling loop."""
        sandbox = WorkspaceSandbox(base_dir=self.workspace_dir, timeout_sec=self.timeout_sec)
        curr_prompt = prompt
        tools_called = 0
        tools_succeeded = 0
        tools_failed = 0
        history: List[Dict[str, Any]] = []
        final_text = ""
        reasoning_text = ""

        try:
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
                    ok, output = self.execute_call_in_sandbox(c, sandbox)
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
                    f"Gemini Environment Feedback:\n{feedback_str}\n\n"
                    "Proceed to complete the response:"
                )

        finally:
            sandbox.close()

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
