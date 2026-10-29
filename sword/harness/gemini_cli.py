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
from sword.harness.protocol import structured_calls


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
        m = re.search(r"<\|channel>\s*thought(.*?)(?:<channel\||<\|end_channel>|$)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m_alt = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return m_alt.group(1).strip() if m_alt else ""

    def parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parses call:func{args} or ```json {"action": ...}``` patterns."""
        return structured_calls(text)

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

        elif fname in ("run_shell_command", "bash", "run_bash", "shell", "run_command"):
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
            path = str(args.get("file_path") or args.get("path") or args.get("file") or "").strip()
            if not path:
                return False, "Missing required 'file_path' parameter."
            start_line = args.get("start_line")
            end_line = args.get("end_line")
            return sandbox.read_file(path, start_line=start_line, end_line=end_line)

        elif fname in ("write_file", "write"):
            path = str(args.get("file_path") or args.get("path") or args.get("file") or "").strip()
            content = str(args.get("content") or "")
            if not path:
                return False, "Missing required 'file_path' parameter."
            return sandbox.write_file(path, content)

        elif fname in ("replace", "edit", "edit_file"):
            path = str(args.get("file_path") or args.get("path") or "").strip()
            old_str = str(args.get("old_string") or args.get("old_str") or "")
            new_str = str(args.get("new_string") or args.get("new_str") or "")
            allow_multiple = bool(args.get("allow_multiple", False))
            if not path:
                return False, "Missing required 'file_path' parameter."
            return sandbox.edit_file(path, old_str, new_str, replace_all=allow_multiple)

        elif fname in ("list_directory", "ls", "list_dir"):
            dpath = str(args.get("dir_path") or args.get("path") or ".").strip()
            ok, entries = sandbox.list_dir(dpath)
            return ok, "\n".join(entries)

        elif fname == "glob":
            pattern = str(args.get("pattern") or "").strip()
            if not pattern:
                return False, "Missing required 'pattern' parameter."
            dpath = str(args.get("dir_path") or "").strip()
            ok, files = sandbox.glob_files(pattern, dir_path=dpath)
            return ok, "\n".join(files) if files else "No matching files"

        elif fname == "grep_search":
            pattern = str(args.get("pattern") or "").strip()
            if not pattern:
                return False, "Missing required 'pattern' parameter."
            dpath = str(args.get("dir_path") or "").strip()
            inc = args.get("include_pattern")
            max_m = int(args.get("total_max_matches") or 100)
            return sandbox.grep_search(pattern, dir_path=dpath, include_pattern=inc, max_matches=max_m)

        return False, f"Unknown Gemini CLI tool: {fname}"

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 8,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the authentic Gemini CLI multi-turn function calling loop."""
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        sandbox = self.create_sandbox()
        curr_prompt = prompt
        tools_called = 0
        tools_succeeded = 0
        tools_failed = 0
        history: List[Dict[str, Any]] = []
        final_text = ""
        reasoning_text = ""
        assistant_turns = []
        is_completed = False
        stop_reason = "turn_limit"

        try:
            for turn in range(max(1, max_turns)):
                response = model_generate_fn(curr_prompt)
                final_text = response
                assistant_turns.append({"turn": turn, "response": response})
                r = self.extract_reasoning(response)
                if r:
                    reasoning_text = r

                calls = self.parse_function_calls(response)
                if not calls:
                    if self.has_final_answer(response):
                        is_completed, stop_reason = True, "final_response"
                        break
                    if turn >= max_turns - 1:
                        stop_reason = "missing_final_answer" if response.strip() else "empty_response"
                        break
                    curr_prompt = self.unfinished_feedback(curr_prompt, response)
                    continue

                turn_feedbacks = []
                for c in calls:
                    tools_called += 1
                    ok, output = self.execute_call_in_sandbox(c, sandbox)
                    self.log_tool_result(turn, c.get("name"), ok, output)
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
                    "Inspect the tool results, edit or verify as needed, then provide Final Answer: when done. "
                    "Successful tool execution alone does not complete the task."
                )

        finally:
            try:
                test_results = self.run_dataset_tests(sandbox, final_text)
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
            test_results=test_results, execution_results=list(sandbox.execution_history),
            code_execution=getattr(sandbox, "code_execution", {}),
            success_rate=success_rate,
            is_completed=is_completed,
            error_message=None if is_completed else "No final answer before " + stop_reason,
            turns=len(assistant_turns), stop_reason=stop_reason, assistant_turns=assistant_turns,
            workspace_dir=sandbox.root_dir, workspace_removed=sandbox.was_removed,
        )
