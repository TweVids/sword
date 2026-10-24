"""
Codex CLI Harness: Evaluates code generation, syntax validation, unit test execution,
and assertion verification for programming tasks with multi-turn error recovery.
"""

import re
import ast
import subprocess
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult


class CodexCliHarness(BaseHarness):
    """
    Harness for Codex CLI style benchmark execution.
    Extracts code solutions, validates abstract syntax trees, executes assertions,
    and coordinates multi-turn self-repair when code contains errors.
    """
    name: str = "codex_cli"

    def __init__(self, timeout_sec: float = 5.0, allow_exec: bool = False):
        self.timeout_sec = timeout_sec
        self.allow_exec = allow_exec

    def extract_reasoning(self, text: str) -> str:
        """Extracts thinking process."""
        m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m_think:
            return m_think.group(1).strip()
        m_chan = re.search(r"<\|channel>thought(.*?)(?:<channel\||<\|end_channel>|$)", text, re.DOTALL)
        if m_chan:
            return m_chan.group(1).strip()
        return ""

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extracts python code blocks from markdown."""
        blocks = []
        for m in re.finditer(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL):
            code = m.group(1).strip()
            if code:
                blocks.append(code)
        return blocks

    def check_syntax(self, code: str) -> Tuple[bool, str]:
        """Checks if generated Python code is syntactically valid."""
        try:
            ast.parse(code)
            return True, "Valid Python syntax"
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}"

    def execute_and_test(self, code: str) -> Tuple[bool, str]:
        """Validates syntax and executes code/assertions if execution is permitted."""
        valid_syntax, msg = self.check_syntax(code)
        if not valid_syntax:
            return False, msg

        if not self.allow_exec:
            return True, "Valid Python syntax (emulated check)"

        try:
            res = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            ok = res.returncode == 0
            out = res.stdout if ok else res.stderr
            return ok, (out[:500] if out else "Execution succeeded without error")
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {self.timeout_sec}s"
        except Exception as e:
            return False, str(e)[:200]

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 2,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the multi-turn Codex CLI validation loop."""
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

            code_blocks = self.extract_code_blocks(response)
            if not code_blocks:
                break

            turn_succeeded = 0
            turn_failed = 0
            feedback_items = []

            for idx, block in enumerate(code_blocks):
                tools_called += 1
                ok, msg = self.execute_and_test(block)
                if ok:
                    tools_succeeded += 1
                    turn_succeeded += 1
                else:
                    tools_failed += 1
                    turn_failed += 1

                feedback_items.append(f"Block {idx + 1}: {msg}")
                history.append({
                    "turn": turn,
                    "block_idx": idx,
                    "syntax_valid": ok,
                    "message": msg,
                })

            # If all blocks are valid or we reached the last turn, stop
            if turn_failed == 0 or turn >= max_turns - 1:
                break

            # Multi-turn self-repair: feed back errors to prompt
            feedback = "\n".join(feedback_items)
            curr_prompt = (
                f"{curr_prompt}\n\nModel:\n{response}\n\n"
                f"Codex CLI Verification:\n{feedback}\n"
                "Please fix the code errors and output the corrected version."
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
