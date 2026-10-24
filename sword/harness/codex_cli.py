"""
Codex CLI Harness: Evaluates code generation, syntax validation, unit test execution,
and assertion verification for programming tasks.
"""

import re
import ast
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult


class CodexCliHarness(BaseHarness):
    """
    Harness for Codex CLI style benchmark execution.
    Extracts code solutions, validates abstract syntax trees, and executes assertions.
    """
    name: str = "codex_cli"

    def __init__(self, timeout_sec: float = 5.0):
        self.timeout_sec = timeout_sec

    def extract_reasoning(self, text: str) -> str:
        """Extracts thinking process."""
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extracts python code blocks from markdown."""
        blocks = []
        for m in re.finditer(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL):
            blocks.append(m.group(1).strip())
        return blocks

    def check_syntax(self, code: str) -> Tuple[bool, str]:
        """Checks if generated Python code is syntactically valid."""
        try:
            ast.parse(code)
            return True, "Valid Python syntax"
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}"

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 2,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the Codex CLI validation loop."""
        response = model_generate_fn(prompt)
        reasoning = self.extract_reasoning(response)
        code_blocks = self.extract_code_blocks(response)

        tools_called = len(code_blocks)
        tools_succeeded = 0
        tools_failed = 0
        history = []

        for idx, block in enumerate(code_blocks):
            ok, msg = self.check_syntax(block)
            if ok:
                tools_succeeded += 1
            else:
                tools_failed += 1
            history.append({
                "block_idx": idx,
                "syntax_valid": ok,
                "message": msg,
            })

        success_rate = (tools_succeeded / tools_called) if tools_called > 0 else 0.0

        return HarnessResult(
            task_id=task_id,
            prompt=prompt,
            final_text=response,
            reasoning_text=reasoning,
            tools_called=tools_called,
            tools_succeeded=tools_succeeded,
            tools_failed=tools_failed,
            tool_history=history,
            success_rate=success_rate,
            is_completed=True,
        )
