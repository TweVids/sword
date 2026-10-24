"""
Codex CLI Harness: Terminal-first sandboxed code execution and test verification harness.
Executes code blocks and assertions in a real sandbox environment, capturing runtime tracebacks,
exit codes, and driving multi-turn self-repair.
"""

import re
import ast
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult
from sword.harness.sandbox import WorkspaceSandbox


class CodexCliHarness(BaseHarness):
    """
    Codex CLI style benchmark and code execution harness.
    Validates Python AST syntax, executes code in isolated WorkspaceSandbox,
    and runs multi-turn error correction loops with real runtime feedback.
    """
    name: str = "codex_cli"

    def __init__(self, timeout_sec: float = 10.0, workspace_dir: Optional[str] = None):
        self.timeout_sec = timeout_sec
        self.workspace_dir = workspace_dir

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
        """Extracts executable python code blocks from markdown."""
        blocks = []
        for m in re.finditer(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL):
            code = m.group(1).strip()
            if code:
                blocks.append(code)
        return blocks

    def validate_and_execute(self, code: str, sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
        """Performs real AST syntax check followed by real subprocess execution in sandbox."""
        # Step 1: Real AST parse
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}"

        # Step 2: Real subprocess execution in sandbox
        ok, output = sandbox.run_python(code)
        if not ok:
            return False, f"RuntimeError: {output}"
        return True, output or "Execution succeeded with exit code 0"

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 2,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the authentic Codex CLI multi-turn code verification and repair loop."""
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

                code_blocks = self.extract_code_blocks(response)
                if not code_blocks:
                    break

                turn_succeeded = 0
                turn_failed = 0
                feedback_items = []

                for idx, block in enumerate(code_blocks):
                    tools_called += 1
                    ok, msg = self.validate_and_execute(block, sandbox)
                    if ok:
                        tools_succeeded += 1
                        turn_succeeded += 1
                        feedback_items.append(f"Block {idx + 1} Passed: {msg}")
                    else:
                        tools_failed += 1
                        turn_failed += 1
                        feedback_items.append(f"Block {idx + 1} Failed: {msg}")

                    history.append({
                        "turn": turn,
                        "block_idx": idx,
                        "success": ok,
                        "message": msg,
                    })

                if turn_failed == 0 or turn >= max_turns - 1:
                    break

                feedback_report = "\n".join(feedback_items)
                curr_prompt = (
                    f"{curr_prompt}\n\nModel:\n{response}\n\n"
                    f"Codex CLI Execution Feedback:\n{feedback_report}\n\n"
                    "Please fix the code errors and output the corrected version."
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
