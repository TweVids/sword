"""
Base schemas and abstract contracts for agentic harnesses.
Tracks tool calls, tool success rates, reasoning traces, and execution results.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, Any, List, Optional, Callable
from sword.harness.protocol import visible_response
from sword.logging_utils import output_preview


@dataclass
class HarnessResult:
    """Detailed telemetry captured from agentic harness execution."""
    task_id: str = ""
    prompt: str = ""
    final_text: str = ""
    reasoning_text: str = ""
    tools_called: int = 0
    tools_succeeded: int = 0
    tools_failed: int = 0
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    success_rate: float = 0.0
    is_completed: bool = True
    error_message: Optional[str] = None
    turns: int = 0
    stop_reason: str = ""
    workspace_dir: str = ""
    workspace_removed: bool = False
    assistant_turns: List[Dict[str, Any]] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    execution_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tools_called": self.tools_called,
            "tools_succeeded": self.tools_succeeded,
            "tools_failed": self.tools_failed,
            "success_rate": round(self.success_rate, 4),
            "is_completed": self.is_completed,
            "error_message": self.error_message,
            "turns": self.turns,
            "stop_reason": self.stop_reason,
            "workspace_dir": self.workspace_dir,
            "workspace_removed": self.workspace_removed,
            "test_results": self.test_results,
        }


class BaseHarness:
    """Abstract base class for all CLI tool-calling harnesses."""
    name: str = "base"

    def create_sandbox(self):
        from sword.harness.sandbox import WorkspaceSandbox
        sandbox = WorkspaceSandbox(
            base_dir=getattr(self, "workspace_dir", None), timeout_sec=self.timeout_sec,
            cleanup=not getattr(self, "keep_workspace", False),
            parent_dir=getattr(self, "workspace_parent_dir", None),
        )
        try:
            for path, content in getattr(self, "initial_files", {}).items():
                ok, message = sandbox.write_file(path, content)
                if not ok:
                    raise ValueError(message)
        except Exception:
            sandbox.close()
            raise
        return sandbox

    def run_dataset_tests(self, sandbox, response):
        import sys
        if sys.exc_info()[0] is not None:
            return []
        from sword.execution import run_dataset_tests
        return run_dataset_tests(sandbox, getattr(self, "dataset_tests", []), response,
                                 getattr(self, "language", None) or "python")

    @staticmethod
    def has_final_answer(text: str) -> bool:
        visible = visible_response(text)
        match = re.search(r"(?im)^\s*(?:#{1,6}\s*)?Final (?:Answer|Solution)\s*:", visible)
        return match is not None and bool(visible[match.end():].strip())

    def unfinished_feedback(self, prompt, response):
        return (f"{prompt}\n\nModel:\n{response}\n\nAgent Feedback:\n"
                "No complete tool call or explicit final answer was received. Continue the task: "
                "emit a complete tool call with valid arguments, inspect its returned result, "
                "edit and verify as needed, then provide Final Answer: followed by the result. "
                "Thinking, a plan, and a partial tool call do not complete the task.")

    def log_tool_result(self, turn, name, success, output):
        if getattr(self, "log_tool_outputs", False):
            print(f"[Agent Tool] harness={self.name} turn={turn + 1} tool={name} success={success}\n"
                  f"{output_preview(output, getattr(self, 'model_output_chars', 1200))}", flush=True)

    def get_system_prompt(self) -> str:
        """Returns the authentic harness system prompt loaded from the cloned repository."""
        from sword.harness.prompts import get_harness_system_prompt
        return get_harness_system_prompt(self.name)

    def format_prompt(
        self,
        problem: str,
        max_prompt_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """
        Combines harness system prompt and tool guidelines with user problem.
        Uses real tokenizer token counts when available and trims along paragraph/section
        boundaries so tool definitions and the trailing Task block stay intact.
        """
        sys_prompt = self.get_system_prompt()
        if sys_prompt:
            sys_prompt = ("Agent loop protocol: use tools to work on the task, inspect their returned results, "
                          "repair failures and verify the result. Continue until complete. "
                          "Finish with Final Answer: followed by your result. "
                          "For tools you may emit JSON <tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call> "
                          "or Gemma <|tool_call>call:tool_name{argument:<|\"|>value<|\"|>}<tool_call|>.\n\n"
                          + sys_prompt)
        if not sys_prompt:
            return problem

        task_block = f"\n\nTask:\n{problem}"
        full_candidate = f"{sys_prompt}{task_block}"

        # Split system prompt into guidance body and trailing tool declarations section
        tool_markers = (
            "\n\n# OpenCode Tool Definitions:",
            "\n\n# Codex CLI Tool Specifications:",
            "\n\n<tool_schemas>",
            "\n\n# Gemini CLI Tool Declarations:",
        )
        body = sys_prompt
        tool_section = ""
        for marker in tool_markers:
            if marker in sys_prompt:
                idx = sys_prompt.rfind(marker)
                body = sys_prompt[:idx]
                tool_section = sys_prompt[idx:]
                break

        def _count_tokens(text: str) -> Optional[int]:
            if tokenizer is None:
                return None
            try:
                if hasattr(tokenizer, "encode"):
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if isinstance(ids, list):
                        return len(ids)
                out = tokenizer(text, add_special_tokens=False)
                ids = out.get("input_ids", []) if isinstance(out, dict) else getattr(out, "input_ids", [])
                if hasattr(ids, "numel"):
                    n = int(ids.numel())
                    # Ignore fixed-length dummy tokenizers that return constant length regardless of input
                    if len(text) > 200 and n <= 8:
                        return None
                    return n
                if isinstance(ids, list):
                    flat = ids[0] if (ids and isinstance(ids[0], list)) else ids
                    if len(text) > 200 and len(flat) <= 8:
                        return None
                    return len(flat)
            except Exception:
                pass
            return None

        tok_count = _count_tokens(full_candidate)
        exceeds_tok = (max_tokens is not None and tok_count is not None and tok_count > max_tokens)

        char_budget = max_prompt_chars
        if char_budget is None and max_tokens is not None and tok_count is None:
            char_budget = max(256, int(max_tokens) * 4)

        exceeds_char = (char_budget is not None and len(full_candidate) > char_budget)

        if not exceeds_tok and not exceeds_char:
            return full_candidate

        target_chars = char_budget if char_budget is not None else max(256, int(max_tokens or 256) * 4)
        marker = "\n...[trimmed]...\n"

        def trim_to_chars(limit: int) -> str:
            avail_for_sys = max(0, limit - len(task_block) - len(marker))
            if not avail_for_sys:
                return task_block
            # Preserve complete tool lines. An oversized first line must not
            # defeat the budget merely because nothing has been collected yet.
            if tool_section:
                ts_lines = tool_section.lstrip("\n").splitlines()
                header = ts_lines[0] if ts_lines else ""
                tail_budget = int(avail_for_sys * 0.75)
                kept_tail = header if len(header) <= tail_budget else ""
                collected = []
                used = len(kept_tail)
                for line in reversed(ts_lines[1:]):
                    if used + len(line) + 1 > tail_budget:
                        break
                    collected.append(line)
                    used += len(line) + 1
                if collected:
                    kept_tail += "\n" + "\n".join(reversed(collected))
                head_budget = max(0, avail_for_sys - len(kept_tail))
                kept_head = body[:head_budget]
                if "\n" in kept_head:
                    kept_head = kept_head.rsplit("\n", 1)[0]
            else:
                head_budget = avail_for_sys // 2
                tail_budget = avail_for_sys - head_budget
                kept_head = sys_prompt[:head_budget]
                kept_tail = sys_prompt[-tail_budget:] if tail_budget else ""
            return f"{kept_head}{marker}{kept_tail}{task_block}"

        candidate = trim_to_chars(target_chars)
        count = _count_tokens(candidate)
        while max_tokens is not None and count is not None and count > max_tokens:
            # Characters-per-token is only a starting estimate. Recount using
            # the real tokenizer, retaining the complete trailing task.
            target_chars = min(target_chars - 1, int(target_chars * max_tokens / count * 0.9))
            candidate = trim_to_chars(target_chars)
            count = _count_tokens(candidate)
            if candidate == task_block and count is not None and count > max_tokens:
                raise ValueError("The task alone exceeds the prompt token budget; increase max_seq_len")
        if char_budget is not None and len(candidate) > char_budget:
            raise ValueError("The task alone exceeds max_prompt_chars")
        return candidate

    def format_followup_prompt(self, problem: str, history: str, max_tokens: int, tokenizer=None) -> str:
        """Retain the task and recent tool observations when conversation grows."""
        turns = [part for part in history.split("\n\nModel:\n") if part]
        while True:
            conversation = "\n\nModel:\n".join(turns)
            task = problem + "\n\nAgent conversation and tool observations:\n" + conversation
            try:
                return self.format_prompt(task, max_tokens=max_tokens, tokenizer=tokenizer)
            except ValueError as error:
                if "task alone exceeds" not in str(error).lower():
                    raise
                if len(turns) > 1:
                    turns.pop(0)
                elif turns and len(turns[0]) > 64:
                    # Very long tool output must fit too. Keep both the action
                    # prefix and latest observation tail, marking omitted content.
                    keep = max(1, len(turns[0]) // 4)
                    turns[0] = turns[0][:keep] + "\n[context trimmed]\n" + turns[0][-keep:]
                else:
                    raise ValueError("max_seq_len is too small for the task and latest tool feedback") from error

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 8,
        task_id: str = "",
    ) -> HarnessResult:
        """Executes the agentic interaction loop and records tool telemetry."""
        raise NotImplementedError
