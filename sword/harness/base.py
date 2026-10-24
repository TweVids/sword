"""
Base schemas and abstract contracts for agentic harnesses.
Tracks tool calls, tool success rates, reasoning traces, and execution results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tools_called": self.tools_called,
            "tools_succeeded": self.tools_succeeded,
            "tools_failed": self.tools_failed,
            "success_rate": round(self.success_rate, 4),
            "is_completed": self.is_completed,
            "error_message": self.error_message,
        }


class BaseHarness:
    """Abstract base class for all CLI tool-calling harnesses."""
    name: str = "base"

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
        avail_for_sys = max(64, target_chars - len(task_block) - len("\n...[trimmed]...\n"))

        # Preserve complete tool lines from the end and preamble lines from the start
        if tool_section:
            if len(tool_section) <= int(avail_for_sys * 0.75):
                kept_tail = tool_section.lstrip("\n")
            else:
                # Keep the tool section header line plus as many complete trailing tool lines as fit
                ts_lines = tool_section.lstrip("\n").splitlines()
                header_line = ts_lines[0] if ts_lines else ""
                tail_budget = max(32, int(avail_for_sys * 0.7) - len(header_line) - 1)
                collected: List[str] = []
                used = 0
                for line in reversed(ts_lines[1:]):
                    if used + len(line) + 1 > tail_budget and collected:
                        break
                    collected.append(line)
                    used += len(line) + 1
                kept_tail = header_line + ("\n" + "\n".join(reversed(collected)) if collected else "")
            head_budget = max(32, avail_for_sys - len(kept_tail))
            kept_head = body[:head_budget].rsplit("\n", 1)[0] if "\n" in body[:head_budget] else body[:head_budget]
        else:
            head_budget = avail_for_sys // 2
            tail_budget = avail_for_sys - head_budget
            kept_head = sys_prompt[:head_budget]
            kept_tail = sys_prompt[-tail_budget:]

        trimmed_sys = f"{kept_head}\n...[trimmed]...\n{kept_tail}"
        return f"{trimmed_sys}{task_block}"

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 3,
        task_id: str = "",
    ) -> HarnessResult:
        """Executes the agentic interaction loop and records tool telemetry."""
        raise NotImplementedError
