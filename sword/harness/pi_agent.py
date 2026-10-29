"""
Pi Agent Harness: Mario Zechner's minimalist terminal coding agent harness (pi.dev).
Ships with foundational tools (read, write, edit, bash) executed in an isolated WorkspaceSandbox,
tracking iterative plan, action, and observation cycles.
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult
from sword.harness.sandbox import WorkspaceSandbox
from sword.harness.protocol import structured_calls, visible_response, without_structured_calls


class PiHarness(BaseHarness):
    """
    Harness for Pi autonomous coding agent (pi.dev).
    Dispatches foundational tools (read, write, edit, bash) to WorkspaceSandbox.
    Captures true execution status, observation outputs, and plan progress.
    """
    name: str = "pi"

    def __init__(self, timeout_sec: float = 10.0, workspace_dir: Optional[str] = None):
        self.timeout_sec = timeout_sec
        self.workspace_dir = workspace_dir

    def extract_reasoning(self, text: str) -> str:
        """Extracts thought / plan sections."""
        m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m_think:
            return m_think.group(1).strip()
        m_chan = re.search(r"<\|channel>\s*thought(.*?)(?:<channel\||<\|end_channel>|$)", text, re.DOTALL)
        if m_chan:
            return m_chan.group(1).strip()
        m_plan = re.search(r"(?:^|\n)(?:Plan|Thinking):\s*(.*?)(?=\n(?:Action|Step\s*\d|Execution):|$)", text, re.DOTALL | re.IGNORECASE)
        return m_plan.group(1).strip() if m_plan else ""

    def parse_actions(self, text: str) -> List[Dict[str, Any]]:
        """Extracts Pi foundational actions (read, write, edit, bash) or code blocks."""
        actions = []
        text = visible_response(text)
        structured = structured_calls(text)
        text = without_structured_calls(text)

        # 1. Structured action directives: Action: read path / Action: bash cmd / Action: write path \n content
        for match in re.finditer(r"(?:^|\n)Action:\s*(read|write|edit|bash|python)\s+(.+?)(?=\n(?:Plan|Observation|Reflection|Action|Final Answer)|\Z)", text, re.DOTALL | re.IGNORECASE):
            tool_name = match.group(1).lower()
            args_raw = match.group(2).strip()
            actions.append({"tool": tool_name, "raw_args": args_raw})

        # 2. Markdown code execution blocks: ```(bash|python) ... ```
        for match in re.finditer(r"```(bash|sh|python)\n(.*?)```", text, re.DOTALL):
            lang = match.group(1).lower()
            code = match.group(2).strip()
            actions.append({"tool": "bash" if lang in ("bash", "sh") else "python", "raw_args": code, "source": "code_block"})

        # 3. JSON tool calls (markdown json blocks or <tool_call> blocks)
        for call in structured:
            actions.append({"tool": call["name"], "params": call["args"], "valid_json": call["valid_json"]})

        return actions

    def execute_action_in_sandbox(self, act: Dict[str, Any], sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
        """Executes Pi foundational tool in real sandbox."""
        tool = str(act.get("tool", "")).lower()
        raw = str(act.get("raw_args", "")).strip()
        params = act.get("params", {})
        if not act.get("valid_json", True) or not isinstance(params, dict):
            return False, "Malformed tool arguments. Emit a complete JSON object."

        if tool in ("bash", "sh"):
            cmd = raw or params.get("command") or params.get("cmd") or ""
            return sandbox.run_bash(cmd)

        elif tool == "python":
            code = raw or params.get("code") or ""
            return sandbox.run_python(code)

        elif tool in ("read", "read_file"):
            path = raw or params.get("path") or ""
            return sandbox.read_file(path)

        elif tool in ("write", "write_file"):
            if params:
                path = params.get("path") or "output.txt"
                content = params.get("content") or ""
            else:
                lines = raw.split("\n", 1)
                path = lines[0].strip()
                content = lines[1] if len(lines) > 1 else ""
            return sandbox.write_file(path, content)

        elif tool in ("edit", "edit_file"):
            path = params.get("path") or ""
            edits = params.get("edits")
            if isinstance(edits, list) and edits:
                ok_r, orig_content = sandbox.read_file(path)
                if not ok_r:
                    return False, orig_content
                spans: List[Tuple[int, int, str]] = []
                for entry in edits:
                    if not isinstance(entry, dict):
                        return False, "Invalid edit entry: expected object"
                    old_t = str(entry.get("oldText") if "oldText" in entry else entry.get("old_str", ""))
                    new_t = str(entry.get("newText") if "newText" in entry else entry.get("new_str", ""))
                    if not old_t:
                        return False, f"Empty oldText in edit for {path}"
                    first_idx = orig_content.find(old_t)
                    if first_idx == -1:
                        return False, f"Target string not found in {path}"
                    if orig_content.find(old_t, first_idx + 1) != -1:
                        return False, f"Ambiguous oldText in {path}: matches multiple locations"
                    spans.append((first_idx, first_idx + len(old_t), new_t))
                spans.sort(key=lambda s: s[0])
                for idx_s in range(len(spans) - 1):
                    if spans[idx_s][1] > spans[idx_s + 1][0]:
                        return False, f"Overlapping edits rejected for {path}"
                updated = orig_content
                for start_i, end_i, replacement in reversed(spans):
                    updated = updated[:start_i] + replacement + updated[end_i:]
                ok_w, msg_w = sandbox.write_file(path, updated)
                if not ok_w:
                    return False, msg_w
                return True, f"Successfully applied {len(spans)} atomic edit(s) to {path}"
            old_str = params.get("oldText") or params.get("old_str") or ""
            new_str = params.get("newText") or params.get("new_str") or ""
            return sandbox.edit_file(path, old_str, new_str)

        return False, f"Unknown Pi tool: {tool}"

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 8,
        task_id: str = "",
    ) -> HarnessResult:
        """Runs the authentic Pi multi-turn plan/action/observation cycle."""
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

                actions = self.parse_actions(response)
                if not actions:
                    if self.has_final_answer(response):
                        is_completed, stop_reason = True, "final_response"
                        break
                    if turn >= max_turns - 1:
                        stop_reason = "missing_final_answer" if response.strip() else "empty_response"
                        break
                    curr_prompt = self.unfinished_feedback(curr_prompt, response)
                    continue

                turn_feedbacks = []
                failures_before = tools_failed
                for act in actions:
                    tools_called += 1
                    ok, msg = self.execute_action_in_sandbox(act, sandbox)
                    self.log_tool_result(turn, act.get("tool"), ok, msg)
                    if ok:
                        tools_succeeded += 1
                    else:
                        tools_failed += 1

                    tag = "Observation" if ok else "Action Error"
                    turn_feedbacks.append(f"{tag}: {msg}")
                    history.append({"turn": turn, "action": act, "success": ok, "output": msg})

                if (self.has_final_answer(response) and tools_failed == failures_before
                        and all(action.get("source") == "code_block" for action in actions)):
                    is_completed = True
                    stop_reason = "final_response"
                    break
                if turn >= max_turns - 1:
                    break

                feedback_str = "\n".join(turn_feedbacks)
                curr_prompt = (
                    f"{curr_prompt}\n\nModel:\n{response}\n\n"
                    f"{feedback_str}\n\nInspect the observations, edit or verify as needed, then provide Final Answer: when done."
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
