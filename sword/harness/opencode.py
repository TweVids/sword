"""
OpenCode CLI Harness: Anomaly's terminal-first dual-agent coding harness (Plan / Build).
Executes real bash commands, Python scripts, and workspace file tools in an isolated sandbox,
capturing real execution status, tool success rates, and reasoning traces.
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult
from sword.harness.sandbox import WorkspaceSandbox
from sword.harness.protocol import structured_calls, visible_response


class OpenCodeHarness(BaseHarness):
    """
    OpenCode agentic loop adapter.
    Dispatches tool calls to an isolated WorkspaceSandbox for real execution.
    Tracks Plan and Build modes, real exit codes, and output telemetry.
    """
    name: str = "opencode"

    def __init__(self, timeout_sec: float = 10.0, workspace_dir: Optional[str] = None):
        self.timeout_sec = timeout_sec
        self.workspace_dir = workspace_dir

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
        """Parses structured OpenCode tool invocations or code blocks from model response."""
        calls = []
        text = visible_response(text)

        # 1. OpenCode JSON tool_call blocks
        for call in structured_calls(text):
            calls.append({"type": "tool_call", "payload": {"name": call["name"], "args": call["args"]},
                          "valid_json": call["valid_json"]})

        # 2. Markdown bash / python code execution blocks
        for match in re.finditer(r"```(bash|sh|python)\n(.*?)```", text, re.DOTALL):
            lang = match.group(1).lower()
            code = match.group(2).strip()
            calls.append({"type": lang, "payload": {"code": code}, "raw": code})

        # 3. File directive blocks: ```file:path/to/file.ext\ncontent\n```
        for match in re.finditer(r"```file:([^\n]+)\n(.*?)```", text, re.DOTALL):
            fpath = match.group(1).strip()
            fcontent = match.group(2)
            calls.append({"type": "write_file", "payload": {"path": fpath, "content": fcontent}, "raw": fcontent})

        return calls

    def execute_tool_in_sandbox(self, call: Dict[str, Any], sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
        """Dispatches tool call to real sandbox execution."""
        ctype = call.get("type", "")
        payload = call.get("payload", {})

        if ctype in ("bash", "sh"):
            cmd = payload.get("code") or payload.get("command") or call.get("raw", "")
            return sandbox.run_bash(cmd)

        elif ctype == "python":
            code = payload.get("code") or call.get("raw", "")
            return sandbox.run_python(code)

        elif ctype == "write_file":
            path = payload.get("path") or "output.txt"
            content = payload.get("content") or ""
            return sandbox.write_file(path, content)

        elif ctype == "tool_call":
            if not call.get("valid_json", True):
                return False, "Malformed JSON tool call. Emit complete arguments before requesting execution."
            if not isinstance(payload, dict):
                return False, "Malformed tool_call payload: expected JSON object"

            action = str(payload.get("action") or payload.get("tool") or payload.get("name") or "").lower()
            params = payload.get("parameters") or payload.get("args") or payload
            if not isinstance(params, dict):
                return False, "Tool arguments must be a JSON object."

            if action in ("shell", "bash", "exec", "exec_command", "terminal", "command"):
                cmd = str(params.get("command") or params.get("cmd") or payload.get("command") or "")
                return sandbox.run_bash(cmd)
            elif action in ("python", "python_eval", "script"):
                code = str(params.get("code") or params.get("script") or payload.get("code") or "")
                return sandbox.run_python(code)
            elif action in ("write", "write_file", "create_file"):
                path = str(params.get("path") or params.get("file") or "workspace.txt")
                content = str(params.get("content") or "")
                return sandbox.write_file(path, content)
            elif action in ("read", "read_file"):
                path = str(params.get("path") or params.get("file") or "")
                offset = params.get("offset")
                limit = params.get("limit")
                end_line = (int(offset) + int(limit) - 1) if (offset is not None and limit is not None) else None
                return sandbox.read_file(path, start_line=offset, end_line=end_line)
            elif action in ("edit", "edit_file"):
                path = str(params.get("path") or "")
                old_str = str(params.get("oldString") or params.get("old_str") or "")
                new_str = str(params.get("newString") or params.get("new_str") or "")
                replace_all = bool(params.get("replaceAll", False))
                return sandbox.edit_file(path, old_str, new_str, replace_all=replace_all)
            else:
                return False, f"Unknown OpenCode tool action: {action}"

        return False, f"Unrecognized tool type: {ctype}"

    def run(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
        max_turns: int = 8,
        task_id: str = "",
    ) -> HarnessResult:
        """Executes the authentic OpenCode multi-turn interaction loop in a WorkspaceSandbox."""
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
                r_chunk = self.extract_reasoning(response)
                if r_chunk:
                    reasoning_text = r_chunk

                calls = self.parse_tools(response)
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
                failures_before = tools_failed
                for c in calls:
                    tools_called += 1
                    ok, output = self.execute_tool_in_sandbox(c, sandbox)
                    self.log_tool_result(turn, c.get("payload", {}).get("name", c["type"]), ok, output)
                    if ok:
                        tools_succeeded += 1
                    else:
                        tools_failed += 1

                    tag = "[Tool Success]" if ok else "[Tool Failed]"
                    turn_feedbacks.append(f"{tag}: {output}")
                    history.append({
                        "turn": turn,
                        "call": c,
                        "success": ok,
                        "output": output,
                    })

                if (self.has_final_answer(response) and tools_failed == failures_before
                        and all(call["type"] in ("python", "bash", "sh") for call in calls)):
                    is_completed = True
                    stop_reason = "final_response"
                    break
                if turn >= max_turns - 1:
                    break

                feedback_block = "\n".join(turn_feedbacks)
                curr_prompt = (
                    f"{curr_prompt}\n\nModel:\n{response}\n\n"
                    f"OpenCode Environment:\n{feedback_block}\n\n"
                    "Inspect the tool results, edit or verify as needed, then provide Final Answer: when done. "
                    "Successful tool execution alone does not complete the task."
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
            is_completed=is_completed,
            error_message=None if is_completed else "No final answer before " + stop_reason,
            turns=len(assistant_turns), stop_reason=stop_reason, assistant_turns=assistant_turns,
            workspace_dir=sandbox.root_dir, workspace_removed=sandbox.was_removed,
        )
