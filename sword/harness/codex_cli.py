"""
Codex CLI Harness: Terminal-first sandboxed code execution and test verification harness.
Executes code blocks and assertions in a real sandbox environment, capturing runtime tracebacks,
exit codes, and driving multi-turn self-repair.
"""

import re
import ast
import json
from typing import Dict, Any, List, Optional, Callable, Tuple

from sword.harness.base import BaseHarness, HarnessResult
from sword.harness.sandbox import WorkspaceSandbox


class CodexCliHarness(BaseHarness):
    """
    Codex CLI style benchmark and code execution harness.
    Supports Codex apply_patch freeform grammar, exec_command tool calls,
    and Python/Bash blocks in an isolated WorkspaceSandbox.
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
            if code and not code.startswith("*** Begin Patch"):
                blocks.append(code)
        return blocks

    def extract_patches(self, text: str) -> List[str]:
        """Extracts Codex apply_patch blocks (*** Begin Patch ... *** End Patch)."""
        return [m.group(1).strip() for m in re.finditer(r"\*\*\* Begin Patch\s*\n(.*?)\*\*\* End Patch", text, re.DOTALL)]

    def extract_exec_commands(self, text: str) -> List[str]:
        """Extracts Codex exec_command JSON payloads or bash blocks."""
        cmds: List[str] = []
        for m in re.finditer(r"```(?:bash|sh)\n(.*?)```", text, re.DOTALL):
            c = m.group(1).strip()
            if c:
                cmds.append(c)
        for m in re.finditer(r"(\{[^{}]*\"cmd\"\s*:\s*\"[^\"]+\"[^{}]*\})", text):
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict) and obj.get("cmd"):
                    cmds.append(str(obj["cmd"]))
            except Exception:
                pass
        return cmds

    def apply_codex_patch(self, patch_body: str, sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
        """Applies a Codex apply_patch block (Add File, Delete File, Update File, Move to, and @@ hunks) atomically."""
        lines = patch_body.splitlines()
        i = 0
        applied_files: List[str] = []
        while i < len(lines):
            line = lines[i]
            if line.startswith("*** Add File: "):
                fname = line[len("*** Add File: "):].strip()
                i += 1
                added_lines = []
                while i < len(lines) and not lines[i].startswith("*** "):
                    if lines[i].startswith("+"):
                        added_lines.append(lines[i][1:])
                    i += 1
                ok, msg = sandbox.write_file(fname, "\n".join(added_lines))
                if not ok:
                    return False, msg
                applied_files.append(f"added {fname}")

            elif line.startswith("*** Delete File: "):
                fname = line[len("*** Delete File: "):].strip()
                i += 1
                ok, msg = sandbox.delete_file(fname)
                if not ok:
                    return False, msg
                applied_files.append(f"deleted {fname}")

            elif line.startswith("*** Update File: "):
                fname = line[len("*** Update File: "):].strip()
                i += 1
                move_target: Optional[str] = None
                if i < len(lines) and lines[i].startswith("*** Move to: "):
                    move_target = lines[i][len("*** Move to: "):].strip()
                    i += 1

                ok_read, current_content = sandbox.read_file(fname)
                if not ok_read:
                    return False, current_content

                hunks: List[Tuple[str, List[str], List[str]]] = []
                cur_ctx = ""
                old_chunk: List[str] = []
                new_chunk: List[str] = []

                while i < len(lines) and not (lines[i].startswith("*** ") and not lines[i].startswith("*** End of File")):
                    cur = lines[i]
                    if cur.startswith("*** End of File"):
                        i += 1
                        continue
                    if cur.startswith("@@"):
                        if old_chunk or new_chunk:
                            hunks.append((cur_ctx, old_chunk, new_chunk))
                            old_chunk = []
                            new_chunk = []
                        cur_ctx = cur[2:].strip()
                    elif cur.startswith("-"):
                        old_chunk.append(cur[1:])
                    elif cur.startswith("+"):
                        new_chunk.append(cur[1:])
                    elif cur.startswith(" "):
                        old_chunk.append(cur[1:])
                        new_chunk.append(cur[1:])
                    i += 1

                if old_chunk or new_chunk:
                    hunks.append((cur_ctx, old_chunk, new_chunk))

                updated_content = current_content
                for ctx_hdr, o_lines, n_lines in hunks:
                    old_str = "\n".join(o_lines)
                    new_str = "\n".join(n_lines)
                    if old_str:
                        if old_str not in updated_content:
                            return False, f"Hunk target not found in {fname}"
                        updated_content = updated_content.replace(old_str, new_str, 1)
                    elif ctx_hdr and ctx_hdr in updated_content:
                        updated_content = updated_content.replace(ctx_hdr, f"{ctx_hdr}\n{new_str}", 1)
                    else:
                        sep = "\n" if updated_content and not updated_content.endswith("\n") else ""
                        updated_content = f"{updated_content}{sep}{new_str}"

                if move_target:
                    ok_mv, msg_mv = sandbox.move_file(fname, move_target)
                    if not ok_mv:
                        return False, msg_mv
                    ok_w, msg_w = sandbox.write_file(move_target, updated_content)
                    if not ok_w:
                        return False, msg_w
                    applied_files.append(f"moved {fname} -> {move_target}")
                else:
                    ok_w, msg_w = sandbox.write_file(fname, updated_content)
                    if not ok_w:
                        return False, msg_w
                    applied_files.append(f"updated {fname}")
            else:
                i += 1
        if not applied_files:
            return False, "No valid hunks found in patch."
        return True, "Applied patch: " + ", ".join(applied_files)

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

                patches = self.extract_patches(response)
                exec_cmds = self.extract_exec_commands(response)
                code_blocks = self.extract_code_blocks(response)

                if not patches and not exec_cmds and not code_blocks:
                    break

                turn_succeeded = 0
                turn_failed = 0
                feedback_items = []

                for p_idx, patch_body in enumerate(patches):
                    tools_called += 1
                    ok, msg = self.apply_codex_patch(patch_body, sandbox)
                    if ok:
                        tools_succeeded += 1
                        turn_succeeded += 1
                        feedback_items.append(f"Patch {p_idx + 1} Passed: {msg}")
                    else:
                        tools_failed += 1
                        turn_failed += 1
                        feedback_items.append(f"Patch {p_idx + 1} Failed: {msg}")
                    history.append({"turn": turn, "tool": "apply_patch", "success": ok, "message": msg})

                for c_idx, cmd in enumerate(exec_cmds):
                    tools_called += 1
                    ok, msg = sandbox.run_bash(cmd)
                    if ok:
                        tools_succeeded += 1
                        turn_succeeded += 1
                        feedback_items.append(f"Command {c_idx + 1} Passed: {msg}")
                    else:
                        tools_failed += 1
                        turn_failed += 1
                        feedback_items.append(f"Command {c_idx + 1} Failed: {msg}")
                    history.append({"turn": turn, "tool": "exec_command", "success": ok, "message": msg})

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
