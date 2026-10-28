"""
Workspace Sandbox: Isolated execution environment for agentic coding harnesses.
Provides real subprocess execution, filesystem tools (read, write, edit, list),
and execution safety policies for OpenCode, Codex CLI, Gemini CLI, and Pi.
"""

import os
import re
import sys
import glob as _glob
import fnmatch
import shutil
import tempfile
import subprocess
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple


class WorkspaceSandbox:
    """
    Isolated directory workspace for agentic harness tool execution.
    Executes real subprocesses (bash, python) and real file operations in an isolated folder.
    """
    def __init__(self, base_dir: Optional[str] = None, timeout_sec: float = 10.0, cleanup: bool = True,
                 parent_dir: Optional[str] = None):
        self.cleanup = cleanup
        self.timeout_sec = timeout_sec
        self.was_removed = False
        self.last_execution_result = None
        self.execution_history = deque(maxlen=32)
        if base_dir:
            self.root_dir = os.path.abspath(base_dir)
            os.makedirs(self.root_dir, exist_ok=True)
            self._created_temp = False
        else:
            if parent_dir:
                parent_dir = os.path.abspath(parent_dir)
                os.makedirs(parent_dir, exist_ok=True)
            self.root_dir = os.path.abspath(tempfile.mkdtemp(prefix="sword_harness_workspace_", dir=parent_dir))
            self._created_temp = True
        self._owned_root = os.path.realpath(self.root_dir)
        self._owned_parent = os.path.dirname(self._owned_root)

    def _resolve_safe_path(self, path: str) -> str:
        """Ensures path resides inside workspace root directory."""
        clean_path = path.strip().lstrip("/\\")
        full_path = os.path.abspath(os.path.join(self.root_dir, clean_path))
        try:
            if os.path.commonpath([self.root_dir, full_path]) != self.root_dir:
                raise PermissionError(f"Access denied: path '{path}' escapes workspace boundary.")
        except ValueError:
            raise PermissionError(f"Access denied: path '{path}' escapes workspace boundary across drives.")
        return full_path

    def write_file(self, path: str, content: str) -> Tuple[bool, str]:
        """Writes content to a file inside the sandbox."""
        try:
            full_path = self._resolve_safe_path(path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, f"File successfully written to {path} ({len(content)} bytes)"
        except Exception as e:
            return False, f"write_file error: {e}"

    def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Reads file contents (or 1-indexed line slice) from inside the sandbox."""
        try:
            full_path = self._resolve_safe_path(path)
            if not os.path.exists(full_path):
                return False, f"File not found: {path}"
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if start_line is not None or end_line is not None:
                lines = content.splitlines()
                s = max(1, int(start_line)) - 1 if start_line is not None else 0
                e = int(end_line) if end_line is not None else len(lines)
                content = "\n".join(lines[s:e])
            return True, content
        except Exception as e:
            return False, f"read_file error: {e}"

    def edit_file(self, path: str, old_str: str, new_str: str, replace_all: bool = False) -> Tuple[bool, str]:
        """Replaces old_str with new_str in a file."""
        try:
            full_path = self._resolve_safe_path(path)
            if not os.path.exists(full_path):
                return False, f"File not found: {path}"
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if old_str == "":
                content = content + ("\n" if content and not content.endswith("\n") else "") + new_str
            else:
                if old_str not in content:
                    return False, f"Target string not found in {path}"
                content = content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, f"Successfully edited {path}"
        except Exception as e:
            return False, f"edit_file error: {e}"

    def delete_file(self, path: str) -> Tuple[bool, str]:
        """Deletes a file inside the sandbox."""
        try:
            full_path = self._resolve_safe_path(path)
            if not os.path.isfile(full_path):
                return False, f"File not found: {path}"
            os.remove(full_path)
            return True, f"Deleted {path}"
        except Exception as e:
            return False, f"delete_file error: {e}"

    def move_file(self, src_path: str, dst_path: str) -> Tuple[bool, str]:
        """Moves/renames a file inside the sandbox."""
        try:
            src_full = self._resolve_safe_path(src_path)
            dst_full = self._resolve_safe_path(dst_path)
            if not os.path.exists(src_full):
                return False, f"Source file not found: {src_path}"
            os.makedirs(os.path.dirname(dst_full), exist_ok=True)
            shutil.move(src_full, dst_full)
            return True, f"Moved {src_path} to {dst_path}"
        except Exception as e:
            return False, f"move_file error: {e}"

    def list_dir(self, path: str = "") -> Tuple[bool, List[str]]:
        """Lists files and directories in workspace relative to root."""
        try:
            target_dir = self._resolve_safe_path(path) if path else self.root_dir
            if not os.path.isdir(target_dir):
                return False, [f"Not a directory: {path}"]
            entries = sorted(os.listdir(target_dir))
            return True, entries
        except Exception as e:
            return False, [str(e)]

    def glob_files(self, pattern: str, dir_path: str = "") -> Tuple[bool, List[str]]:
        """Finds files matching a glob pattern inside the sandbox."""
        if not pattern or not pattern.strip():
            return False, ["Empty glob pattern"]
        try:
            target_dir = self._resolve_safe_path(dir_path) if dir_path else self.root_dir
            if not os.path.isdir(target_dir):
                return False, [f"Not a directory: {dir_path}"]
            raw_matches = _glob.glob(os.path.join(target_dir, pattern), recursive=True)
            safe_rel: List[str] = []
            for m in sorted(raw_matches):
                abs_m = os.path.abspath(m)
                if os.path.commonpath([self.root_dir, abs_m]) == self.root_dir and os.path.isfile(abs_m):
                    safe_rel.append(os.path.relpath(abs_m, self.root_dir).replace("\\", "/"))
            return True, safe_rel
        except Exception as e:
            return False, [f"glob_files error: {e}"]

    def grep_search(
        self,
        pattern: str,
        dir_path: str = "",
        include_pattern: Optional[str] = None,
        max_matches: int = 100,
    ) -> Tuple[bool, str]:
        """Searches files inside the sandbox for a regular expression pattern."""
        if not pattern:
            return False, "Empty search pattern"
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return False, f"Invalid regex pattern: {e}"
        try:
            target_dir = self._resolve_safe_path(dir_path) if dir_path else self.root_dir
            if not os.path.exists(target_dir):
                return False, f"Directory not found: {dir_path}"
            matches: List[str] = []
            files_to_scan: List[str] = []
            if os.path.isfile(target_dir):
                files_to_scan.append(target_dir)
            else:
                for root, _, filenames in os.walk(target_dir):
                    for fn in sorted(filenames):
                        if include_pattern and not fnmatch.fnmatch(fn, include_pattern):
                            continue
                        files_to_scan.append(os.path.join(root, fn))

            for fpath in files_to_scan:
                rel = os.path.relpath(fpath, self.root_dir).replace("\\", "/")
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for lineno, line in enumerate(f, start=1):
                            if rx.search(line):
                                matches.append(f"{rel}:{lineno}:{line.rstrip()}")
                                if len(matches) >= max_matches:
                                    return True, "\n".join(matches)
                except Exception:
                    continue
            return True, "\n".join(matches) if matches else "No matches found"
        except Exception as e:
            return False, f"grep_search error: {e}"

    def run_command(self, command, *, shell=False, stdin=None) -> Tuple[bool, str]:
        """Run a process and retain bounded stdout/stderr separately from display text."""
        started = time.monotonic()
        try:
            res = subprocess.run(command, shell=shell, cwd=self.root_dir,
                                 capture_output=True, text=True, input=stdin,
                                 timeout=self.timeout_sec,
                                 executable="/bin/bash" if shell and os.name != "nt" and os.path.exists("/bin/bash") else None)
            result = {"success": res.returncode == 0, "exit_code": res.returncode,
                      "stdout": res.stdout[:8192], "stderr": res.stderr[:8192], "timed_out": False,
                      "output_truncated": len(res.stdout) > 8192 or len(res.stderr) > 8192}
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return (value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or "")[:8192]
            result = {"success": False, "exit_code": None, "timed_out": True,
                      "stdout": decoded(exc.stdout), "stderr": decoded(exc.stderr),
                      "error": f"Command timed out after {self.timeout_sec}s"}
        except Exception as exc:
            result = {"success": False, "exit_code": None, "timed_out": False,
                      "stdout": "", "stderr": str(exc), "error": f"Execution failed: {exc}"}
        result["duration_sec"] = round(time.monotonic() - started, 4)
        self.last_execution_result = result
        self.execution_history.append(result)
        output = result["stdout"] if result["success"] else (result.get("error") or result["stderr"] or result["stdout"] or f"Exit code {result['exit_code']}")
        return result["success"], output[:2000].strip() or "Command completed with no output (exit code 0)"

    def run_bash(self, command: str, stdin=None) -> Tuple[bool, str]:
        """Executes a shell command inside the sandbox."""
        self.last_execution_result = None
        cmd_clean = command.strip()
        if not cmd_clean:
            return False, "Empty shell command"

        dangerous = ("rm -rf /", ":(){ :|:& };:", "dd if=/dev/zero", "mkfs.")
        if any(bad in cmd_clean for bad in dangerous):
            return False, "Command rejected: dangerous operation detected."

        return self.run_command(cmd_clean, shell=True, stdin=stdin)

    def run_python(self, code: str, stdin=None) -> Tuple[bool, str]:
        """Executes Python code inside the sandbox using the active Python interpreter."""
        self.last_execution_result = None
        code_clean = code.strip()
        if not code_clean:
            return False, "Empty Python code"

        # Never overwrite a project file to execute an assistant's Python snippet.
        script_fd, script_path = tempfile.mkstemp(prefix=".sword_agent_", suffix=".py", dir=self.root_dir)
        os.close(script_fd)
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code_clean)

            return self.run_command([sys.executable, script_path], stdin=stdin)
        except Exception as e:
            return False, f"Python execution error: {e}"
        finally:
            if os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except Exception:
                    pass

    def close(self):
        """Cleans up workspace if temporary."""
        if self._created_temp and self.cleanup and os.path.exists(self.root_dir):
            resolved = os.path.realpath(self.root_dir)
            if resolved != self._owned_root or os.path.dirname(resolved) != self._owned_parent:
                raise PermissionError("Refusing to remove a workspace that is not the owned temporary directory")
            shutil.rmtree(self.root_dir, ignore_errors=True)
            self.was_removed = not os.path.exists(self.root_dir)
