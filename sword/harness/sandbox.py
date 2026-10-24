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
from typing import Dict, Any, List, Optional, Tuple


class WorkspaceSandbox:
    """
    Isolated directory workspace for agentic harness tool execution.
    Executes real subprocesses (bash, python) and real file operations in an isolated folder.
    """
    def __init__(self, base_dir: Optional[str] = None, timeout_sec: float = 10.0, cleanup: bool = True):
        self.cleanup = cleanup
        self.timeout_sec = timeout_sec
        if base_dir:
            self.root_dir = os.path.abspath(base_dir)
            os.makedirs(self.root_dir, exist_ok=True)
            self._created_temp = False
        else:
            self.root_dir = os.path.abspath(tempfile.mkdtemp(prefix="sword_harness_workspace_"))
            self._created_temp = True

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

    def run_bash(self, command: str) -> Tuple[bool, str]:
        """Executes a shell command inside the sandbox."""
        cmd_clean = command.strip()
        if not cmd_clean:
            return False, "Empty shell command"

        dangerous = ("rm -rf /", ":(){ :|:& };:", "dd if=/dev/zero", "mkfs.")
        if any(bad in cmd_clean for bad in dangerous):
            return False, "Command rejected: dangerous operation detected."

        try:
            shell_executable = None
            if os.name == "nt":
                # On Windows, execute via cmd / powershell
                shell_executable = True
            else:
                shell_executable = "/bin/bash" if os.path.exists("/bin/bash") else True

            res = subprocess.run(
                cmd_clean,
                shell=True,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                executable=shell_executable if shell_executable is not True else None,
            )
            success = (res.returncode == 0)
            output = res.stdout if success else (res.stderr or res.stdout or f"Exit code {res.returncode}")
            return success, output[:2000].strip() or "Command completed with no output (exit code 0)"
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {self.timeout_sec}s"
        except Exception as e:
            return False, f"Execution failed: {e}"

    def run_python(self, code: str) -> Tuple[bool, str]:
        """Executes Python code inside the sandbox using the active Python interpreter."""
        code_clean = code.strip()
        if not code_clean:
            return False, "Empty Python code"

        script_path = os.path.join(self.root_dir, "_agent_runner.py")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code_clean)

            res = subprocess.run(
                [sys.executable, script_path],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            success = (res.returncode == 0)
            output = res.stdout if success else (res.stderr or res.stdout or f"Exit code {res.returncode}")
            return success, output[:2000].strip() or "Execution succeeded with no output"
        except subprocess.TimeoutExpired:
            return False, f"Python execution timed out after {self.timeout_sec}s"
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
            shutil.rmtree(self.root_dir, ignore_errors=True)
