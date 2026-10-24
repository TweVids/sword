"""
Workspace Sandbox: Isolated execution environment for agentic coding harnesses.
Provides real subprocess execution, filesystem tools (read, write, edit, list),
and execution safety policies for OpenCode, Codex CLI, Gemini CLI, and Pi.
"""

import os
import sys
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
        if not full_path.startswith(self.root_dir):
            raise PermissionError(f"Access denied: path '{path}' escapes workspace boundary.")
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

    def read_file(self, path: str) -> Tuple[bool, str]:
        """Reads file contents from inside the sandbox."""
        try:
            full_path = self._resolve_safe_path(path)
            if not os.path.exists(full_path):
                return False, f"File not found: {path}"
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return True, content
        except Exception as e:
            return False, f"read_file error: {e}"

    def edit_file(self, path: str, old_str: str, new_str: str) -> Tuple[bool, str]:
        """Replaces old_str with new_str in a file."""
        try:
            full_path = self._resolve_safe_path(path)
            if not os.path.exists(full_path):
                return False, f"File not found: {path}"
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if old_str not in content:
                return False, f"Target string not found in {path}"
            content = content.replace(old_str, new_str, 1)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, f"Successfully edited {path}"
        except Exception as e:
            return False, f"edit_file error: {e}"

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
