"""Dataset-defined checks and a one-shot executor for coding rows without a harness."""
import json
import re
import shlex
import subprocess
import sys
from typing import Any

from sword.harness.base import HarnessResult
from sword.harness.protocol import visible_response
from sword.harness.sandbox import WorkspaceSandbox


def normalize_tests(value: Any) -> list:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = value
        value = decoded
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("tests must be Python test code or a list of test definitions")
    tests = []
    for index, item in enumerate(value):
        item = {"code": item} if isinstance(item, str) else item
        if not isinstance(item, dict) or bool(item.get("code")) == bool(item.get("command")):
            raise ValueError("Each dataset test must specify exactly one of code or command")
        timeout = float(item.get("timeout_sec", 10))
        if not 0 < timeout <= 300:
            raise ValueError("Dataset test timeout_sec must be between 0 and 300")
        tests.append({"name": str(item.get("name", f"test_{index + 1}")),
                      **{key: item[key] for key in ("code", "command", "stdin", "expected_stdout") if key in item},
                      "timeout_sec": timeout})
    return tests


def coding_row(row) -> bool:
    return bool(row.tests or row.language or re.search(
        r"(?:^|[_\s-])(code|coding|programming|swe|software)(?:$|[_\s-])", row.category,
        re.IGNORECASE) or re.search(r"```(?:python|py|javascript|js|bash|sh)\s*\n", row.problem))


def candidate_source(response, language="python"):
    text = visible_response(response).strip()
    language = {"py": "python", "js": "javascript", "sh": "bash"}.get(language, language)
    aliases = {"python": {"python", "py", ""}, "javascript": {"javascript", "js"},
               "bash": {"bash", "sh"}}.get(language, {language})
    blocks = [match.group(2) for match in re.finditer(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL)
              if match.group(1).strip().lower() in aliases]
    if blocks:
        return "\n\n".join(blocks).strip(), language
    # A plain source answer is accepted; prose naturally fails syntax/runtime checks.
    return re.sub(r"^\s*Final (?:Answer|Solution)\s*:\s*", "", text, flags=re.IGNORECASE), language


def run_dataset_tests(sandbox, tests, response="", language="python"):
    if not tests:
        return []
    source, language = candidate_source(response, language)
    # Harnesses usually create project files themselves. Materialize fenced final
    # code only when there is no solution module already in the workspace.
    if source and "```" in visible_response(response) and language == "python":
        from pathlib import Path
        if not (Path(sandbox.root_dir) / "solution.py").exists():
            sandbox.write_file("solution.py", source)
    results = []
    for test in tests:
        timeout = sandbox.timeout_sec
        sandbox.timeout_sec = test["timeout_sec"]
        try:
            if "code" in test:
                # Tests may call solution functions directly or import project files.
                code = ("import os, runpy\n"
                        "scope = runpy.run_path('solution.py') if os.path.isfile('solution.py') else {}\n"
                        f"exec({test['code']!r}, scope)\n")
                ok, output = sandbox.run_python(code, stdin=test.get("stdin"))
            else:
                python = subprocess.list2cmdline([sys.executable]) if sys.platform == "win32" else shlex.quote(sys.executable)
                command = str(test["command"]).replace("{python}", python)
                ok, output = sandbox.run_bash(command, stdin=test.get("stdin"))
            execution = dict(sandbox.last_execution_result or {})
            expected = test.get("expected_stdout")
            if expected is not None:
                ok = ok and execution.get("stdout", "").strip() == str(expected).strip()
            results.append({**execution, "name": test["name"], "success": ok,
                            "output": output, **({"expected_stdout": str(expected)} if expected is not None else {})})
        finally:
            sandbox.timeout_sec = timeout
    return results


def execute_coding_answer(row, response, *, parent_dir=None, keep_workspace=False):
    sandbox = WorkspaceSandbox(parent_dir=parent_dir, cleanup=not keep_workspace)
    history = []
    tests = []
    try:
        for path, content in row.initial_files.items():
            ok, output = sandbox.write_file(path, content)
            if not ok:
                raise ValueError(output)
        fence = re.search(r"```(python|py|javascript|js|bash|sh)\s*\n", visible_response(response))
        source, language = candidate_source(response, row.language or (fence.group(1) if fence else "python"))
        filename, command = {"python": ("solution.py", [sys.executable, "solution.py"]),
                             "javascript": ("solution.js", ["node", "solution.js"]),
                             "bash": ("solution.sh", ["bash", "solution.sh"])}.get(language, ("", None))
        if command is None:
            history.append({"tool": "code_executor", "success": False,
                            "output": f"Unsupported executor language: {language}"})
        elif not source:
            history.append({"tool": "code_executor", "success": False, "output": "No final source code supplied"})
        else:
            ok, output = sandbox.write_file(filename, source)
            if ok:
                ok, output = sandbox.run_command(command)
            history.append({"tool": "code_executor", "success": ok, "output": output,
                            **dict(sandbox.last_execution_result or {})})
        tests = run_dataset_tests(sandbox, row.tests, response, language)
    finally:
        sandbox.close()
    succeeded = sum(bool(item["success"]) for item in history)
    return HarnessResult(final_text=response, tools_called=len(history), tools_succeeded=succeeded,
                         tools_failed=len(history) - succeeded, tool_history=history,
                         test_results=tests, execution_results=list(sandbox.execution_history),
                         turns=1, stop_reason="code_executor", workspace_dir=sandbox.root_dir,
                         workspace_removed=sandbox.was_removed)
