"""Dataset-defined checks and execution of the policy's final source code."""
import ast
import json
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from sword.harness.base import HarnessResult
from sword.harness.protocol import visible_response
from sword.harness.sandbox import WorkspaceSandbox


_LANGUAGES = {"python": "python", "py": "python", "javascript": "javascript",
              "js": "javascript", "bash": "bash", "sh": "bash"}
_CODE_BLOCK = re.compile(r"```([^\n`]*)\n(.*?)```|'''(.*?)'''", re.DOTALL)
_STDOUT_COMPARISON = "Text equality after normalizing line endings and ignoring final newlines."


def code_io_text(value):
    """Strings are literal stream contents; other JSON values use JSON text."""
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def normalize_code_tests(value: Any) -> list:
    """Normalize the single dataset column containing input/output case pairs."""
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, str):
        def unique_keys(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("Duplicate code_tests key %r; put each input/output pair in a separate object" % key)
                result[key] = item
            return result
        try:
            value = json.loads(value, object_pairs_hook=unique_keys)
        except ValueError as exc:
            raise ValueError("code_tests must be a JSON list of separate input/output objects: %s" % exc) from exc
    if not isinstance(value, list):
        raise ValueError("code_tests must be a list of input/output objects")
    cases = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or "input" not in item or "output" not in item:
            raise ValueError("Each code_tests object must contain both input and output")
        try:
            code_io_text(item["input"])
            code_io_text(item["output"])
            timeout = float(item.get("timeout_sec", 10))
        except (TypeError, ValueError) as exc:
            raise ValueError("code_tests input/output must be JSON values and timeout_sec must be numeric") from exc
        if not 0 < timeout <= 300:
            raise ValueError("code_tests timeout_sec must be between 0 and 300")
        cases.append({**item, "name": str(item.get("name", "code_case_%s" % (index + 1))),
                      "timeout_sec": timeout})
    return cases


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
    return bool(row.tests or getattr(row, "code_tests", []) or row.language or re.search(
        r"(?:^|[_\s-])(code|coding|programming|swe|software)(?:$|[_\s-])", row.category,
        re.IGNORECASE) or has_candidate_code(row.problem))


def _candidate_text(response):
    text = visible_response(response).strip()
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        pass
    else:
        # Plain Python source may also contain answer labels in string literals.
        return text
    # A program may print a literal "Final Answer:" line. Only response labels
    # outside code blocks delimit the final emitted source.
    labels_text = _CODE_BLOCK.sub(lambda match: match.group(0)[0] + " " * (len(match.group(0)) - 2)
                                 + match.group(0)[-1], text)
    labels = list(re.finditer(r"(?im)^\s*(?:#{1,6}\s*)?Final (?:Answer|Solution)\s*:\s*", labels_text))
    return text[labels[-1].end():].strip() if labels else text


def _code_blocks(response):
    # Markdown matches consume any Python docstrings inside them, so a nested
    # triple-quoted string cannot accidentally replace the surrounding code.
    for match in _CODE_BLOCK.finditer(_candidate_text(response)):
        if match.group(3) is None:
            tag = match.group(1).strip().lower()
            yield textwrap.dedent(match.group(2)).strip(), _LANGUAGES.get(tag, tag) or None, "markdown"
        else:
            body = match.group(3).lstrip("\r\n")
            header, separator, rest = body.partition("\n")
            tag = header.strip().lower()
            if separator and tag in _LANGUAGES:
                yield textwrap.dedent(rest).strip(), _LANGUAGES[tag], "quoted"
            else:
                yield textwrap.dedent(body).strip(), None, "quoted"


def has_candidate_code(response):
    """Detect supported final fences or syntactically valid quoted Python code."""
    for source, language, delimiter in _code_blocks(response):
        if language in ("javascript", "bash") or (delimiter == "markdown" and language == "python"):
            return True
        if language not in (None, "python") or not source:
            continue
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        # A bare word or literal in quotation marks is usually prose. Calls,
        # assignments, definitions and other executable statements are code.
        if any(not (isinstance(node, ast.Expr) and isinstance(node.value, (ast.Name, ast.Constant)))
               for node in tree.body):
            return True
    return False


def candidate_source(response, language=None):
    blocks = list(_code_blocks(response))
    language = str(language).strip().lower() if language else next(
        (tag for _, tag, _ in blocks if tag in ("python", "javascript", "bash")), "python")
    language = _LANGUAGES.get(language, language)
    blocks = [source for source, tag, _ in blocks if tag is None or tag == language]
    if blocks:
        return "\n\n".join(blocks).strip(), language
    # A plain source answer is accepted; prose naturally fails syntax/runtime checks.
    return _candidate_text(response), language


def _code_command(language):
    return {"python": ("solution.py", [sys.executable, "solution.py"]),
            "javascript": ("solution.js", ["node", "solution.js"]),
            "bash": ("solution.sh", ["bash", "solution.sh"])}.get(language, ("", None))


def _run_code_cases(sandbox, cases, source, language, filename, command, preparation_error=""):
    results = []
    for case in cases:
        stdin, expected = code_io_text(case["input"]), code_io_text(case["output"])
        timeout = sandbox.timeout_sec
        sandbox.timeout_sec = case["timeout_sec"]
        executed = not preparation_error
        try:
            if executed:
                # Each case must execute the same emitted source, even if the
                # previous process edited its own entrypoint.
                ok, message = sandbox.write_file(filename, source)
                if not ok:
                    executed = False
                    preparation_error = message
            if executed:
                sandbox.last_execution_result = None
                ok, output = sandbox.run_command(command, stdin=stdin)
                execution = dict(sandbox.last_execution_result or {})
            else:
                ok, output = False, preparation_error
                execution = {"stdout": "", "stderr": preparation_error, "exit_code": None, "timed_out": False}
            # Compare the full stream before the log's output cap is applied.
            stdout = getattr(sandbox, "last_stdout", None) if executed else ""
            if stdout is None:
                stdout = execution.get("stdout", "")
            normalized = lambda text: text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
            matched = executed and normalized(stdout) == normalized(expected)
            results.append({**execution, "kind": "code_io", "name": case["name"],
                            "input": case["input"], "output": case["output"], "stdin": stdin,
                            "expected_stdout": expected, "output_match": matched,
                            "process_success": bool(ok), "success": bool(ok and matched), "executed": executed,
                            "code_file": filename, "language": language, "timeout_sec": case["timeout_sec"],
                            "comparison": _STDOUT_COMPARISON, "message": output})
        finally:
            sandbox.timeout_sec = timeout
    sandbox.code_execution = {
        "note": "The runner attempted each case separately using the generated source. For executed cases, it supplied stdin "
                "from input and captured stdout/stderr. output is the dataset expectation; stdout is the actual result. "
                "executed=false indicates that source preparation failed.",
        "source": source, "language": language, "code_file": filename, "cases": results,
    }
    return results


def run_dataset_tests(sandbox, tests, response="", language=None, *, code_tests=()):
    if not tests and not code_tests:
        return []
    source, language = candidate_source(response, language)
    filename, command = _code_command(language)
    error = "Unsupported executor language: %s" % language if command is None else ""
    target = Path(sandbox.root_dir) / filename if filename else None
    blocks = list(_code_blocks(response))
    fenced = any(tag is None or tag == language for _, tag, _ in blocks)
    if code_tests and blocks:
        if not fenced:
            error = "Final code block language does not match the dataset language"
        elif not source:
            error = "Final code block is empty"
    # For I/O checks the final emitted code is authoritative. Harness project
    # files remain usable when the final answer describes an implementation.
    if target is not None and source and ((fenced and (code_tests or not target.exists()))
                                         or (code_tests and not target.exists())):
        ok, message = sandbox.write_file(filename, source)
        if not ok:
            error = message
    if code_tests and target is not None:
        if target.is_file() and not error:
            try:
                source = target.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                error = "Cannot read generated source: %s" % exc
        elif not error:
            error = "No final source code or solution file supplied"
    results = _run_code_cases(sandbox, code_tests, source, language, filename, command, error) if code_tests else []
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
        source, language = candidate_source(response, row.language)
        filename, command = _code_command(language)
        code_tests = getattr(row, "code_tests", [])
        if code_tests:
            tests = run_dataset_tests(sandbox, row.tests, response, language, code_tests=code_tests)
            history.extend({**test, "tool": "code_executor", "success": test["process_success"]}
                           for test in tests if test.get("kind") == "code_io")
        elif command is None:
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
        if not code_tests:
            tests = run_dataset_tests(sandbox, row.tests, response, language)
    finally:
        sandbox.close()
    succeeded = sum(bool(item["success"]) for item in history)
    return HarnessResult(final_text=response, tools_called=len(history), tools_succeeded=succeeded,
                         tools_failed=len(history) - succeeded, tool_history=history,
                         test_results=tests, execution_results=list(sandbox.execution_history),
                         code_execution=getattr(sandbox, "code_execution", {}),
                         turns=1, stop_reason="code_executor", workspace_dir=sandbox.root_dir,
                         workspace_removed=sandbox.was_removed)
