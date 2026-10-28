"""Real executable checks, bounded evidence, and their effect on training rewards."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.execution import execute_coding_answer, normalize_tests
from sword.execution_evidence import build_execution_evidence
from sword.harness import OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness, WorkspaceSandbox
from sword.long_trainer import parse_dataset_row, load_dataset_records
from sword.reward import MultiMetricRewardScorer


def test_dataset_tests_and_seed_files_survive_serialization_and_change_hashes():
    raw = {"problem": "Implement add", "solution": "implementation", "category": "new_coding_task",
           "tests": [{"name": "addition", "code": "assert add(2, 3) == 5"}],
           "initial_files": {"data.txt": "seed"}, "language": "python"}
    row = parse_dataset_row(raw)
    assert row.tests[0]["name"] == "addition"
    assert row.to_dict()["initial_files"] == {"data.txt": "seed"}
    assert parse_dataset_row(row.to_dict()).row_hash == row.row_hash
    assert load_dataset_records([raw])[1] != load_dataset_records([{**raw, "tests": "assert add(2, 3) == 6"}])[1]
    old = parse_dataset_row({"problem": "p", "solution": "4"})
    assert "tests" not in old.to_dict()
    canonical = {key: value for key, value in old.to_dict().items() if key not in ("problem_id", "row_hash")}
    assert old.row_hash == hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]
    assert parse_dataset_row({"problem": "p", "tests": json.dumps(raw["tests"])}).tests == row.tests
    assert parse_dataset_row({"problem": "p", "test_commands": "echo checked"}).tests[0]["command"] == "echo checked"


@pytest.mark.parametrize("test", [{}, {"code": "pass", "command": "echo pass"}, {"code": "pass", "timeout_sec": 0}])
def test_invalid_test_definitions_fail_during_dataset_loading(test):
    with pytest.raises(ValueError):
        normalize_tests([test])


@pytest.mark.parametrize("expression,passed", [("a + b", True), ("a - b", False)])
def test_fallback_executor_runs_authoritative_tests_and_preserves_tool_totals(tmp_path, expression, passed):
    row = parse_dataset_row({"problem": "Implement add", "category": "coding", "tests": "assert add(2, 3) == 5"})
    response = f"```python\ndef add(a, b):\n    return {expression}\n```"
    result = execute_coding_answer(row, response, parent_dir=str(tmp_path))
    assert result.tools_called == result.tools_succeeded == 1
    assert result.test_results[0]["success"] is passed
    assert result.workspace_removed and not Path(result.workspace_dir).exists()
    evidence = build_execution_evidence(result.tool_history, tools_called=result.tools_called,
                                        tools_succeeded=result.tools_succeeded, test_results=result.test_results)
    reward = MultiMetricRewardScorer(None).score(row.problem, "", response, can_be_verify=False,
                                                tools_called=1, tools_succeeded=1, execution_evidence=evidence)
    assert reward.truth == float(passed)
    assert reward.tools_success == 1.0


def test_command_tests_match_stdout_and_have_fresh_workspaces(tmp_path):
    row = parse_dataset_row({"problem": "Read a seed", "language": "python", "initial_files": {"data.txt": "seed"},
                             "tests": [{"command": "{python} solution.py", "expected_stdout": "seed"},
                                       {"command": "{python} solution.py", "expected_stdout": "different"}]})
    response = "```python\nfrom pathlib import Path\nprint(Path('data.txt').read_text())\n```"
    first = execute_coding_answer(row, response, parent_dir=str(tmp_path), keep_workspace=True)
    second = execute_coding_answer(row, response, parent_dir=str(tmp_path), keep_workspace=True)
    assert first.workspace_dir != second.workspace_dir
    assert [test["success"] for test in first.test_results] == [True, False]
    assert first.test_results[1]["exit_code"] == 0


@pytest.mark.parametrize("cls", [OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness])
def test_existing_harness_runs_dataset_tests_before_workspace_cleanup(tmp_path, cls):
    harness = cls()
    harness.workspace_parent_dir = str(tmp_path)
    harness.dataset_tests = normalize_tests("assert add(2, 3) == 5")
    response = "Final Answer:\n```python\ndef add(a, b):\n    return a + b\n```"
    result = harness.run("Implement add", lambda _: response)
    assert result.is_completed and result.test_results[0]["success"]
    assert result.workspace_removed


def test_process_evidence_keeps_both_streams_and_timeout(tmp_path):
    sandbox = WorkspaceSandbox(base_dir=str(tmp_path), timeout_sec=0.1)
    ok, _ = sandbox.run_python("import sys\nprint('stdout')\nprint('stderr', file=sys.stderr)")
    assert ok
    assert sandbox.last_execution_result["stdout"].strip() == "stdout"
    assert sandbox.last_execution_result["stderr"].strip() == "stderr"
    ok, _ = sandbox.run_python("import time\ntime.sleep(2)")
    assert not ok and sandbox.last_execution_result["timed_out"]


def test_judge_evidence_has_exact_counts_prioritizes_failures_and_stays_bounded():
    history = [{"tool": "python", "success": True, "output": "x" * 20000} for _ in range(20)]
    history.append({"tool": "python", "success": False, "output": "REAL_FAILURE"})
    evidence = build_execution_evidence(history, tools_called=21, tools_succeeded=20, max_chars=1024)
    assert evidence["tools"] == {"called": 21, "succeeded": 20, "failed": 1}
    assert len(json.dumps(evidence, ensure_ascii=False)) <= 1024
    assert "REAL_FAILURE" in json.dumps(evidence)
    assert evidence["truncated"]
    judge = ModelBJudge(ModelBJudgeConfig(auto_setup=False, execution_evidence_chars=1024))
    prompt = judge.build_user_prompt("problem", "reference", "", "", "answer", execution_evidence=evidence)
    section = prompt.split("Measured Execution Evidence (runner-captured, bounded samples):\n")[1].split("\n\n")[0]
    assert json.loads(section) == evidence and len(section) <= 1024


def test_live_judge_cannot_override_failed_dataset_tests():
    judge = ModelBJudge(ModelBJudgeConfig(auto_setup=False, reasoning=False, fallback_model_name=None))
    judge.ensure_ready = lambda: True
    scores = {"truth": 1., "formatted": 1., "rule_reasoning_follow": 1., "rule_answer_follow": 1.,
              "tools_success": 1., "cheating": False}
    evidence = build_execution_evidence(tools_called=1, tools_succeeded=1,
                                        test_results=[{"name": "check", "success": False, "output": "assertion failed"}])
    with patch.object(judge, "_request_chat", return_value={"message": {"content": json.dumps(scores)}}) as request:
        reward = MultiMetricRewardScorer(judge).score("Implement", "", "```python\npass\n```", can_be_verify=False,
                                                     tools_called=1, tools_succeeded=1, execution_evidence=evidence)
    assert reward.truth == 0 and reward.judge_scores["truth"] == 1
    assert "assertion failed" in request.call_args.args[0]["messages"][1]["content"]
