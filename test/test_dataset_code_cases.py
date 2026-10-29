"""Dataset I/O cases, final-source execution, and complete judge evidence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.execution import candidate_source, execute_coding_answer, has_candidate_code, normalize_code_tests, run_dataset_tests
from sword.execution_evidence import build_execution_evidence
from sword.harness.base import HarnessResult
from sword.harness.sandbox import WorkspaceSandbox
from sword.long_trainer import load_dataset_records, parse_dataset_row, policy_task_prompt, restore_dataset_row, task_rollout_solved
from sword.reward import MultiMetricRewardScorer
from sword.training_logger import TrainingLogger


class TestDatasetCodeCases(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def run_code(self, source, cases, **fields):
        row = parse_dataset_row({"problem": "Compute the requested result from stdin.", "code_tests": cases, **fields})
        result = execute_coding_answer(row, source, parent_dir=str(self.root))
        return row, result

    def test_merged_cases_preserve_zero_false_empty_and_nested_values(self):
        cases = [{"input": 0, "output": 0}, {"input": False, "output": False},
                 {"input": "", "output": ""}, {"input": [1, 2], "output": {"sum": 3}}]
        row = parse_dataset_row({"problem": "TASK", "code_tests": json.dumps(cases)})
        self.assertEqual([(case["input"], case["output"]) for case in row.code_tests],
                         [(case["input"], case["output"]) for case in cases])
        self.assertEqual(parse_dataset_row(row.to_dict()).row_hash, row.row_hash)
        self.assertNotIn("code_input", row.to_dict())
        self.assertNotIn("output_expect", row.to_dict())

    def test_invalid_or_duplicate_case_pairs_fail_without_silently_losing_cases(self):
        for invalid in ({"input": "x", "output": "x"}, [{}], [{"input": "x"}],
                        [{"input": "x", "output": "x", "timeout_sec": 0}],
                        '[{"input":"a","output":"a","input":"b","output":"b"}]'):
            with self.assertRaises(ValueError):
                normalize_code_tests(invalid)

    def test_local_json_and_csv_keep_all_cases_and_reject_duplicate_json_keys(self):
        import csv
        raw = {"problem": "TASK", "code_tests": [{"input": "a\n", "output": "a"}, {"input": "b", "output": "b"}]}
        for extension in ("json", "jsonl", "csv", "tsv"):
            path = self.root / ("cases." + extension)
            with path.open("w", encoding="utf-8", newline="") as stream:
                if extension in ("csv", "tsv"):
                    writer = csv.DictWriter(stream, fieldnames=list(raw), delimiter="\t" if extension == "tsv" else ",")
                    writer.writeheader()
                    writer.writerow({**raw, "code_tests": json.dumps(raw["code_tests"])})
                else:
                    json.dump(raw, stream)
            rows, _ = load_dataset_records(str(path))
            self.assertEqual(len(rows[0].code_tests), 2)
        duplicate = '{"problem":"TASK","code_tests":[{"input":"a","output":"a","input":"b","output":"b"}]}'
        for extension in ("json", "jsonl"):
            path = self.root / ("duplicate." + extension)
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset_records(str(path))

    def test_whole_row_metadata_survives_and_legacy_verification_flags_have_no_effect(self):
        raw = {"question": "TASK", "answer": "REFERENCE", "id": 0, "answer_field": "Assess every code case.",
               "custom_rubric": {"weight": 3, "note": "PRIVATE_GUIDANCE"}}
        row = parse_dataset_row({**raw, "can_be_verify": False, "can_be__verify": True})
        self.assertEqual(row.problem_id, "0")
        self.assertEqual(row.answer_rule, "Assess every code case.")
        self.assertEqual(row.to_dict()["custom_rubric"], raw["custom_rubric"])
        self.assertEqual(row.to_dict()["question"], "TASK")
        self.assertNotIn("can_be_verify", row.to_dict())
        self.assertNotIn("can_be__verify", row.to_dict())
        self.assertEqual(row.row_hash, parse_dataset_row(raw).row_hash)
        self.assertEqual(parse_dataset_row(row.to_dict()).row_hash, row.row_hash)
        self.assertFalse(hasattr(row, "can_be_verify"))

    def test_changed_cases_or_judge_metadata_change_dataset_identity(self):
        raw = {"problem": "TASK", "code_tests": [{"input": 1, "output": 2}], "rubric": "one"}
        first = load_dataset_records([raw])[1]
        self.assertTrue(first != load_dataset_records([{**raw, "code_tests": [{"input": 1, "output": 3}]}])[1])
        self.assertTrue(first != load_dataset_records([{**raw, "rubric": "two"}])[1])

    def test_policy_never_receives_reference_or_expected_outputs_or_private_metadata(self):
        row = parse_dataset_row({"problem": "TASK", "solution": "PRIVATE_REFERENCE", "answer_rule": "Return code.",
                                 "code_tests": [{"input": "PRIVATE_INPUT", "output": "PRIVATE_EXPECTED"}],
                                 "rubric": "PRIVATE_RUBRIC"})
        prompt = policy_task_prompt(row)
        self.assertIn("Return code.", prompt)
        for value in ("PRIVATE_REFERENCE", "PRIVATE_INPUT", "PRIVATE_EXPECTED", "PRIVATE_RUBRIC"):
            self.assertNotIn(value, prompt)

    def test_quotes_markdown_and_language_tags_extract_only_final_visible_code(self):
        hidden = "<think>'''print('draft')'''</think>\n"
        response = hidden + "'''print('example')'''\nFinal Answer:\n'''python\nprint(input())\n'''"
        self.assertEqual(candidate_source(response), ("print(input())", "python"))
        self.assertTrue(has_candidate_code(response))
        self.assertFalse(has_candidate_code(hidden))
        self.assertFalse(has_candidate_code("'''An ordinary explanation in words'''"))
        self.assertFalse(has_candidate_code("'''print('unfinished' '''"))
        markdown = "```python\ndef echo(value):\n    '''A function docstring.'''\n    return value\n```"
        self.assertIn("A function docstring.", candidate_source(markdown)[0])
        self.assertEqual(candidate_source("'''js\nconsole.log('ok');\n'''"), ("console.log('ok');", "javascript"))

    def test_two_quoted_cases_run_with_input_without_an_extra_empty_input_execution(self):
        source = "'''a, b = map(int, input().split())\nprint(a + b)'''"
        row, result = self.run_code(source, [{"input": "2 3\n", "output": 5}, {"input": "4 6\n", "output": 10}])
        self.assertEqual(result.tools_called, 2)
        self.assertEqual(result.tools_succeeded, 2)
        self.assertEqual(len(result.execution_results), 2)
        self.assertEqual([case["stdout"].strip() for case in result.test_results], ["5", "10"])
        self.assertTrue(all(case["success"] for case in result.test_results))
        self.assertEqual(result.code_execution["source"], "a, b = map(int, input().split())\nprint(a + b)")
        self.assertEqual(result.code_execution["cases"], result.test_results)
        self.assertTrue(result.workspace_removed)
        self.assertFalse(Path(result.workspace_dir).exists())
        self.assertTrue(task_rollout_solved(row, source, SimpleNamespace(truth=1, cheating=False), result))

    def test_json_input_values_are_sent_as_json_stream_contents(self):
        source = "'''import json, sys\nprint(json.dumps(json.loads(sys.stdin.read()), separators=(',', ':')))'''"
        _, result = self.run_code(source, [{"input": 0, "output": 0}, {"input": False, "output": False},
                                          {"input": [1, 2], "output": [1, 2]}])
        self.assertEqual([case["stdin"] for case in result.test_results], ["0", "false", "[1,2]"])
        self.assertTrue(all(case["success"] for case in result.test_results))

    def test_empty_input_and_output_are_valid_cases(self):
        _, result = self.run_code("'''import sys\nsys.stdout.write(sys.stdin.read())'''", [{"input": "", "output": ""}])
        self.assertTrue(result.test_results[0]["success"])
        self.assertEqual(result.test_results[0]["stdin"], "")
        self.assertEqual(result.test_results[0]["stdout"], "")

    def test_unicode_input_and_output_and_indented_quoted_code(self):
        source = "'''\n    line = input()\n    print(line)\n'''"
        self.assertTrue(has_candidate_code(source))
        _, result = self.run_code(source, [{"input": "Tiếng Việt 😄", "output": "Tiếng Việt 😄"}])
        self.assertTrue(result.test_results[0]["success"])
        self.assertEqual(result.test_results[0]["stdout"].strip(), "Tiếng Việt 😄")

    def test_answer_labels_inside_code_strings_do_not_truncate_the_source(self):
        source = 'print("""\nFinal Answer:\nhello\n""")'
        for response in ("'''" + source + "'''", source):
            self.assertEqual(candidate_source(response)[0], source)
            _, result = self.run_code(response, [{"input": "", "output": "\nFinal Answer:\nhello\n"}])
            self.assertTrue(result.test_results[0]["success"])

    def test_empty_or_wrong_language_blocks_cannot_receive_credit_from_seed_code(self):
        for response in ("'''\n'''", "```js\nconsole.log('different');\n```"):
            _, result = self.run_code(response, [{"input": "", "output": "seed"}], language="python",
                                      initial_files={"solution.py": "print('seed')"})
            self.assertFalse(result.test_results[0]["executed"])
            self.assertFalse(result.test_results[0]["success"])

    def test_mismatch_is_distinct_from_process_failure(self):
        row, result = self.run_code("'''print(int(input()) + 1)'''",
                                    [{"input": 1, "output": 2}, {"input": 2, "output": 99}])
        second = result.test_results[1]
        self.assertTrue(second["process_success"])
        self.assertEqual(second["exit_code"], 0)
        self.assertFalse(second["output_match"])
        self.assertFalse(second["success"])
        evidence = build_execution_evidence(test_results=result.test_results)
        score = MultiMetricRewardScorer(None).score(row.problem, "", result.final_text,
                                                   execution_evidence=evidence, dataset_row=row.to_dict(),
                                                   code_execution=result.code_execution)
        self.assertEqual(score.truth, 0.5)
        self.assertFalse(task_rollout_solved(row, result.final_text, score, result))

    def test_syntax_runtime_and_missing_source_errors_are_recorded(self):
        for source in ("'''print('broken' '''", "'''raise ValueError('RUNNER_ERROR')'''", ""):
            _, result = self.run_code(source, [{"input": "x", "output": ""}])
            case = result.test_results[0]
            self.assertFalse(case["process_success"])
            self.assertFalse(case["success"])
            self.assertTrue(case["stderr"])
        self.assertFalse(result.test_results[0]["executed"])
        self.assertFalse(result.test_results[0]["output_match"])

    def test_matching_stdout_still_fails_when_process_exits_with_error(self):
        _, result = self.run_code("'''import sys\nprint('right output')\nsys.exit(7)'''",
                                  [{"input": "", "output": "right output"}])
        case = result.test_results[0]
        self.assertTrue(case["output_match"])
        self.assertEqual(case["exit_code"], 7)
        self.assertFalse(case["success"])

    def test_timeout_and_per_case_timeout_restore_are_preserved(self):
        sandbox = WorkspaceSandbox(parent_dir=str(self.root), timeout_sec=17)
        def timeout(command, *, stdin=None):
            self.assertEqual(sandbox.timeout_sec, 0.1)
            self.assertEqual(stdin, "input")
            sandbox.last_execution_result = {"exit_code": None, "stdout": "", "stderr": "timeout",
                                             "timed_out": True, "success": False}
            return False, "Command timed out"
        with patch.object(sandbox, "run_command", timeout):
            results = run_dataset_tests(sandbox, [], "'''print(input())'''",
                                        code_tests=normalize_code_tests([{"input": "input", "output": "expected", "timeout_sec": 0.1}]))
        self.assertEqual(sandbox.timeout_sec, 17)
        self.assertTrue(results[0]["timed_out"])
        self.assertFalse(results[0]["success"])
        sandbox.close()

    def test_comparison_preserves_spaces_but_normalizes_line_endings(self):
        _, result = self.run_code("'''import sys\nsys.stdout.write(' a \\n')'''",
                                  [{"input": "", "output": " a \r\n"}, {"input": "", "output": "a"}])
        self.assertEqual([case["success"] for case in result.test_results], [True, False])

    def test_large_stdout_compares_before_capture_truncation(self):
        _, result = self.run_code("'''print('x' * 9000)'''", [{"input": "", "output": "x" * 9000}])
        case = result.test_results[0]
        self.assertTrue(case["success"])
        self.assertTrue(case["stdout_truncated"])
        self.assertEqual(len(case["stdout"]), 8192)
        self.assertEqual(case["stdout_char_count"], 9001)

    def test_each_case_restarts_the_program_and_restores_its_original_source(self):
        source = "'''from pathlib import Path\nPath('solution.py').write_text('raise RuntimeError()')\nprint(input())'''"
        _, result = self.run_code(source, [{"input": "first", "output": "first"}, {"input": "second", "output": "second"}])
        self.assertTrue(all(case["success"] for case in result.test_results))
        self.assertIn("write_text", result.code_execution["source"])

    def test_function_assertions_still_accept_quoted_source(self):
        _, result = self.run_code("'''def add(a, b):\n    return a + b'''", [],
                                  tests="assert add(2, 3) == 5", category="coding")
        self.assertTrue(result.test_results[0]["success"])
        self.assertEqual(result.tools_called, 1)

    def test_all_harnesses_run_merged_cases_before_cleanup_and_return_the_source(self):
        from sword.harness.opencode import OpenCodeHarness
        from sword.harness.codex_cli import CodexCliHarness
        from sword.harness.gemini_cli import GeminiCliHarness
        from sword.harness.pi_agent import PiHarness
        cases = normalize_code_tests([{"input": "a", "output": "a"}, {"input": "b", "output": "b"}])
        for cls in (OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness):
            harness = cls()
            harness.workspace_parent_dir = str(self.root)
            harness.code_tests = cases
            result = harness.run("TASK", lambda _: "Final Answer:\n'''print(input())'''")
            self.assertTrue(result.is_completed)
            self.assertTrue(all(case["success"] for case in result.test_results))
            self.assertEqual(len(result.code_execution["cases"]), 2)
            self.assertEqual(result.code_execution["source"], "print(input())")
            self.assertTrue(result.workspace_removed)

    def test_full_row_and_all_case_results_reach_actual_judge_request_with_answer_guidance(self):
        row, result = self.run_code("'''print(input())'''", [{"input": str(i), "output": str(i)} for i in range(8)],
                                    solution="PRIVATE_REFERENCE", answer_rule="ASSESS_ALL_CASES",
                                    custom_rubric={"note": "PRIVATE_METADATA"})
        evidence = build_execution_evidence(test_results=result.test_results, max_chars=512, max_success=0, max_failed=0)
        self.assertEqual(evidence["samples"], [])
        judge = ModelBJudge(ModelBJudgeConfig(auto_setup=False, reasoning=False, force_gpu=False,
                                             fallback_model_name=None, log_model_outputs=False, execution_evidence_chars=512))
        judge.ensure_ready = lambda: True
        requests = []
        scores = {"truth": 1.0, "formatted": 1.0, "rule_reasoning_follow": 0.9,
                  "rule_answer_follow": 0.75, "tools_success": 1.0, "cheating": False}
        def request(payload):
            requests.append(payload)
            return {"message": {"content": json.dumps(scores)}}
        with patch.object(judge, "_request_chat", request):
            score = MultiMetricRewardScorer(judge).score(row.problem, row.solution, result.final_text,
                                                       answer_rule=row.answer_rule, dataset_row=row.to_dict(),
                                                       code_execution=result.code_execution, execution_evidence=evidence)
        prompt = requests[0]["messages"][1]["content"]
        self.assertIn(json.dumps(row.to_dict(), ensure_ascii=False), prompt)
        self.assertIn(json.dumps(result.code_execution, ensure_ascii=False), prompt)
        self.assertIn("stdin and produced the stdout", prompt)
        self.assertIn("ASSESS_ALL_CASES", prompt)
        self.assertIn("PRIVATE_METADATA", prompt)
        self.assertEqual(score.rule_answer_follow, 0.75)
        self.assertEqual(score.truth, 1.0)
        self.assertEqual(set(requests[0]["format"]["required"]), set(scores))

    def test_configured_but_unexecuted_cases_receive_no_correctness_credit(self):
        row = parse_dataset_row({"problem": "TASK", "code_tests": [{"input": "x", "output": "x"}]})
        score = MultiMetricRewardScorer(None).score("TASK", "", "'''print(input())'''", dataset_row=row.to_dict())
        self.assertEqual(score.truth, 0)
        self.assertFalse(task_rollout_solved(row, "code", SimpleNamespace(truth=1, cheating=False), HarnessResult()))
        other_tests = HarnessResult(test_results=[{"success": True}])
        self.assertFalse(task_rollout_solved(row, "code", SimpleNamespace(truth=1, cheating=False), other_tests))

    def test_math_judge_truth_is_not_controlled_by_legacy_flag_or_exact_number_match(self):
        seen = []
        judge = SimpleNamespace(last_evaluation_source="controlled")
        def evaluate(**kwargs):
            seen.append(kwargs)
            from sword.b_model import ModelBEvaluationResult
            return ModelBEvaluationResult(truth=0.8, rule_answer_follow=0.6)
        judge.evaluate = evaluate
        for legacy in (True, False, None):
            score = MultiMetricRewardScorer(judge).score("2+2", "4", "Final Answer: four",
                                                       can_be_verify=legacy, category="math")
            self.assertEqual(score.truth, 0.8)
            self.assertEqual(score.rule_answer_follow, 0.6)
            self.assertNotIn("can_be_verify", seen[-1])
        self.assertEqual(MultiMetricRewardScorer(None).score("2+2", "4", "Final Answer: 4").truth, 1)
        self.assertEqual(MultiMetricRewardScorer(None).score("2+2", "4", "<think>4</think>").truth, 0)

    def test_old_checkpoint_row_migration_keeps_queue_identity_and_drops_removed_flag(self):
        saved = {"problem": "TASK", "solution": "4", "problem_id": "legacy-task", "row_hash": "legacy-row-hash",
                 "can_be_verify": True, "effort": "medium"}
        row = restore_dataset_row(saved)
        self.assertEqual(row.row_hash, "legacy-row-hash")
        self.assertEqual(row.problem_id, "legacy-task")
        self.assertEqual(row.effort, "medium")
        self.assertNotIn("can_be_verify", row.to_dict())

    def test_gradio_score_state_retains_complete_code_case_results(self):
        _, result = self.run_code("'''print(input())'''", [{"input": "x", "output": "x"}])
        logger = TrainingLogger()
        logger._apply("score", {"rollout": 0, "scores": {"truth": 1}, "code_execution": result.code_execution})
        self.assertEqual(logger.snapshot()["scores"][0]["code_execution"], result.code_execution)


if __name__ == "__main__":
    unittest.main()
