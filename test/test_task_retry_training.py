"""Long-trainer scheduling and reset recovery with controlled inference and storage."""
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sword.checkpoint as checkpoint
import sword.long_trainer as training
from sword.training_logger import TrainingLogger
from sword.reward import MultiMetricRewardScorer
from test.task_retry_fakes import Bridge, Hub, Score, Scorer, StorageModel, StorageTorch


class TestTaskRetryTraining(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.torch, self.hub = StorageTorch(), Hub()
        Bridge.instances, Bridge.answers, Bridge.oom_attempts = [], {}, 0
        patches = [patch.object(training, "torch", self.torch),
                   patch.object(checkpoint, "torch", self.torch),
                   patch.object(training, "GRPOBridge", Bridge),
                   patch.object(training, "GRPOBridgeConfig", lambda **values: SimpleNamespace(**values)),
                   patch.object(training, "MultiMetricRewardScorer", Scorer),
                   patch.object(training, "model_b_config", lambda *args, **kwargs: SimpleNamespace(auto_setup=False)),
                   patch.object(training, "get_harness", lambda name: None),
                   patch.object(training, "harness_repo_policy", lambda **kwargs: nullcontext()),
                   patch.object(training, "rollout_prompt_budget", lambda *args, **kwargs: 10),
                   patch.object(checkpoint, "HAS_HF_HUB", True),
                   patch.object(checkpoint, "HfApi", lambda **kwargs: self.hub, create=True),
                   patch.object(checkpoint, "hf_hub_download", self.hub.download, create=True),
                   patch.object(checkpoint.time, "sleep", lambda seconds: None),
                   patch.dict(sys.modules, {"safetensors.torch": None, "numpy": None})]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.directory.cleanup)

    def run_training(self, dataset=None, *, output="run", steps=10, upload=False, arguments=None, **options):
        if "resume_from_checkpoint" not in options and dataset is None:
            dataset = [{"problem": "A", "solution": "reference", "can_be_verify": False, "category": "math"}]
        args = {"steps": steps, "concurrency_g": 2, "max_seq_len": 16, "max_new_tokens": 1,
                "output_dir": str(self.root / output), **(arguments or {})}
        return training.run_long_train(
            dataset=dataset, model="mock", trainer_model=StorageModel(), tokenizer=SimpleNamespace(),
            b_model=False, auto_code_executor=options.pop("auto_code_executor", False), return_step_metrics=True,
            show_detail_logs=False, is_save_and_upload_checkpoint=upload,
            upload_id="user/run" if upload else None, Argument=args, **options)

    def test_high_rewards_with_wrong_answers_retry_then_wait_without_marking_complete(self):
        result = self.run_training()
        self.assertEqual(result["status"], "waiting_approval")
        self.assertEqual(result["final_step"], 4)
        self.assertEqual(Bridge.instances[-1].prompts, ["A"] * 4)
        summary = result["task_progress"]["summary"]
        self.assertEqual(summary["failed_attempts"], 4)
        self.assertEqual(summary["failed_rollouts"], 8)
        self.assertEqual(summary["solved"], 0)
        self.assertEqual(result["step_metrics"][-1]["task_status"], "waiting_approval")
        self.assertNotIn(result["dataset_sha256"], checkpoint.CheckpointManager(
            output_dir=str(self.root / "run"), is_save_and_upload_checkpoint=False).get_completed_dataset_hashes())

    def test_whole_sandbox_reset_restores_hub_dataset_queue_scores_and_baseline(self):
        dataset = [{"problem": "A", "solution": "reference", "can_be_verify": False, "category": "math"},
                   {"problem": "B", "solution": "reference", "can_be_verify": False, "category": "code"}]
        first = self.run_training(dataset, steps=3, upload=True, arguments={"save_steps": 3})
        self.assertEqual(first["status"], "partial")
        self.assertEqual(Bridge.instances[-1].prompts, ["A", "A", "A"])
        Bridge.answers = {"B": "correct"}
        resumed = self.run_training(None, output="cold", steps=1, upload=True, resume_from_checkpoint="latest")
        self.assertEqual(Bridge.instances[-1].prompts, ["B"])
        self.assertEqual(Bridge.instances[-1].observed_baselines[0], (0.7, 0.25))
        self.assertEqual(resumed["final_step"], 4)
        self.assertEqual(resumed["dataset_sha256"], first["dataset_sha256"])
        self.assertEqual(resumed["task_progress"]["summary"]["failed_attempts"], 3)
        self.assertEqual(resumed["task_progress"]["summary"]["solved"], 1)
        state = checkpoint.CheckpointManager.read_training_state(resumed["checkpoint_dir"])
        self.assertEqual(state["score_count"], 8)
        self.assertEqual(resumed["checkpoint_upload"]["last_uploaded_step"], 4)

    def test_missing_original_local_dataset_uses_checkpoint_snapshot(self):
        source = self.root / "dataset.jsonl"
        source.write_text(json.dumps({"problem": "A", "solution": "reference", "can_be_verify": False}), encoding="utf-8")
        first = self.run_training(str(source), steps=1)
        source.unlink()
        resumed = self.run_training(str(source), steps=1, resume_from_checkpoint="latest")
        self.assertEqual(resumed["dataset_sha256"], first["dataset_sha256"])
        self.assertEqual(resumed["task_progress"]["summary"]["failed_attempts"], 2)

    def test_code_cases_and_whole_row_reach_scoring_and_survive_hub_reset(self):
        raw = {"problem": "Echo stdin", "solution": "print(input())", "answer_rule": "Return Python code.",
               "code_tests": [{"input": "one", "output": "one"}, {"input": "two", "output": "two"}],
               "custom_rubric": {"method": "Check all inputs"}}
        row = training.parse_dataset_row(raw)
        Bridge.answers = {training.policy_task_prompt(row): "'''print(input())'''"}
        seen = []
        class InspectScorer(MultiMetricRewardScorer):
            def score(self, **kwargs):
                seen.append(kwargs)
                return super().score(**kwargs)
        with patch.object(training, "MultiMetricRewardScorer", InspectScorer):
            first = self.run_training([raw], steps=1, upload=True, auto_code_executor=True)
            resumed = self.run_training(None, output="cold-code", steps=1, upload=True,
                                        auto_code_executor=True, resume_from_checkpoint="latest")
        self.assertEqual(first["task_progress"]["summary"]["solved"], 1)
        self.assertEqual(resumed["final_step"], 2)
        self.assertEqual(resumed["dataset_sha256"], first["dataset_sha256"])
        for scored in seen:
            self.assertEqual(scored["dataset_row"]["custom_rubric"], raw["custom_rubric"])
            self.assertEqual([case["stdout"].strip() for case in scored["code_execution"]["cases"]], ["one", "two"],
                             repr(scored["code_execution"]))
            self.assertNotIn("can_be_verify", scored)
        snapshot = checkpoint.CheckpointManager.read_dataset_snapshot(resumed["checkpoint_dir"])
        self.assertEqual(snapshot["records"][0]["code_tests"], row.code_tests)
        self.assertEqual(snapshot["records"][0]["custom_rubric"], raw["custom_rubric"])

    def test_legacy_snapshot_verification_field_is_migrated_without_resetting_queue(self):
        first = self.run_training(steps=1)
        manager = checkpoint.CheckpointManager(output_dir=str(self.root / "run"), is_save_and_upload_checkpoint=False)
        snapshot = manager.read_dataset_snapshot(first["checkpoint_dir"])
        snapshot.pop("schema_version", None)
        snapshot["records"][0]["can_be_verify"] = False
        state = manager.read_training_state(first["checkpoint_dir"])
        manager.save_checkpoint(step=1, model=StorageModel(), training_state=state, dataset_snapshot=snapshot)
        resumed = self.run_training(None, steps=1, resume_from_checkpoint="latest")
        self.assertEqual(resumed["dataset_sha256"], first["dataset_sha256"])
        self.assertEqual(resumed["task_progress"]["summary"]["failed_attempts"], 2)
        migrated = manager.read_dataset_snapshot(resumed["checkpoint_dir"])
        self.assertNotIn("can_be_verify", migrated["records"][0])

    def test_live_approval_restarts_training_and_rejection_exhausts_queue(self):
        logger = TrainingLogger()
        logger._launch = lambda: setattr(logger, "app", object())
        original_publish = logger.publish
        decisions = []
        def publish(event, **data):
            if event == "status" and data.get("status") == "Waiting for task approval":
                task = logger.task_queue.monitor_snapshot()["waitlist"][0]
                action = "approve" if not decisions else "reject"
                decisions.append(action)
                logger.review_task(task["row_hash"], action)
            original_publish(event, **data)
        logger.publish = publish
        with patch("sword.training_logger.create_training_logger", lambda **kwargs: logger.start()):
            result = self.run_training(gradio_logger=True, steps=20)
        logger._worker.join(2)
        self.assertEqual(decisions, ["approve", "reject"])
        self.assertEqual(result["status"], "exhausted")
        self.assertEqual(result["final_step"], 8)
        self.assertEqual(result["task_progress"]["summary"]["failed_attempts"], 8)
        self.assertEqual(result["task_progress"]["summary"]["removed"], 1)

    def test_remove_policy_preserves_unsolved_counts_and_does_not_mark_complete(self):
        result = self.run_training(arguments={"task_failure_action": "remove"})
        self.assertEqual(result["status"], "exhausted")
        self.assertEqual(result["final_step"], 4)
        self.assertEqual(result["task_progress"]["summary"]["removed"], 1)
        self.assertEqual(result["task_progress"]["summary"]["unsolved"], 1)

    def test_oom_retries_do_not_increment_task_failure_or_attempt_counts(self):
        Bridge.oom_attempts = 1
        result = self.run_training(steps=1)
        self.assertEqual(Bridge.instances[-1].train_calls, 2)
        self.assertEqual(result["task_progress"]["summary"]["failed_attempts"], 1)
        self.assertEqual(result["task_progress"]["tasks"][0]["attempts"], 1)
        records = [json.loads(line) for line in Path(result["rollout_details_path"]).read_text().splitlines()]
        self.assertEqual(len(records), 2)

    def test_persistent_oom_checkpoint_keeps_task_at_original_position(self):
        Bridge.oom_attempts = 5
        result = self.run_training(steps=2)
        self.assertEqual(result["status"], "oom_aborted")
        state = checkpoint.CheckpointManager.read_training_state(result["emergency_checkpoint"])
        self.assertEqual(state["step"], 0)
        task = next(iter(state["task_queue"]["tasks"].values()))
        self.assertEqual(task["attempts"], 0)
        self.assertEqual(task["failures"], 0)
        self.assertEqual(state["visited_row_hashes"], [])

    def test_successful_dataset_is_completed_and_skipped_on_next_run(self):
        Bridge.answers = {"A": "correct"}
        first = self.run_training(steps=1)
        self.assertEqual(first["task_progress"]["summary"]["solved"], 1)
        self.assertEqual(self.run_training(steps=1)["status"], "already_trained")

    def test_unknown_resume_checkpoint_is_an_error_instead_of_fresh_training(self):
        with self.assertRaises(FileNotFoundError):
            self.run_training(resume_from_checkpoint="latest")

    def test_changed_dataset_on_resume_is_rejected(self):
        self.run_training(steps=1)
        with self.assertRaises(ValueError):
            self.run_training([{"problem": "Changed A", "solution": "reference"}],
                              steps=1, resume_from_checkpoint="latest")

    def test_correctness_requires_completed_answer_and_observed_dataset_tests(self):
        row = training.parse_dataset_row({"problem": "A", "tests": "assert True", "can_be_verify": False})
        score = Score(truth=1.0)
        self.assertFalse(training.task_rollout_solved(row, "correct", score))
        completed = SimpleNamespace(is_completed=True, test_results=[{"success": True}])
        self.assertTrue(training.task_rollout_solved(row, "correct", score, completed))
        completed.is_completed = False
        self.assertFalse(training.task_rollout_solved(row, "correct", score, completed))
        completed.is_completed = True
        completed.test_results[0]["success"] = False
        self.assertFalse(training.task_rollout_solved(row, "correct", score, completed))
        self.assertFalse(training.task_rollout_solved(row, "", score, completed))

    def test_math_correctness_uses_the_full_judge_truth_score(self):
        row = training.parse_dataset_row({"problem": "Compute 2+2", "solution": "4"})
        self.assertTrue(training.task_rollout_solved(row, "4", Score(truth=1.0)))
        self.assertFalse(training.task_rollout_solved(row, "4", Score(truth=0.5)))
        self.assertFalse(training.task_rollout_solved(row, "5", Score(truth=0.5)))
        self.assertFalse(training.task_rollout_solved(row, "4", Score(truth=0.5, cheating=True)))
        self.assertFalse(training.task_rollout_solved(row, "4", Score(truth=0.5, judge_source="primary")))

    def test_random_state_is_restored_after_engine_initialization(self):
        self.run_training(steps=1)
        generate = Bridge.generate_rollouts
        def observe(bridge, prompts):
            self.assertEqual(self.torch.rng, [1, 2, 3])
            return generate(bridge, prompts)
        with patch.object(Bridge, "initialize_vllm", lambda bridge: setattr(self.torch, "rng", [42])), \
             patch.object(Bridge, "generate_rollouts", observe):
            self.run_training(None, steps=1, resume_from_checkpoint="latest")


if __name__ == "__main__":
    unittest.main()
