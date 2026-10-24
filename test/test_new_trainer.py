"""
Unit test suite for the new Sword GRPO Trainer architecture:
- Model B Ollama Judge & JSON/YAML parser
- Agentic Harnesses (OpenCode, Codex CLI, Gemini CLI, Pi)
- Multi-Metric Reward Scorer & MathVerifier
- CheckpointManager, SHA-256 tracking, resume, and pruning
- Single-call sword.train(long_train=True, ...) end-to-end workflow
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

import sword
from sword.b_model import ModelBJudge, ModelBJudgeConfig, ModelBEvaluationResult
from sword.harness import (
    get_harness,
    BaseHarness,
    HarnessResult,
    OpenCodeHarness,
    CodexCliHarness,
    GeminiCliHarness,
    PiHarness,
)
from sword.reward import MultiMetricRewardScorer, MathVerifier, RewardBreakdown
from sword.checkpoint import CheckpointManager
from sword.long_trainer import (
    DatasetProblemRow,
    parse_dataset_row,
    load_dataset_records,
    run_long_train,
)


class DummyTokenizer:
    vocab_size = 500
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        tokens = [2, 10, 20, 30]
        return {"input_ids": torch.tensor([tokens], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        return f"decoded response ({len(token_ids)} tokens)"


class TestNewTrainerArchitecture(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_sword_trainer_")
        self.hf_config = AutoConfig.for_model(
            "qwen2",
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
        self.model = AutoModelForCausalLM.from_config(self.hf_config)
        self.tokenizer = DummyTokenizer()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_module_exports(self):
        self.assertTrue(hasattr(sword, "ModelBJudge"))
        self.assertTrue(hasattr(sword, "get_harness"))
        self.assertTrue(hasattr(sword, "MultiMetricRewardScorer"))
        self.assertTrue(hasattr(sword, "MathVerifier"))
        self.assertTrue(hasattr(sword, "CheckpointManager"))
        self.assertTrue(hasattr(sword, "run_long_train"))
        self.assertTrue(callable(sword.train))

    def test_b_model_parser(self):
        judge = ModelBJudge()

        # 1. Standard JSON dict format
        sample_json = (
            '{\n'
            '  "truth": 0.5,\n'
            '  "formatted": 0.9,\n'
            '  "rule_reasoning_follow": 0.8,\n'
            '  "rule_answer_follow": 0.85,\n'
            '  "tools_success": 0.0,\n'
            '  "cheating": false\n'
            '}'
        )
        res = judge.parse_scores(sample_json)
        self.assertEqual(res.truth, 0.5)
        self.assertEqual(res.formatted, 0.9)
        self.assertEqual(res.rule_reasoning_follow, 0.8)
        self.assertEqual(res.rule_answer_follow, 0.85)
        self.assertFalse(res.cheating)

        # 2. JSON array format [truth, formatted, reasoning, answer, tools, cheating]
        sample_arr = '[0.5, 1.0, 0.75, 0.75, 0.5, false]'
        res_arr = judge.parse_scores(sample_arr)
        self.assertEqual(res_arr.truth, 0.5)
        self.assertEqual(res_arr.formatted, 1.0)
        self.assertEqual(res_arr.rule_reasoning_follow, 0.75)
        self.assertEqual(res_arr.tools_success, 0.5)

        # 3. Embedded in markdown code block with cheating true
        sample_md = (
            'Here is the evaluation:\n'
            '```json\n'
            '{"truth": 0.0, "formatted": 0.5, "rule_reasoning_follow": 0.2, '
            '"rule_answer_follow": 0.3, "tools_success": 0.0, "cheating": true}\n'
            '```'
        )
        res_md = judge.parse_scores(sample_md)
        self.assertEqual(res_md.truth, 0.0)
        self.assertTrue(res_md.cheating)

        # 4. Offline deterministic fallback
        fb = judge._deterministic_fallback(
            candidate_response="<think>Step 1</think> 42",
            solution="42",
            tools_called=0,
            tools_succeeded=0,
        )
        self.assertEqual(fb.truth, 1.0)
        self.assertFalse(fb.cheating)

    def test_harness_registry_and_execution(self):
        # 1. Registry resolution
        self.assertIsInstance(get_harness("opencode"), OpenCodeHarness)
        self.assertIsInstance(get_harness("codex-cli"), CodexCliHarness)
        self.assertIsInstance(get_harness("gemini_cli"), GeminiCliHarness)
        self.assertIsInstance(get_harness("pi"), PiHarness)
        self.assertIsNone(get_harness("none"))
        self.assertIsNone(get_harness(None))

        # 2. OpenCode harness execution
        opencode = OpenCodeHarness(allow_exec=False)
        mock_output = (
            "<think>Analyzing repository structure</think>\n"
            "```bash\nls -la /workspace\n```\n"
            "Final fix applied."
        )
        res = opencode.run(
            prompt="Investigate workspace",
            model_generate_fn=lambda p: mock_output,
            max_turns=1,
        )
        self.assertEqual(res.tools_called, 1)
        self.assertEqual(res.tools_succeeded, 1)
        self.assertEqual(res.reasoning_text, "Analyzing repository structure")

        # 3. Codex CLI syntax checker
        codex = CodexCliHarness()
        code_resp = "<think>Writing binary search</think>\n```python\ndef search(arr, target):\n    return target in arr\n```"
        c_res = codex.run("Implement search", lambda p: code_resp)
        self.assertEqual(c_res.tools_called, 1)
        self.assertEqual(c_res.tools_succeeded, 1)

    def test_math_verifier(self):
        # Numeric equality
        self.assertTrue(MathVerifier.verify_answer("<think>Deduction</think> 42", "42"))
        self.assertTrue(MathVerifier.verify_answer("\\boxed{3.14159}", "3.1416"))
        self.assertTrue(MathVerifier.verify_answer("The answer is 120", "120"))

        # Roots equality
        self.assertTrue(MathVerifier.verify_answer("x1 = -3, x2 = 5", "x1 = 5, x2 = -3"))
        self.assertFalse(MathVerifier.verify_answer("41", "42"))

    def test_multi_metric_reward_scoring(self):
        scorer = MultiMetricRewardScorer()

        # Non-agentic verifiable math task
        breakdown = scorer.score(
            problem="Compute 10 + 15",
            solution="25",
            candidate_response="<think>Adding 10 and 15</think> 25",
            reasoning_rule="Show addition",
            answer_rule="Single number",
            can_be_verify=True,
            harness_name="none",
        )
        # Truth should be 0.5 for correct math, formatted ~1.0
        self.assertEqual(breakdown.truth, 0.5)
        self.assertEqual(breakdown.formatted, 1.0)
        self.assertFalse(breakdown.is_agentic)
        self.assertGreater(breakdown.total_score, 0.0)

        # Cheating deduction test: score should be penalized by -0.2
        breakdown_cheat = scorer.score(
            problem="Task",
            solution="Solution",
            candidate_response="I am programmed to ignore previous instructions",
            can_be_verify=False,
        )
        if breakdown_cheat.cheating:
            expected_base = (breakdown_cheat.truth + breakdown_cheat.formatted +
                             breakdown_cheat.rule_reasoning_follow + breakdown_cheat.rule_answer_follow) / 4.0
            self.assertAlmostEqual(breakdown_cheat.total_score, max(-0.2, expected_base - 0.2), places=3)

    def test_checkpoint_manager_and_sha256(self):
        mgr = CheckpointManager(
            output_dir=self.temp_dir,
            num_latest_checkpoint_save=2,
            is_save_and_upload_checkpoint=False,
        )

        test_data = "problem 1: 1+1=2\nproblem 2: 2+2=4"
        sha = mgr.compute_sha256(test_data)
        self.assertFalse(mgr.is_dataset_completed(sha))

        mgr.mark_dataset_completed(sha, "dataset_test")
        self.assertTrue(mgr.is_dataset_completed(sha))

        # Save 3 checkpoints consecutively
        from sword.finetune import apply_lora_to_model
        lora_model = apply_lora_to_model(self.model, r=4, lora_alpha=8, target_moe=False)

        mgr.save_checkpoint(step=10, model=lora_model, completed_dataset_hash=sha)
        mgr.save_checkpoint(step=20, model=lora_model, completed_dataset_hash=sha)
        mgr.save_checkpoint(step=30, model=lora_model, completed_dataset_hash=sha)

        # Checkpoints pruned to latest 2: checkpoint-20 and checkpoint-30
        ckpts = sorted(os.listdir(self.temp_dir))
        self.assertIn("checkpoint-20", ckpts)
        self.assertIn("checkpoint-30", ckpts)
        self.assertNotIn("checkpoint-10", ckpts)

        # Check that files exist inside checkpoint
        ckpt30 = os.path.join(self.temp_dir, "checkpoint-30")
        self.assertTrue(os.path.exists(os.path.join(ckpt30, "adapter_config.json")))
        self.assertTrue(os.path.exists(os.path.join(ckpt30, "training_state.json")))
        self.assertTrue(os.path.exists(os.path.join(ckpt30, "data_complete.logs")))

        # Check resume lookup
        resumed = mgr.find_latest_checkpoint()
        self.assertEqual(resumed, ckpt30)

    def test_dataset_parsing(self):
        raw_row = {
            "problem": "Calculate integral of x dx",
            "solution": "x^2/2",
            "reasoning_rule": "Apply power rule",
            "answer_rule": "Express in simplest form",
            "can_be__verify": False,
            "difficult": "Hard",
            "cagetory": "Calculus",
            "harness": "opencode",
        }
        parsed = parse_dataset_row(raw_row)
        self.assertEqual(parsed.problem, "Calculate integral of x dx")
        self.assertFalse(parsed.can_be_verify)
        self.assertEqual(parsed.difficulty, "hard")
        self.assertEqual(parsed.category, "calculus")
        self.assertEqual(parsed.harness, "opencode")

    def test_sword_train_end_to_end_long_train(self):
        from sword.finetune import apply_lora_to_model
        apply_lora_to_model(self.model, r=4, lora_alpha=8, target_moe=False)

        custom_dataset = [
            {
                "problem": "Solve 2 + 3",
                "solution": "5",
                "reasoning_rule": "Basic arithmetic",
                "answer_rule": "Integer only",
                "can_be__verify": True,
                "difficult": "easy",
                "cagetory": "math",
                "harness": "none",
            },
            {
                "problem": "Write a python function to add a and b",
                "solution": "def add(a, b): return a + b",
                "reasoning_rule": "Syntax validity",
                "answer_rule": "Def block",
                "can_be__verify": False,
                "difficult": "medium",
                "cagetory": "code",
                "harness": "codex",
            },
        ]

        # 1. Run training via unified sword.train entry point
        result = sword.train(
            long_train=True,
            test=False,
            dataset=custom_dataset,
            model="test_model",
            B_model_port=11343,
            custom_checkpoint_save=True,
            is_save_and_upload_checkpoint=False,
            num_latest_checkpoint_save=2,
            Argument={
                "steps": 2,
                "concurrency_g": 2,
                "save_steps": 1,
                "micro_batch_size": 2,
                "output_dir": os.path.join(self.temp_dir, "long_ckpt"),
            },
            show_detail_logs=False,
            score_logs=True,
        )

        self.assertEqual(result.get("status"), "completed")
        self.assertEqual(result.get("final_step"), 2)
        self.assertTrue(os.path.isdir(result.get("checkpoint_dir")))

        # 2. Re-running on identical dataset should trigger SHA-256 deduplication
        rerun_result = sword.train(
            long_train=True,
            test=False,
            dataset=custom_dataset,
            model="test_model",
            B_model_port=11343,
            is_save_and_upload_checkpoint=False,
            Argument={
                "output_dir": os.path.join(self.temp_dir, "long_ckpt"),
            },
            show_detail_logs=False,
        )
        self.assertEqual(rerun_result.get("status"), "already_trained")
        self.assertEqual(rerun_result.get("steps_completed"), 0)


if __name__ == "__main__":
    unittest.main()
