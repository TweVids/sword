"""
Unit test suite for the new Sword GRPO Trainer architecture:
- Model B Ollama Judge & JSON/YAML parser with VRAM lifecycle management
- Authentic Agentic Harnesses (OpenCode, Codex CLI, Gemini CLI, Pi) with real WorkspaceSandbox
- Multi-Metric Reward Scorer & MathVerifier
- CheckpointManager, SHA-256 tracking, read-back verification, resume, and pruning
- OOM auto-recovery and emergency checkpointing without false completion logging
- Expandable dataset deduplication with per-row SHA-256
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
    WorkspaceSandbox,
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

    def __call__(self, text, return_tensors="pt", add_special_tokens=True, **kwargs):
        tokens = [2, 10, 20, 30]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([tokens], dtype=torch.long)}
        return {"input_ids": tokens}

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
        self.assertTrue(hasattr(sword, "WorkspaceSandbox"))
        self.assertTrue(hasattr(sword, "MultiMetricRewardScorer"))
        self.assertTrue(hasattr(sword, "MathVerifier"))
        self.assertTrue(hasattr(sword, "CheckpointManager"))
        self.assertTrue(hasattr(sword, "run_long_train"))
        self.assertTrue(callable(sword.train))

    def test_real_workspace_sandbox(self):
        sandbox = WorkspaceSandbox(cleanup=True)
        try:
            # 1. Write file
            ok_w, msg_w = sandbox.write_file("test.py", "x = 10\nprint('sandbox_result:', x * 2)")
            self.assertTrue(ok_w)

            # 2. Read file
            ok_r, content_r = sandbox.read_file("test.py")
            self.assertTrue(ok_r)
            self.assertIn("sandbox_result", content_r)

            # 3. Edit file
            ok_e, msg_e = sandbox.edit_file("test.py", "x * 2", "x * 5")
            self.assertTrue(ok_e)
            _, edited_c = sandbox.read_file("test.py")
            self.assertIn("x * 5", edited_c)

            # 4. Real python execution
            ok_py, out_py = sandbox.run_python("print(40 + 2)")
            self.assertTrue(ok_py)
            self.assertEqual(out_py.strip(), "42")

            # 5. Dangerous path escape rejected
            ok_bad, msg_bad = sandbox.write_file("../../escaped.txt", "hacked")
            self.assertFalse(ok_bad)
            self.assertIn("denied", msg_bad.lower())
            with self.assertRaises(PermissionError):
                sandbox._resolve_safe_path("../../escaped.txt")
        finally:
            sandbox.close()

    def test_b_model_parser_and_lifecycle(self):
        config = ModelBJudgeConfig(num_predict=1024, auto_unload=True)
        self.assertEqual(config.num_predict, 1024)
        self.assertTrue(config.auto_unload)
        judge = ModelBJudge(config=config)

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

        # 4. Memory lifecycle and guard
        self.assertTrue(judge.check_memory_guard())
        judge.prepare_for_eval()
        judge.release_after_eval()
        self.assertFalse(judge.unload())

    def test_harness_registry_and_real_execution(self):
        # 1. Registry resolution
        self.assertIsInstance(get_harness("opencode"), OpenCodeHarness)
        self.assertIsInstance(get_harness("codex-cli"), CodexCliHarness)
        self.assertIsInstance(get_harness("gemini_cli"), GeminiCliHarness)
        self.assertIsInstance(get_harness("pi"), PiHarness)
        self.assertIsNone(get_harness("none"))
        self.assertIsNone(get_harness(None))

        # 2. Real OpenCode harness execution with genuine subprocess python execution
        opencode = OpenCodeHarness()
        mock_output = (
            "<think>Analyzing repository structure</think>\n"
            "```python\ndef solution():\n    return 42\nprint('computed:', solution())\n```\n"
            "Final fix applied."
        )
        res = opencode.run(
            prompt="Investigate workspace",
            model_generate_fn=lambda p: mock_output,
            max_turns=1,
        )
        self.assertEqual(res.tools_called, 1)
        self.assertEqual(res.tools_succeeded, 1)
        self.assertEqual(res.tools_failed, 0)
        self.assertEqual(res.reasoning_text, "Analyzing repository structure")

        # 3. Real OpenCode error detection on broken script
        broken_output = "```python\nraise ValueError('real failure')\n```"
        res_broken = opencode.run("Run code", lambda p: broken_output, max_turns=1)
        self.assertEqual(res_broken.tools_called, 1)
        self.assertEqual(res_broken.tools_succeeded, 0)
        self.assertEqual(res_broken.tools_failed, 1)

    def test_codex_cli_real_multiturn_self_repair(self):
        codex = CodexCliHarness()
        turns = [
            # Turn 1: syntax error
            "```python\ndef broken(\n```",
            # Turn 2: corrected code that executes successfully
            "```python\ndef fixed():\n    return 42\nassert fixed() == 42\n```",
        ]
        call_count = [0]

        def step_generator(p: str) -> str:
            resp = turns[min(call_count[0], len(turns) - 1)]
            call_count[0] += 1
            return resp

        res = codex.run("Write code", model_generate_fn=step_generator, max_turns=2)
        self.assertEqual(call_count[0], 2)
        self.assertEqual(res.tools_called, 2)
        self.assertEqual(res.tools_succeeded, 1)
        self.assertEqual(res.tools_failed, 1)
        self.assertIn("def fixed", res.final_text)

    def test_gemini_cli_real_execution(self):
        gemini = GeminiCliHarness()
        turns = [
            "call:calculator{\"expression\": \"12 + 30\"}",
            "The calculation gives 42.",
        ]
        idx = [0]

        def step_gen(p: str) -> str:
            resp = turns[min(idx[0], len(turns) - 1)]
            idx[0] += 1
            return resp

        res = gemini.run("Calculate 12+30", model_generate_fn=step_gen, max_turns=2)
        self.assertEqual(res.tools_called, 1)
        self.assertEqual(res.tools_succeeded, 1)
        self.assertIn("42", res.final_text)

    def test_pi_harness_real_execution(self):
        pi = PiHarness()
        turns = [
            "Action: write hello.txt\nhello from pi sandbox\n",
            "Action: read hello.txt\n",
        ]
        idx = [0]

        def step_gen(p: str) -> str:
            resp = turns[min(idx[0], len(turns) - 1)]
            idx[0] += 1
            return resp

        res = pi.run("Create file and verify", model_generate_fn=step_gen, max_turns=2)
        self.assertEqual(res.tools_called, 2)
        self.assertEqual(res.tools_succeeded, 2)
        self.assertEqual(res.tools_failed, 0)

    def test_math_verifier(self):
        self.assertTrue(MathVerifier.verify_answer("<think>Deduction</think> 42", "42"))
        self.assertTrue(MathVerifier.verify_answer("\\boxed{3.14159}", "3.1416"))
        self.assertTrue(MathVerifier.verify_answer("The answer is 120", "120"))
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

    def test_checkpoint_manager_save_load_and_readback(self):
        mgr = CheckpointManager(
            output_dir=self.temp_dir,
            num_latest_checkpoint_save=2,
            is_save_and_upload_checkpoint=False,
        )

        from sword.finetune import apply_lora_to_model
        lora_model = apply_lora_to_model(self.model, r=4, lora_alpha=8, target_moe=False)
        optimizer = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=1e-4)

        # Save checkpoint
        ckpt_dir = mgr.save_checkpoint(
            step=15,
            model=lora_model,
            optimizer=optimizer,
            base_model_name="test_base",
        )

        # Read-back verification was automatically executed and succeeded
        self.assertTrue(os.path.isdir(ckpt_dir))
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "adapter_model.safetensors")) or
                        os.path.exists(os.path.join(ckpt_dir, "adapter_model.bin")))
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "optimizer.pt")))

        # Load back into fresh optimizer and model
        fresh_optimizer = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=5e-5)
        res_info = mgr.load_checkpoint(ckpt_dir, model=lora_model, optimizer=fresh_optimizer)
        self.assertEqual(res_info["step"], 15)
        self.assertTrue(res_info["optimizer_restored"])
        self.assertGreater(res_info["matched_keys"], 0)

    def test_resume_only_when_explicitly_requested(self):
        mgr = CheckpointManager(
            output_dir=self.temp_dir,
            num_latest_checkpoint_save=2,
            is_save_and_upload_checkpoint=False,
        )
        from sword.finetune import apply_lora_to_model
        lora_model = apply_lora_to_model(self.model, r=4, lora_alpha=8, target_moe=False)
        mgr.save_checkpoint(step=10, model=lora_model)

        # Default empty or None must NOT resume
        self.assertIsNone(mgr.find_latest_checkpoint(None))
        self.assertIsNone(mgr.find_latest_checkpoint(""))
        self.assertIsNone(mgr.find_latest_checkpoint("none"))

        # Explicit 'latest' or 'auto' must resume
        latest = mgr.find_latest_checkpoint("latest")
        self.assertIsNotNone(latest)
        self.assertIn("checkpoint-10", latest)

    def test_oom_recovery_does_not_mark_dataset_completed(self):
        from sword.finetune import apply_lora_to_model
        apply_lora_to_model(self.model, r=4, lora_alpha=8, target_moe=False)

        custom_dataset = [
            {"problem": "Compute 1+1", "solution": "2", "can_be__verify": True},
        ]
        out_dir = os.path.join(self.temp_dir, "oom_test_ckpt")

        # Patch GRPOBridge.train_step to simulate persistent CUDA Out of Memory
        with patch("sword.grpo_bridge.GRPOBridge.train_step", side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory")):
            res = sword.train(
                long_train=True,
                test=False,
                dataset=custom_dataset,
                model="test_model",
                custom_checkpoint_save=True,
                is_save_and_upload_checkpoint=False,
                Argument={
                    "steps": 2,
                    "concurrency_g": 2,
                    "output_dir": out_dir,
                },
                show_detail_logs=False,
            )

            self.assertEqual(res.get("status"), "oom_aborted")
            self.assertTrue(os.path.isdir(res.get("emergency_checkpoint")))

            # Check that data_complete.logs does NOT exist or does NOT contain dataset hash
            ckpt_mgr = CheckpointManager(output_dir=out_dir)
            dataset_sha = res.get("dataset_sha256")
            self.assertFalse(ckpt_mgr.is_dataset_completed(dataset_sha))

        # A subsequent run outside OOM patch must NOT be rejected with 'already_trained'
        retry_res = sword.train(
            long_train=True,
            test=False,
            dataset=custom_dataset,
            model="test_model",
            custom_checkpoint_save=True,
            is_save_and_upload_checkpoint=False,
            Argument={
                "steps": 1,
                "concurrency_g": 2,
                "output_dir": out_dir,
            },
            show_detail_logs=False,
        )
        # Without OOM mock, it succeeds and completes
        self.assertEqual(retry_res.get("status"), "completed")

    def test_expandable_dataset_deduplication(self):
        from sword.finetune import apply_lora_to_model
        apply_lora_to_model(self.model, r=4, lora_alpha=8, target_moe=False)

        dataset_v1 = [
            {"problem": "Compute 2 + 2", "solution": "4", "can_be__verify": True},
            {"problem": "Compute 3 + 3", "solution": "6", "can_be__verify": True},
        ]
        out_dir = os.path.join(self.temp_dir, "expandable_ckpt")

        # 1. Train on dataset v1
        res1 = sword.train(
            long_train=True,
            test=False,
            dataset=dataset_v1,
            model="test_model",
            is_save_and_upload_checkpoint=False,
            Argument={"steps": 2, "concurrency_g": 2, "output_dir": out_dir},
            show_detail_logs=False,
        )
        self.assertEqual(res1.get("status"), "completed")

        # 2. Expand dataset with 1 new problem row
        dataset_v2 = list(dataset_v1) + [
            {"problem": "Compute 4 + 4", "solution": "8", "can_be__verify": True},
        ]

        # 3. Training on dataset v2 detects already trained rows and trains the new row
        res2 = sword.train(
            long_train=True,
            test=False,
            dataset=dataset_v2,
            model="test_model",
            is_save_and_upload_checkpoint=False,
            Argument={"steps": 1, "concurrency_g": 2, "output_dir": out_dir},
            show_detail_logs=False,
        )
        self.assertEqual(res2.get("status"), "completed")

        # 4. Rerunning dataset v2 when all rows are complete triggers already_trained
        res3 = sword.train(
            long_train=True,
            test=False,
            dataset=dataset_v2,
            model="test_model",
            is_save_and_upload_checkpoint=False,
            Argument={"steps": 1, "concurrency_g": 2, "output_dir": out_dir},
            show_detail_logs=False,
        )
        self.assertEqual(res3.get("status"), "already_trained")

    def test_partial_dataset_train_only_marks_visited_rows(self):
        from sword.finetune import apply_lora_to_model
        apply_lora_to_model(self.model, r=4, lora_alpha=8, target_moe=False)

        dataset = [
            {"problem": "Row 1: 1+1", "solution": "2", "can_be__verify": True, "harness": "opencode"},
            {"problem": "Row 2: 2+2", "solution": "4", "can_be__verify": True, "harness": "codex"},
            {"problem": "Row 3: 3+3", "solution": "6", "can_be__verify": True, "harness": "pi"},
        ]
        out_dir = os.path.join(self.temp_dir, "partial_ckpt")

        # Train only 1 step on 3-row dataset
        res = sword.train(
            long_train=True,
            test=False,
            dataset=dataset,
            model="test_model",
            is_save_and_upload_checkpoint=False,
            Argument={"steps": 1, "concurrency_g": 2, "output_dir": out_dir},
            show_detail_logs=False,
        )
        self.assertEqual(res.get("status"), "partial")
        self.assertEqual(res.get("visited_rows"), 1)
        self.assertEqual(res.get("total_rows"), 3)

        # Entire dataset must NOT be marked completed because 2 rows remain unvisited
        ckpt_mgr = CheckpointManager(output_dir=out_dir)
        dataset_sha = res.get("dataset_sha256")
        self.assertFalse(ckpt_mgr.is_dataset_completed(dataset_sha))

        # But row 1 must be marked completed
        records, _ = load_dataset_records(dataset)
        completed_hashes = ckpt_mgr.get_completed_dataset_hashes()
        self.assertIn(records[0].row_hash, completed_hashes)
        self.assertNotIn(records[1].row_hash, completed_hashes)
        self.assertNotIn(records[2].row_hash, completed_hashes)

    def test_upstream_prompts_and_schemas_and_budget_guard(self):
        from sword.harness.prompts import (
            load_opencode_prompt,
            load_codex_prompt,
            load_pi_prompt,
            load_gemini_prompt,
        )

        # 1. Verify ensure_harness_repo is invoked by all 4 prompt loaders
        with patch("sword.harness.prompts.ensure_harness_repo", return_value=True) as mock_ensure:
            opencode_p = load_opencode_prompt()
            codex_p = load_codex_prompt()
            pi_p = load_pi_prompt()
            gemini_p = load_gemini_prompt()
            called_keys = [c.args[0] for c in mock_ensure.call_args_list]
            self.assertEqual(called_keys, ["opencode", "codex", "pi", "gemini_cli"])

        # 2. Verify authentic upstream text extracted from third_party/ files and no empty bullet lines
        self.assertIn("You are an AI agent running in OpenCode", opencode_p)
        self.assertIn("Writes a file to the local filesystem", opencode_p)
        self.assertIn("You are Codex, based on GPT-5", codex_p)
        self.assertIn("*** Begin Patch", codex_p)
        self.assertIn("You are an expert coding assistant operating inside pi, a coding agent harness.", pi_p)
        self.assertIn("Make precise file edits with exact text replacement", pi_p)
        self.assertIn("# Core Mandates", gemini_p)
        self.assertIn("Handle Ambiguity/Expansion", gemini_p)
        self.assertIn("Explain Before Acting", gemini_p)
        self.assertIn("Non-Interactive Environment", gemini_p)
        self.assertIn("# Primary Workflows", gemini_p)
        self.assertIn("1. **Research:** Systematically map the codebase", gemini_p)
        self.assertIn("run_shell_command", gemini_p)
        for line in gemini_p.splitlines():
            self.assertNotEqual(line.strip(), "-")

        # 3. Verify pinned commits and prompt caching
        from sword.harness.prompts import HARNESS_GIT_COMMITS, get_harness_system_prompt, clear_prompt_cache
        self.assertEqual(set(HARNESS_GIT_COMMITS.keys()), {"opencode", "pi", "codex", "gemini_cli"})
        clear_prompt_cache()
        p_cached_1 = get_harness_system_prompt("opencode", use_cache=True)
        with patch("sword.harness.prompts.load_opencode_prompt", side_effect=RuntimeError("Should hit cache")):
            p_cached_2 = get_harness_system_prompt("opencode", use_cache=True)
        self.assertEqual(p_cached_1, p_cached_2)

        # 4. Verify tokenizer-aware and character-budget guard in BaseHarness.format_prompt
        gemini_h = GeminiCliHarness()
        task_text = "Write a function to compute Fibonacci numbers."
        trimmed = gemini_h.format_prompt(task_text, max_prompt_chars=500)
        self.assertLessEqual(len(trimmed), 500)
        self.assertTrue(trimmed.endswith(f"Task:\n{task_text}"))
        self.assertIn("...[trimmed]...", trimmed)

        class WordTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

        tok_trimmed = gemini_h.format_prompt(task_text, max_tokens=80, tokenizer=WordTokenizer())
        self.assertTrue(tok_trimmed.endswith(f"Task:\n{task_text}"))
        self.assertIn("# Gemini CLI Tool Declarations:", tok_trimmed)
        self.assertIn("...[trimmed]...", tok_trimmed)

        # 5. Verify Codex apply_patch freeform grammar: Add File, Update File (multi-hunk + Move to), Delete File
        codex_h = CodexCliHarness()
        patch_turn = (
            "*** Begin Patch\n"
            "*** Add File: calc.py\n"
            "+x = 21\n"
            "+y = 2\n"
            "+print(x * y)\n"
            "*** Add File: temp.txt\n"
            "+temporary\n"
            "*** End Patch\n"
            "*** Begin Patch\n"
            "*** Update File: calc.py\n"
            "*** Move to: moved_calc.py\n"
            "@@\n"
            "-x = 21\n"
            "+x = 50\n"
            "@@\n"
            "-y = 2\n"
            "+y = 3\n"
            "*** Delete File: temp.txt\n"
            "*** End Patch\n"
            "```python\n"
            "import os\n"
            "assert not os.path.exists('temp.txt')\n"
            "assert not os.path.exists('calc.py')\n"
            "with open('moved_calc.py') as f:\n"
            "    code = f.read()\n"
            "assert 'x = 50' in code and 'y = 3' in code\n"
            "```"
        )
        res_codex = codex_h.run("Patch and verify", lambda p: patch_turn, max_turns=1)
        self.assertEqual(res_codex.tools_called, 3)
        self.assertEqual(res_codex.tools_succeeded, 3)

        # 6. Verify Pi atomic edits[] (both non-overlapping succeed, and failed edit leaves file untouched)
        pi_h = PiHarness()
        pi_turns = [
            '```json\n{"action": "write", "parameters": {"path": "note.txt", "content": "alpha beta gamma"}}\n```',
            '```json\n{"action": "edit", "parameters": {"path": "note.txt", "edits": [{"oldText": "alpha", "newText": "FIRST"}, {"oldText": "gamma", "newText": "LAST"}]}}\n```',
            '```json\n{"action": "edit", "parameters": {"path": "note.txt", "edits": [{"oldText": "FIRST", "newText": "BROKEN"}, {"oldText": "NONEXISTENT", "newText": "FAIL"}]}}\n```',
            '```python\nwith open("note.txt") as f:\n    c = f.read()\nassert c == "FIRST beta LAST", f"Unexpected content: {c}"\n```',
        ]
        t_idx = [0]

        def pi_gen(p: str) -> str:
            r = pi_turns[min(t_idx[0], len(pi_turns) - 1)]
            t_idx[0] += 1
            return r

        res_pi = pi_h.run("Edit note.txt atomically", pi_gen, max_turns=4)
        self.assertEqual(res_pi.tools_called, 4)
        self.assertEqual(res_pi.tools_succeeded, 3)
        self.assertEqual(res_pi.tools_failed, 1)

        # 7. Verify real grep_search and glob in GeminiCliHarness
        gem_turns = [
            'call:write_file{"file_path": "src/util.py", "content": "def magic_helper():\\n    return 99\\n"}',
            'call:glob{"pattern": "**/*.py"}',
            'call:grep_search{"pattern": "magic_helper"}',
        ]
        g_idx = [0]

        def gem_gen(p: str) -> str:
            r = gem_turns[min(g_idx[0], len(gem_turns) - 1)]
            g_idx[0] += 1
            return r

        res_gem = gemini_h.run("Search workspace", gem_gen, max_turns=3)
        self.assertEqual(res_gem.tools_called, 3)
        self.assertEqual(res_gem.tools_succeeded, 3)
        self.assertIn("src/util.py", res_gem.tool_history[1]["output"])
        self.assertIn("src/util.py:1:def magic_helper():", res_gem.tool_history[2]["output"])

    def test_prep_and_logs_execution(self):
        import sword
        res = sword.train(
            prep_and_logs=True,
            B_model_port=11343,
            test=True,
            mock_test=True,
        )
        self.assertTrue(res.get("ok"))
        self.assertIn("patch_result", res)
        self.assertIn("harness_status", res)
        self.assertIn("b_model_diagnostic", res)
        self.assertIn("train_result", res)
        train_res = res["train_result"]
        self.assertEqual(train_res.get("status"), "completed")
        self.assertEqual(train_res.get("final_step"), 5)
        self.assertEqual(train_res.get("visited_rows"), 5)
        self.assertEqual(os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"), "0")
        self.assertEqual(os.environ.get("VLLM_USE_FLASHINFER"), "0")
        self.assertEqual(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"), "spawn")


if __name__ == "__main__":
    unittest.main()
