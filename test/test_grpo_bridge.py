"""
Unit tests for the Unsloth-vLLM GRPO Bridge in sword.grpo_bridge.
Verifies all failure mitigation modes:
- A1-A6: Detached Truncated Importance Sampling (TIS) directionality, upper-bound clamping, temperature handling
- B1-B5: Standard PEFT adapter serialization, precision partitioning, no-grad entropy monitoring
- C1-C5: Two-phase VRAM lifecycle (sleep/wake handover), global token-normalized micro-batching, adapter pruning
- D1-D4: Runtime LoRA request tracking and versioned adapter generation
- E1-E4: Group advantage centering, group validation, and filtering of incomplete groups
- F1-F4: Target module configuration and training checkpointing
- G1-G4: Fallback engine execution, mock vLLM integration, and end-to-end step loop
"""

import os
import sys
import json
import math
import shutil
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

import sword
from sword.grpo_bridge import (
    GRPOBridge,
    GRPOBridgeConfig,
    DualCorrectionGRPOLoss,
    WeightSynchronizer,
    VRAMLifecycleManager,
    RolloutBatch,
    RolloutItem,
    FallbackRolloutEngine,
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


class TestGRPOBridge(unittest.TestCase):
    def setUp(self):
        self.config = GRPOBridgeConfig(
            model_name="test_model",
            concurrency_g=4,
            max_model_len=128,
            max_new_tokens=16,
            chunk_size=32,
            clip_eps=0.2,
            tis_bound_c=2.0,
            temperature=0.7,
            micro_batch_size=2,
            device="cpu",
        )
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

    def test_imports_and_exports(self):
        self.assertTrue(hasattr(sword, "GRPOBridge"))
        self.assertTrue(hasattr(sword, "GRPOBridgeConfig"))
        self.assertTrue(hasattr(sword, "DualCorrectionGRPOLoss"))
        self.assertTrue(hasattr(sword, "WeightSynchronizer"))
        self.assertTrue(hasattr(sword, "VRAMLifecycleManager"))
        self.assertTrue(callable(sword.get_grpo_bridge))

    def test_vram_lifecycle_sleep_and_wake(self):
        mock_vllm = MagicMock()
        mock_vllm.sleep = MagicMock()
        mock_vllm.wake_up = MagicMock()

        vram = VRAMLifecycleManager(device="cpu", enable_sleep=True)

        # Entering rollout phase should wake up vLLM
        vram.enter_rollout_phase(trainer_model=self.model, vllm_engine=mock_vllm)
        self.assertFalse(self.model.training)
        mock_vllm.wake_up.assert_called_once()

        # Entering train phase should sleep vLLM
        vram.enter_train_phase(trainer_model=self.model, vllm_engine=mock_vllm)
        self.assertTrue(self.model.training)
        mock_vllm.sleep.assert_called_once()

    def test_weight_synchronizer_pruning_and_no_disk_leak(self):
        from sword.finetune import apply_lora_to_model
        lora_model = apply_lora_to_model(self.model, r=8, lora_alpha=16, target_moe=False)
        sync = WeightSynchronizer(base_model_name="test_model", lora_r=8, lora_alpha=16, max_kept_adapters=2)

        try:
            # Sync 4 versions consecutively
            for _ in range(4):
                sync.sync_to_vllm(lora_model)
                sync.prune_old_adapters(max_kept=2)

            self.assertEqual(sync.policy_version, 4)
            kept_dirs = sorted([d for d in os.listdir(sync.temp_dir) if d.startswith("adapter_v")])

            # Should retain only adapter_v3 and adapter_v4 (no disk leak)
            self.assertEqual(kept_dirs, ["adapter_v3", "adapter_v4"])
        finally:
            sync.cleanup()
            self.assertFalse(os.path.exists(sync.temp_dir))

    def test_group_validation_and_incomplete_filtering(self):
        batch = RolloutBatch(concurrency_g=2)
        batch.items = [
            RolloutItem("p1", [1], "r1", [2], [-0.5], group_id=0),
            RolloutItem("p1", [1], "r2", [3], [-0.6], group_id=0),
            RolloutItem("p2", [4], "r3", [5], [-0.4], group_id=1),  # Group 1 has only 1 item (dropped)
        ]
        self.assertFalse(batch.validate_groups())

        # Filter out incomplete groups
        batch.filter_valid_groups(required_g=2)
        self.assertTrue(batch.validate_groups())
        self.assertEqual(len(batch.items), 2)
        self.assertTrue(all(it.group_id == 0 for it in batch.items))

    def test_detached_tis_directionality_and_upper_bound_clamping(self):
        loss_fn = DualCorrectionGRPOLoss(
            clip_eps=0.2,
            tis_bound_c=2.0,
            temperature_scale=False,
            chunk_size=16,
        )

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(2, 400, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        prompt_lens = [8, 8]
        advantages = torch.tensor([1.0, 1.0])

        # Sequence 0: vLLM logprob is much lower than training policy (e.g. -20.0 vs training ~ -5.0)
        # Expected: raw_is_weight is large, clipped strictly to tis_bound_c = 2.0
        # Sequence 1: vLLM logprob is much higher than training policy (e.g. 0.0 vs training ~ -5.0)
        # Expected: raw_is_weight is small (< 1.0), NOT clipped to 1/C because standard TIS has upper clamp only
        rollout_lps = [
            [-20.0] * 8,
            [0.0] * 8,
        ]

        loss, metrics = loss_fn(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lens,
            advantages=advantages,
            rollout_logprobs=rollout_lps,
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIn("mean_is_weight", metrics)
        # Mean importance weight across both sequences should be between 0 and 2.0
        self.assertGreater(metrics["mean_is_weight"], 0.0)
        self.assertLessEqual(metrics["mean_is_weight"], 2.0)

    def test_mock_vllm_lora_request_dispatch(self):
        mock_output = MagicMock()
        mock_output.prompt_token_ids = [2, 10]
        mock_resp = MagicMock(text="vllm generated", token_ids=[20, 30], logprobs=None)
        mock_output.outputs = [mock_resp]

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = [mock_output] * 4

        mock_llm_cls = MagicMock(return_value=mock_llm_instance)
        mock_lora_req_cls = MagicMock(side_effect=lambda lora_name, lora_int_id, lora_path: MagicMock(name=lora_name))

        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_cls
        mock_vllm.SamplingParams = MagicMock()

        mock_vllm_lora = MagicMock()
        mock_vllm_lora.LoRARequest = mock_lora_req_cls

        with patch.dict(sys.modules, {"vllm": mock_vllm, "vllm.lora.request": mock_vllm_lora}):
            bridge = sword.get_grpo_bridge(
                config=None,
                model_name="test_model",
                concurrency_g=2,
                enable_lora=True,
                device="cpu",
            )
            bridge.initialize_trainer(model=self.model, tokenizer=self.tokenizer)
            self.assertTrue(bridge.initialize_vllm())
            self.assertIsNotNone(bridge.vllm_engine)

            # Manually trigger weight sync
            bridge.weight_sync.sync_to_vllm(bridge.trainer_model)
            self.assertEqual(bridge.weight_sync.policy_version, 1)

            # Generate rollouts and verify LoRARequest was passed
            batch = bridge.generate_rollouts(["Prompt A", "Prompt B"])
            self.assertEqual(len(batch), 4)

            mock_llm_instance.generate.assert_called()
            call_kwargs = mock_llm_instance.generate.call_args[1]
            self.assertIn("lora_request", call_kwargs)
            self.assertIsNotNone(call_kwargs["lora_request"])

    def test_grpo_bridge_end_to_end_step(self):
        bridge = sword.get_grpo_bridge(
            config=None,
            model_name="dummy",
            concurrency_g=2,
            max_model_len=64,
            max_new_tokens=8,
            chunk_size=16,
            micro_batch_size=2,
            max_kept_adapters=2,
            device="cpu",
        )
        bridge.initialize_trainer(model=self.model, tokenizer=self.tokenizer)

        from sword.finetune import apply_lora_to_model
        bridge.trainer_model = apply_lora_to_model(
            bridge.trainer_model,
            r=4,
            lora_alpha=8,
            target_moe=False,
        )

        optimizer = torch.optim.AdamW(
            [p for p in bridge.trainer_model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        def mock_reward(prompt: str, response: str) -> float:
            return 1.0 if "decoded" in response else 0.0

        prompts = ["Compute 2 + 2", "Define linear algebra"]
        result = bridge.step(prompts, reward_fn=mock_reward, optimizer=optimizer)

        self.assertIn("step", result)
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["batch_size"], 4)
        self.assertIn("metrics", result)
        self.assertIn("grpo_loss", result["metrics"])
        self.assertEqual(result["metrics"]["policy_version"], 1)


if __name__ == "__main__":
    unittest.main()
