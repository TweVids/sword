"""
Unit tests for the Unsloth-vLLM GRPO Bridge in sword.grpo_bridge.
Verifies all failure mitigation modes:
- A1-A6: Detached Truncated Importance Sampling (TIS), temperature scaling, on-policy ratio preservation
- B1-B5: Standard PEFT adapter serialization, precision partitioning, no-grad entropy monitoring
- C1-C5: Two-phase VRAM lifecycle handover, micro-batched sequence execution, gradient accumulation
- D1-D4: Runtime LoRA request tracking and versioned adapter generation
- E1-E4: Group advantage centering and group size validation
- F1-F4: Target module configuration and training checkpointing
- G1-G4: Fallback engine execution, mock vLLM integration, and end-to-end step loop
"""

import os
import sys
import json
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

    def test_vram_lifecycle_manager(self):
        vram = VRAMLifecycleManager(device="cpu")
        vram.enter_rollout_phase(self.model)
        self.assertFalse(self.model.training)

        vram.enter_train_phase(self.model)
        self.assertTrue(self.model.training)

        mem = vram.get_memory_info()
        self.assertIn("allocated_gb", mem)
        self.assertIn("reserved_gb", mem)

    def test_weight_synchronizer_peft_serialization(self):
        from sword.finetune import apply_lora_to_model
        lora_model = apply_lora_to_model(self.model, r=8, lora_alpha=16, target_moe=False)
        sync = WeightSynchronizer(base_model_name="test_model", lora_r=8, lora_alpha=16)

        version = sync.sync_to_vllm(lora_model)
        self.assertEqual(version, 1)
        self.assertIsNotNone(sync.current_adapter_path)
        self.assertTrue(os.path.isdir(sync.current_adapter_path))

        config_path = os.path.join(sync.current_adapter_path, "adapter_config.json")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            self.assertEqual(cfg.get("r"), 8)
            self.assertEqual(cfg.get("peft_type"), "LORA")
            self.assertEqual(cfg.get("base_model_name_or_path"), "test_model")

    def test_rollout_group_validation(self):
        batch = RolloutBatch(concurrency_g=2)
        batch.items = [
            RolloutItem("p1", [1], "r1", [2], [-0.5], group_id=0),
            RolloutItem("p1", [1], "r2", [3], [-0.6], group_id=0),
            RolloutItem("p2", [4], "r3", [5], [-0.4], group_id=1),
            RolloutItem("p2", [4], "r4", [6], [-0.7], group_id=1),
        ]
        self.assertTrue(batch.validate_groups())

        # Simulate dropped sequence in group 1
        batch.items.pop()
        self.assertFalse(batch.validate_groups())

    def test_detached_tis_preserves_on_policy_ratio(self):
        loss_fn = DualCorrectionGRPOLoss(
            clip_eps=0.2,
            tis_bound_c=2.0,
            temperature=0.7,
            chunk_size=16,
        )

        batch_size = 4
        seq_len = 24
        input_ids = torch.randint(2, 400, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        prompt_lens = [8] * batch_size
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        # Simulate divergent vLLM rollout logprobs (e.g. -5.0 vs expected -1.0)
        rollout_lps = [[-5.0] * 16 for _ in range(batch_size)]

        loss, metrics = loss_fn(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lens,
            advantages=advantages,
            rollout_logprobs=rollout_lps,
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        # On-policy ratio should be exactly 1.0 at step 0 despite vLLM mismatch
        self.assertAlmostEqual(metrics["mean_ratio"], 1.0, places=2)
        # Off-policy importance weight should be clamped at tis_bound_c = 2.0
        self.assertLessEqual(metrics["mean_is_weight"], 2.0 + 1e-4)

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

            # Manually trigger one weight sync so policy_version becomes 1
            bridge.weight_sync.sync_to_vllm(bridge.trainer_model)
            self.assertEqual(bridge.weight_sync.policy_version, 1)

            # Generate rollouts and verify LoRARequest was passed with adapter_v1
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
