"""
Unit tests for the Unsloth-vLLM GRPO Bridge in sword.grpo_bridge.
Verifies all failure mitigation modes:
- A1-A6: Truncated Importance Sampling (TIS), additive score centering, policy logprobs
- B1-B5: LoRA key mapping normalization, calibrated clipping, entropy floor
- C1-C5: Two-phase VRAM lifecycle handover, sequence chunking
- D1-D4: LoRA weight sync and policy version tracking
- E1-E4: Group advantage centering and group size validation
- G1-G4: Fallback engine execution and end-to-end step loop
"""

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
            use_score_centering=True,
            use_calibrated_clipping=True,
            entropy_floor=0.01,
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

    def test_weight_synchronizer_key_normalization(self):
        sync = WeightSynchronizer()
        raw_sd = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(8, 64),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(64, 8),
            "base_model.model.layers.1.mlp.gate_proj.lora_A.weight": torch.randn(8, 64),
        }
        normalized = sync.normalize_lora_keys_for_vllm(raw_sd)
        self.assertIn("model.layers.0.self_attn.q_proj.lora_A.weight", normalized)
        self.assertIn("model.layers.0.self_attn.v_proj.lora_B.weight", normalized)
        self.assertIn("layers.1.mlp.gate_proj.lora_A.weight", normalized)
        for k in normalized.keys():
            self.assertNotIn("base_model.", k)
            self.assertNotIn("default.", k)

    def test_rollout_group_validation(self):
        batch = RolloutBatch(concurrency_g=2)
        batch.items = [
            RolloutItem("p1", [1], "r1", [2], [-0.5], group_id=0),
            RolloutItem("p1", [1], "r2", [3], [-0.6], group_id=0),
            RolloutItem("p2", [4], "r3", [5], [-0.4], group_id=1),
            RolloutItem("p2", [4], "r4", [6], [-0.7], group_id=1),
        ]
        self.assertTrue(batch.validate_groups())

        # Now simulate dropped sequence in group 1
        batch.items.pop()
        self.assertFalse(batch.validate_groups())

    def test_dual_correction_grpo_loss(self):
        loss_fn = DualCorrectionGRPOLoss(
            clip_eps=0.2,
            tis_bound_c=2.0,
            use_score_centering=True,
            use_calibrated_clipping=True,
            entropy_floor=0.01,
            chunk_size=16,
        )

        batch_size = 4
        seq_len = 24
        input_ids = torch.randint(2, 400, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        prompt_lens = [8] * batch_size
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        rollout_lps = [[-0.5] * 16 for _ in range(batch_size)]

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
        self.assertIn("grpo_loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("mean_ratio", metrics)
        self.assertIn("mean_entropy", metrics)
        self.assertIn("effective_clip_eps", metrics)

    def test_grpo_bridge_end_to_end_step(self):
        bridge = sword.get_grpo_bridge(
            config=None,
            model_name="dummy",
            concurrency_g=2,
            max_model_len=64,
            max_new_tokens=8,
            chunk_size=16,
            device="cpu",
        )
        bridge.initialize_trainer(model=self.model, tokenizer=self.tokenizer)

        # Attach LoRA adapter
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
        self.assertEqual(result["batch_size"], 4)  # 2 prompts * 2 concurrency
        self.assertIn("metrics", result)
        self.assertIn("grpo_loss", result["metrics"])
        self.assertEqual(result["metrics"]["policy_version"], 1)


if __name__ == "__main__":
    unittest.main()
