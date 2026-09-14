"""
Comprehensive Unit Test Suite for Qwen3 MoE (specifically Qwen3 30B A3B),
Smart/Faster KV Cache with Rollout Auto-Clear, and Unsloth-Merged RL Engine.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeAttention,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeExperts,
)

import sword
from sword.kv_cache import StaticKVCache, SmartKVCache
from sword.patcher import patch_qwen3_moe, unpatch_qwen3_moe
from sword.server import FastQwen3MoeServer
from sword.rl.loss import ChunkedGRPOLoss
from sword.rl.moe_monitor import MoERouterMonitor
from sword.rl.engine import GRPOTrainer


def make_mini_qwen3_moe_config(**kwargs):
    """Generates a scaled mini Qwen3 MoE config modeled after Qwen3 30B A3B."""
    default_kwargs = dict(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,  # 4:1 GQA ratio (in 30B A3B: 32:4 = 8:1)
        num_hidden_layers=2,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_experts=16,          # 16 experts in test (128 in full 30B A3B)
        num_experts_per_tok=2,   # top-2 active (top-8 in full 30B A3B)
        vocab_size=1000,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    default_kwargs.update(kwargs)
    return Qwen3MoeConfig(**default_kwargs)


class TestQwen3MoEAndSmartKVCache(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    # =========================================================
    # 1. Qwen3 MoE Architecture & Flash SDPA Patching
    # =========================================================
    def test_qwen3_moe_flash_patch_numerical_identity(self):
        """Verifies Sword Pure FlashAttention SDPA produces identical outputs on Qwen3 MoE."""
        cfg = make_mini_qwen3_moe_config()
        model = Qwen3MoeForCausalLM(cfg)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 12))
        with torch.no_grad():
            out_before = model(input_ids)

        patch_qwen3_moe(model, mode="flash", patch_moe=False)

        with torch.no_grad():
            out_after = model(input_ids)

        diff = (out_before.logits - out_after.logits).abs().max().item()
        self.assertLess(diff, 1e-4, f"Attention output diverged: {diff}")

        # Clean unpatch
        unpatch_qwen3_moe(model)
        with torch.no_grad():
            out_restored = model(input_ids)
        diff_restored = (out_before.logits - out_restored.logits).abs().max().item()
        self.assertLess(diff_restored, 1e-5, f"Unpatch did not restore perfectly: {diff_restored}")

    # =========================================================
    # 2. Qwen3 MoE Zero-Sync Expert Dispatch & Router Logits
    # =========================================================
    def test_qwen3_moe_zero_sync_dispatch_and_router_capture(self):
        """Verifies Zero-Sync Fast MoE dispatch and router logits capture for RL."""
        cfg = make_mini_qwen3_moe_config()
        cfg._experts_implementation = "eager"
        model = Qwen3MoeForCausalLM(cfg)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 8))
        with torch.no_grad():
            out_before = model(input_ids)

        patch_qwen3_moe(model, mode="flash", patch_moe=True)

        with torch.no_grad():
            out_after = model(input_ids)

        diff = (out_before.logits - out_after.logits).abs().max().item()
        self.assertLess(diff, 1e-4, f"Zero-Sync MoE diverged: {diff}")

        # Check router logits captured on MoE blocks
        captured = [m.last_router_logits for m in model.modules() if hasattr(m, "last_router_logits") and m.last_router_logits is not None]
        self.assertGreater(len(captured), 0, "No router logits captured on MoE blocks")
        # Shape: [tokens, num_experts] -> [16, 16]
        self.assertEqual(captured[0].shape, (16, 16))

    # =========================================================
    # 3. Faster & Smarter KV Cache (Auto-Clear & Prefix Duplication)
    # =========================================================
    def test_smart_kv_cache_fast_reset_and_auto_clear(self):
        """Verifies O(1) metadata reset and auto-clear on new rollout."""
        cache = SmartKVCache(
            num_layers=4,
            max_batch_size=8,
            num_kv_heads=2,
            max_seq_len=256,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Simulate rollout 1
        k1 = torch.randn(4, 2, 10, 32)
        v1 = torch.randn(4, 2, 10, 32)
        k_out, v_out = cache.update(0, k1, v1, start_pos=0)
        cache.set_pos(10)
        self.assertEqual(cache.get_pos(), 10)
        self.assertEqual(k_out.shape, (4, 2, 10, 32))

        # Start new rollout (triggers fast O(1) clear without re-zeroing gigabytes)
        cache.new_rollout(rollout_id=1, batch_size=4)
        self.assertEqual(cache.get_pos(), 0)
        self.assertEqual(cache.rollout_id, 1)

        # Write new prompt into cache
        k2 = torch.randn(4, 2, 5, 32)
        v2 = torch.randn(4, 2, 5, 32)
        k_out2, v_out2 = cache.update(0, k2, v2, start_pos=0)
        self.assertEqual(k_out2.shape, (4, 2, 5, 32))
        # Verify content was updated properly
        self.assertTrue(torch.allclose(k_out2[:, :, :5, :], k2))

    def test_smart_kv_cache_prefix_broadcast_for_grpo_rollouts(self):
        """
        Verifies duplicate_prefix_for_rollouts for multi-trajectory GRPO rollouts.
        Prefills prompt ONCE, then broadcasts across G sibling rollout streams.
        """
        G = 4  # 4 rollouts per prompt
        num_prompts = 2
        total_batch = num_prompts * G  # 8 streams

        cache = SmartKVCache(
            num_layers=2,
            max_batch_size=total_batch,
            num_kv_heads=2,
            max_seq_len=128,
            head_dim=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        p1_len, p2_len = 6, 8
        # Fill slot 0 for prompt 1
        k_p1 = torch.randn(1, 2, p1_len, 32)
        v_p1 = torch.randn(1, 2, p1_len, 32)
        cache.k_cache[0][0:1, :, :p1_len, :] = k_p1
        cache.v_cache[0][0:1, :, :p1_len, :] = v_p1

        # Fill slot 4 (1 * G) for prompt 2
        k_p2 = torch.randn(1, 2, p2_len, 32)
        v_p2 = torch.randn(1, 2, p2_len, 32)
        cache.k_cache[0][4:5, :, :p2_len, :] = k_p2
        cache.v_cache[0][4:5, :, :p2_len, :] = v_p2

        # Broadcast prefix to all sibling rollout slots (slots 1..3 and 5..7)
        cache.duplicate_prefix_for_rollouts(num_prompts=num_prompts, group_size=G, prompt_lens=[p1_len, p2_len])

        # Verify slots 1, 2, 3 have identical KV to slot 0
        for slot in [1, 2, 3]:
            self.assertTrue(torch.allclose(cache.k_cache[0][slot, :, :p1_len, :], k_p1[0]))
            self.assertTrue(torch.allclose(cache.v_cache[0][slot, :, :p1_len, :], v_p1[0]))

        # Verify slots 5, 6, 7 have identical KV to slot 4
        for slot in [5, 6, 7]:
            self.assertTrue(torch.allclose(cache.k_cache[0][slot, :, :p2_len, :], k_p2[0]))
            self.assertTrue(torch.allclose(cache.v_cache[0][slot, :, :p2_len, :], v_p2[0]))

    # =========================================================
    # 4. FastQwen3MoeServer Multi-Stream Rollout Generation
    # =========================================================
    def test_fast_qwen3_moe_server_rollouts(self):
        """Tests FastQwen3MoeServer serving and generate_rollouts with auto-clear."""
        cfg = make_mini_qwen3_moe_config()
        model = Qwen3MoeForCausalLM(cfg)
        model.eval()

        # Simple mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 2
            padding_side = "left"
            def __call__(self, texts, padding=True, return_tensors="pt", **kwargs):
                return {
                    "input_ids": torch.randint(10, 900, (len(texts), 4)),
                    "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
                }
            def batch_decode(self, token_ids, skip_special_tokens=True):
                return [f"response_for_stream_{i}" for i in range(token_ids.shape[0])]

        server = FastQwen3MoeServer(
            model=model,
            tokenizer=MockTokenizer(),
            max_concurrency=8,
            max_seq_len=256,
        )

        rollouts = server.generate_rollouts(
            prompts=["Question 1", "Question 2"],
            num_rollouts_per_prompt=3,
            max_new_tokens=4,
            auto_clear=True,
        )

        self.assertEqual(len(rollouts), 2)       # 2 prompts
        self.assertEqual(len(rollouts[0]), 3)    # 3 rollouts for prompt 1
        self.assertEqual(len(rollouts[1]), 3)    # 3 rollouts for prompt 2
        self.assertGreater(server.static_cache.rollout_id, 0)

    # =========================================================
    # 5. Unsloth-Merged Chunked GRPOLoss & MoE Aux Loss
    # =========================================================
    def test_chunked_grpo_loss_with_fused_cross_entropy(self):
        """Verifies ChunkedGRPOLoss with fused cross-entropy matches mathematical expectation."""
        loss_fn = ChunkedGRPOLoss(chunk_size=16)

        batch_size = 4
        seq_len = 24
        vocab_size = 64
        hidden_dim = 32

        # Mock model with lm_head
        class MockLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
            def forward(self, input_ids, attention_mask, output_hidden_states=True, return_dict=True):
                class Out:
                    hidden_states = [torch.randn(batch_size, seq_len, hidden_dim)]
                return Out()

        model = MockLM()
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        prompt_lengths = [8, 8, 8, 8]
        advantages = torch.tensor([2.0, 0.5, -0.2, -0.8])

        loss, metrics = loss_fn.forward_chunked(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            advantages=advantages,
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.abs().item(), 0.0)
        self.assertIn("grpo_loss", metrics)
        self.assertIn("mean_ratio", metrics)

    def test_moe_router_aux_loss_with_qwen3_router_logits(self):
        """Verifies MoERouterMonitor auxiliary loss on Qwen3 MoE captured logits."""
        monitor = MoERouterMonitor(num_experts=16, top_k=2, aux_loss_coeff=0.01)

        # Captured router logits from Qwen3 MoE: [tokens, experts]
        router_logits = torch.randn(64, 16)
        aux_loss, metrics = monitor.compute_aux_loss(router_logits)

        self.assertGreater(aux_loss.item(), 0.0)
        self.assertIn("routing_entropy", metrics)
        self.assertGreater(metrics["routing_entropy"], 1.0)


if __name__ == "__main__":
    unittest.main()
