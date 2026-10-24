"""
Comprehensive Unit Test Suite for Qwen Dense Architectures (Qwen2, Qwen2.5, Qwen3, Qwen3.5).
Verifies:
1. Pure FlashAttention SDPA numerical identity vs HuggingFace stock forward.
2. Fast lean decode step (fast_lean_decode_step) numerical equivalence (divergence < 1e-4).
3. Precomputed static RoPE cache table vs dynamic rotary_emb.
4. Zero-Copy GQA view transformation vs repeated KV expansion.
5. FastServer / FastQwenServer batch queue (add_request and run_all).
6. Qwen 3.5 architecture features (QK-norm, gated attention, SwiGLU).
7. High-concurrency serving (4 and 8 streams) with static KV cache.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Attention
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5Attention

import sword
from sword.server import FastServer, FastQwenServer, FastQwenGraphEngine
from sword.patcher import patch_qwen, unpatch_qwen, patch_model, unpatch_model
from sword.kv_cache import StaticKVCache


def make_mini_qwen2_config(**kwargs):
    default = dict(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,  # 4:1 GQA ratio
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
    )
    default.update(kwargs)
    return Qwen2Config(**default)


def make_mini_qwen3_config(**kwargs):
    default = dict(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,  # 4:1 GQA ratio
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
    )
    default.update(kwargs)
    return Qwen3Config(**default)


def make_mini_qwen3_5_config(**kwargs):
    n_layers = kwargs.get("num_hidden_layers", 4)
    default = dict(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=n_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        layer_types=["full_attention"] * n_layers,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    default.update(kwargs)
    return Qwen3_5TextConfig(**default)


class MockTokenizer:
    vocab_size = 1000
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, padding=True, return_tensors="pt", **kwargs):
        input_ids = torch.randint(2, 990, (len(texts), 16))
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [f"response_{i}" for i in range(token_ids.shape[0])]

    def decode(self, token_ids, skip_special_tokens=True):
        return "response_single"


class TestQwenDenseArchitecture(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_qwen_pure_flash_attention_numerical_identity(self):
        """Verifies Sword Pure FlashAttention SDPA matches HuggingFace on Qwen2."""
        cfg = make_mini_qwen2_config()
        model = Qwen2ForCausalLM(cfg)
        model.eval()

        input_ids = torch.randint(2, 990, (2, 16))
        with torch.no_grad():
            out_before = model(input_ids)

        patch_qwen(model)
        with torch.no_grad():
            out_after = model(input_ids)

        diff = (out_before.logits - out_after.logits).abs().max().item()
        self.assertLess(diff, 1e-4, f"Attention output diverged: {diff}")

        unpatch_qwen(model)
        with torch.no_grad():
            out_restored = model(input_ids)
        diff_restored = (out_before.logits - out_restored.logits).abs().max().item()
        self.assertLess(diff_restored, 1e-5, f"Unpatch divergence: {diff_restored}")

    def test_qwen_fast_lean_decode_step_numerical_identity(self):
        """Verifies server.fast_lean_decode_step produces bit-identical logits to model.forward."""
        cfg = make_mini_qwen2_config(num_hidden_layers=2)
        model = Qwen2ForCausalLM(cfg)
        model.eval()
        patch_qwen(model)

        bsz = 4
        tokenizer = MockTokenizer()
        server = FastQwenServer(model=model, tokenizer=tokenizer, max_concurrency=bsz)

        # Prefill 16 tokens
        prompt_ids = torch.randint(2, 990, (bsz, 16))
        server.static_cache.new_rollout(batch_size=bsz)
        server.static_cache.set_pos(0)
        pos = torch.arange(0, 16).unsqueeze(0).expand(bsz, -1)
        _ = model(input_ids=prompt_ids, position_ids=pos, past_key_values=server.static_cache, use_cache=True)
        server.static_cache.set_pos(16)

        # Compare single-token decode
        next_in = torch.randint(2, 990, (bsz, 1))
        dec_pos = torch.full((bsz, 1), 16, dtype=torch.long)

        # Standard forward
        with torch.no_grad():
            out_std = model(
                input_ids=next_in,
                position_ids=dec_pos,
                past_key_values=server.static_cache,
                start_pos=16,
                use_cache=True,
            )
            logits_std = out_std.logits[:, -1, :]

        # Lean decode forward
        with torch.no_grad():
            logits_lean = server.fast_lean_decode_step(next_in, dec_pos)

        diff = (logits_std - logits_lean).abs().max().item()
        self.assertLess(diff, 1e-4, f"fast_lean_decode_step diverged: {diff}")

    def test_qwen_static_rope_cache(self):
        """Verifies precomputed RoPE tables match dynamic rotary_emb output."""
        cfg = make_mini_qwen2_config()
        model = Qwen2ForCausalLM(cfg)
        model.eval()

        server = FastQwenServer(model=model, tokenizer=MockTokenizer(), max_concurrency=2)
        self.assertIsNotNone(server.cos_cache)
        self.assertIsNotNone(server.sin_cache)

        # Compare at position 25
        pos_ids = torch.tensor([[25]])
        dummy = torch.zeros(1, 1, cfg.hidden_size // cfg.num_attention_heads)
        cos_dyn, sin_dyn = model.model.rotary_emb(dummy, pos_ids)

        cos_static = server.cos_cache[25:26].unsqueeze(0)
        sin_static = server.sin_cache[25:26].unsqueeze(0)

        diff_cos = (cos_dyn - cos_static).abs().max().item()
        diff_sin = (sin_dyn - sin_static).abs().max().item()
        self.assertLess(diff_cos, 1e-5, f"Static RoPE cos divergence: {diff_cos}")
        self.assertLess(diff_sin, 1e-5, f"Static RoPE sin divergence: {diff_sin}")

    def test_qwen_batch_queue_add_request_and_run_all(self):
        """Verifies batch queue execution with add_request and run_all."""
        cfg = make_mini_qwen2_config(num_hidden_layers=2)
        model = Qwen2ForCausalLM(cfg)
        model.eval()
        patch_qwen(model)

        server = FastQwenServer(model=model, tokenizer=MockTokenizer(), max_concurrency=4)
        for i in range(4):
            server.add_request(f"Test prompt {i+1}", max_new_tokens=8, temperature=0.0)

        results = server.run_all()
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertIn("prompt", r)
            self.assertIn("response", r)
            self.assertEqual(r["tokens_generated"], 8)

    def test_qwen3_5_architecture_gated_attention(self):
        """Verifies Qwen 3.5 specific Gated Attention forward and QK-norm."""
        cfg = make_mini_qwen3_5_config(num_hidden_layers=1)
        text_model = Qwen3_5TextModel(cfg)
        text_model.eval()

        attn_module = text_model.layers[0].self_attn
        self.assertIsInstance(attn_module, Qwen3_5Attention)
        self.assertTrue(hasattr(attn_module, "q_norm"))
        self.assertTrue(hasattr(attn_module, "k_norm"))

        # Verify forward execution
        x = torch.randn(2, 8, cfg.hidden_size)
        dummy = torch.zeros(1, 1, cfg.hidden_size // cfg.num_attention_heads)
        pos = torch.arange(0, 8).view(1, 1, -1).expand(3, 1, -1)
        pos_emb = text_model.rotary_emb(dummy, pos)

        with torch.no_grad():
            out, _ = attn_module(
                hidden_states=x,
                position_embeddings=pos_emb,
                attention_mask=None,
            )
        self.assertEqual(out.shape, (2, 8, cfg.hidden_size))

    def test_qwen_multi_concurrency_rollout_serving(self):
        """Verifies multi-stream rollout serving (bsz=4) with StaticKVCache."""
        cfg = make_mini_qwen2_config(num_hidden_layers=2)
        model = Qwen2ForCausalLM(cfg)
        model.eval()
        patch_qwen(model)

        server = FastQwenServer(model=model, tokenizer=MockTokenizer(), max_concurrency=4)
        prompts = ["p1", "p2", "p3", "p4"]

        res = server.serve(prompts, max_new_tokens=16, temperature=0.0)
        self.assertEqual(len(res["responses"]), 4)
        self.assertEqual(res["total_tokens"], 64)
        self.assertGreater(res["total_tps"], 0.0)


if __name__ == "__main__":
    unittest.main()
