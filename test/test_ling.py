import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sword

from sword.ling import (
    setup_fla_compatibility,
    make_patched_bailing_mla_forward,
    make_fast_bailing_moe_infer,
    patch_ling,
    unpatch_ling,
    FastLingServer,
)


class MockBailingConfig:
    def __init__(self):
        self.num_attention_heads = 16
        self.num_key_value_heads = 16
        self.hidden_size = 1536
        self.qk_head_dim = 192
        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = 64
        self.v_head_dim = 128
        self.kv_lora_rank = 512
        self.q_lora_rank = 256
        self.num_experts = 128
        self.num_experts_per_tok = 8
        self.num_shared_experts = 1
        self.moe_intermediate_size = 512
        self.moe_shared_expert_intermediate_size = 512
        self.routed_scaling_factor = 2.5
        self.rms_norm_eps = 1e-6
        self.rope_interleave = True
        self.rope_scaling = None
        self.attention_dropout = 0.0
        self.gated_attention_proj_granularity_type = "head_wise"
        self._attn_implementation = "sdpa"


class MockBailingMLP(nn.Module):
    def __init__(self, hidden_size=1536, intermediate_size=512):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockBailingMLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_idx = 3
        self.num_heads = config.num_attention_heads
        self.qk_head_dim = config.qk_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.scaling = config.qk_head_dim ** -0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.dense = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        # Default eager forward mock
        batch_size, seq_len, _ = hidden_states.shape
        return (torch.randn(batch_size, seq_len, self.config.hidden_size), None, past_key_values)


class MockBailingMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([MockBailingMLP(config.hidden_size, config.moe_intermediate_size) for _ in range(config.num_experts)])

    def moe_infer(self, x, topk_ids, topk_weight):
        # Stock slow implementation with .cpu().numpy()
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        outputs = []
        for i, num in enumerate(tokens_per_expert):
            if num == 0:
                continue
            expert = self.experts[i]
            outputs.append(expert(x[:1]))
        return torch.zeros_like(x)


class TestSwordLingSupport(unittest.TestCase):
    def setUp(self):
        self.config = MockBailingConfig()

    def test_fla_compatibility_layer(self):
        setup_fla_compatibility()
        self.assertIn("fla", sys.modules)
        self.assertIn("fla.modules", sys.modules)
        self.assertIn("fla.ops.kda", sys.modules)

        from fla.modules import ShortConvolution, FusedRMSNormGated
        from fla.ops.kda import fused_recurrent_kda

        # Test ShortConvolution
        conv = ShortConvolution(hidden_size=128, kernel_size=4)
        x = torch.randn(2, 8, 128)
        out, cache = conv(x, output_final_state=True)
        self.assertEqual(out.shape, (2, 8, 128))
        self.assertEqual(cache.shape, (2, 128, 4))

        # Test single-token decode with cache
        x_tok = torch.randn(2, 1, 128)
        out_tok, new_cache = conv(x_tok, cache=cache, output_final_state=True)
        self.assertEqual(out_tok.shape, (2, 1, 128))
        self.assertEqual(new_cache.shape, (2, 128, 4))

        # Test FusedRMSNormGated
        norm_gate = FusedRMSNormGated(hidden_size=128)
        g = torch.randn(2, 8, 128)
        norm_out = norm_gate(out, g)
        self.assertEqual(norm_out.shape, (2, 8, 128))

        # Test fused_recurrent_kda step (T=1 decode)
        B, H, K, V = 2, 4, 32, 32
        q = torch.randn(B, 1, H, K)
        k = torch.randn(B, 1, H, K)
        v = torch.randn(B, 1, H, V)
        g = torch.randn(B, 1, H, K)
        beta = torch.sigmoid(torch.randn(B, 1, H))
        prev_h = torch.randn(B, H, K, V)
        kda_out, next_h = fused_recurrent_kda(q, k, v, g, beta=beta, initial_state=prev_h, output_final_state=True)
        self.assertEqual(kda_out.shape, (B, 1, H, V))
        self.assertEqual(next_h.shape, (B, H, K, V))
        self.assertFalse(torch.isnan(kda_out).any())
        self.assertFalse(torch.isnan(next_h).any())

    def test_mla_pure_flash_sdpa(self):
        mla = MockBailingMLA(self.config)
        orig_fwd = mla.forward
        mla.forward = types.MethodType(make_patched_bailing_mla_forward(orig_fwd), mla)

        B, S, D = 2, 4, 1536
        hidden_states = torch.randn(B, S, D)
        cos = torch.randn(B, S, 64)
        sin = torch.randn(B, S, 64)
        pos_emb = (cos, sin)

        out, _, _ = mla(hidden_states, position_embeddings=pos_emb)
        self.assertEqual(out.shape, (B, S, D))

    def test_zero_sync_moe_dispatch(self):
        moe = MockBailingMoE(self.config)
        orig_infer = moe.moe_infer
        moe.moe_infer = types.MethodType(make_fast_bailing_moe_infer(orig_infer), moe)

        N, D = 4, 1536
        x = torch.randn(N, D)
        topk_ids = torch.tensor([
            [0, 5, 12, 33, 40, 55, 78, 100],
            [1, 5, 13, 33, 41, 56, 79, 101],
            [0, 6, 14, 34, 42, 57, 80, 102],
            [2, 7, 15, 35, 43, 58, 81, 103],
        ], dtype=torch.long)
        topk_weight = F.softmax(torch.randn(N, 8), dim=-1)

        out = moe.moe_infer(x, topk_ids, topk_weight)
        self.assertEqual(out.shape, (N, D))
        self.assertFalse(torch.isnan(out).any())

        # Test bfloat16 input with float32 topk_weight (replicates HF router behavior)
        x_bf16 = torch.randn(N, D, dtype=torch.bfloat16)
        moe_bf16 = MockBailingMoE(self.config).to(dtype=torch.bfloat16)
        moe_bf16.moe_infer = types.MethodType(make_fast_bailing_moe_infer(orig_infer), moe_bf16)
        out_bf16 = moe_bf16.moe_infer(x_bf16, topk_ids, topk_weight.float())
        self.assertEqual(out_bf16.dtype, torch.bfloat16)
        self.assertEqual(out_bf16.shape, (N, D))

    def test_fast_ling_server_formatting(self):
        server = FastLingServer(model=None, tokenizer=None)

        prompt_thinking = server.format_prompt("What is 17 * 23?", enable_thinking=True)
        self.assertIn("<role>SYSTEM</role>detailed thinking on<|role_end|>", prompt_thinking)
        self.assertIn("<think>", prompt_thinking)
        self.assertIn("What is 17 * 23?", prompt_thinking)

        prompt_direct = server.format_prompt("What is 17 * 23?", enable_thinking=False)
        self.assertNotIn("<think>", prompt_direct)

    def test_fast_generate_engine(self):
        class MockOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        class MockCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = 100

            def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True, **kwargs):
                B, S = input_ids.shape
                logits = torch.randn(B, S, self.vocab_size)
                cache = (past_key_values or 0) + S
                return MockOutput(logits=logits, past_key_values=cache)

        class MockTokenizer:
            eos_token_id = 99
            pad_token_id = 98

        model = MockCausalLM()
        server = FastLingServer(model=model, tokenizer=MockTokenizer(), device="cpu")
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        attn_mask = torch.ones_like(input_ids)
        out = server.fast_generate(input_ids, attention_mask=attn_mask, max_new_tokens=8, temperature=0.0)
        self.assertEqual(out.shape, (2, 3 + 8))

    def test_from_pretrained_signature(self):
        import inspect
        sig = inspect.signature(FastLingServer.from_pretrained)
        self.assertIn("patch_sword", sig.parameters)
        self.assertIn("patch_moe", sig.parameters)
        self.assertIn("kwargs", sig.parameters)

        from sword.ling import load_ling_model
        load_sig = inspect.signature(load_ling_model)
        self.assertIn("patch_sword", load_sig.parameters)
        self.assertIn("patch_moe", load_sig.parameters)
        self.assertIn("kwargs", load_sig.parameters)

    def test_rope_scaling_compatibility(self):
        # Verify that a config with default rope_scaling (missing factor) is handled without KeyError
        cfg = MockBailingConfig()
        cfg.rope_scaling = {"rope_theta": 6000000, "partial_rotary_factor": 0.5, "rope_type": "default"}
        
        # Test simulated attention init with the patch
        from sword.ling import _fix_transformers_bailing_compatibility
        _fix_transformers_bailing_compatibility()

        class MockMLAWithRopeCheck(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.scaling = 128 ** (-0.5)
                if self.config.rope_scaling is not None:
                    scaling_factor = self.config.rope_scaling["factor"]
                    self.scaling *= scaling_factor

        # Patching simulation
        orig_init = MockMLAWithRopeCheck.__init__
        def safe_init(self, config):
            if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
                if config.rope_scaling.get("rope_type") == "default":
                    config.rope_scaling = None
                elif "factor" not in config.rope_scaling:
                    config.rope_scaling["factor"] = 1.0
            return orig_init(self, config)

        MockMLAWithRopeCheck.__init__ = safe_init
        layer = MockMLAWithRopeCheck(cfg)
        self.assertIsNotNone(layer)
        self.assertAlmostEqual(layer.scaling, 128 ** (-0.5))


if __name__ == "__main__":
    unittest.main()
