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
        self.assertEqual(cache.shape, (2, 128, 3))

        # Test single-token decode with cache
        x_tok = torch.randn(2, 1, 128)
        out_tok, new_cache = conv(x_tok, cache=cache, output_final_state=True)
        self.assertEqual(out_tok.shape, (2, 1, 128))
        self.assertEqual(new_cache.shape, (2, 128, 3))

        # Test FusedRMSNormGated
        norm_gate = FusedRMSNormGated(hidden_size=128)
        g = torch.randn(2, 8, 128)
        norm_out = norm_gate(out, g)
        self.assertEqual(norm_out.shape, (2, 8, 128))

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

    def test_fast_ling_server_formatting(self):
        server = FastLingServer(model=None, tokenizer=None)

        prompt_thinking = server.format_prompt("What is 17 * 23?", enable_thinking=True)
        self.assertIn("<role>SYSTEM</role>detailed thinking on<|role_end|>", prompt_thinking)
        self.assertIn("<think>", prompt_thinking)
        self.assertIn("What is 17 * 23?", prompt_thinking)

        prompt_direct = server.format_prompt("What is 17 * 23?", enable_thinking=False)
        self.assertNotIn("<think>", prompt_direct)
        self.assertIn("<role>HUMAN</role>What is 17 * 23?<|role_end|>", prompt_direct)


if __name__ == "__main__":
    unittest.main()
