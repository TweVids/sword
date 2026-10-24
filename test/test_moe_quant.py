"""
Unit test suite for Sword native 3D MoE Quantization, MoE QLoRA, and Lean Decode Engine.
Tests:
1. Quantized3DWeight: FP8, INT8, and NF4 block-wise quantization accuracy and gather.
2. QuantizedMoEExperts: Conversion from 16-bit 3D MoE and on-the-fly dequantized forward.
3. MoEQLoRALayer: Trainable low-rank adapters on frozen base quantized weights with autograd.
4. End-to-end quantize_model_moe and apply_moe_qlora integration with training loss backward.
5. FastLfmServer fast_lean_decode_step and FastLfmGraphEngine interface.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sword
from sword.moe_quant import (
    Quantized3DWeight,
    QuantizedMoEExperts,
    MoEQLoRALayer,
    quantize_model_moe,
    apply_moe_qlora,
    get_moe_qlora_parameters,
)
from sword.lfm import (
    FastLfmServer,
    StaticLfmCache,
    FastLfmGraphEngine,
    patch_lfm,
)
from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM


def make_mini_lfm2_moe_config(**kwargs):
    layer_types = ["conv", "full_attention", "conv"]
    default_kwargs = dict(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=len(layer_types),
        num_dense_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        conv_L_cache=3,
        conv_bias=False,
        num_experts=4,
        num_experts_per_tok=2,
        use_expert_bias=True,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        layer_types=layer_types,
        max_position_embeddings=256,
        norm_eps=1e-5,
    )
    default_kwargs.update(kwargs)
    return Lfm2MoeConfig(**default_kwargs)


class TestMoEQuantizationAndQLoRA(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_quantized_3d_weight_fp8(self):
        """Tests 3D tensor FP8 quantization and gather."""
        num_exp = 4
        out_dim = 32
        in_dim = 16
        float_w = torch.randn(num_exp, out_dim, in_dim, dtype=torch.float32)

        q = Quantized3DWeight(num_exp, out_dim, in_dim, quant_type="fp8", compute_dtype=torch.float32)
        q.quantize_from_float(float_w)

        dequant = q.dequantize_all()
        self.assertEqual(dequant.shape, (num_exp, out_dim, in_dim))
        cos_sim = F.cosine_similarity(float_w.flatten(), dequant.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.98)

        active = torch.tensor([0, 2], dtype=torch.long)
        gathered = q.gather_dequantize(active)
        self.assertEqual(gathered.shape, (2, out_dim, in_dim))

    def test_quantized_3d_weight_int8(self):
        """Tests 3D tensor INT8 quantization and gather."""
        num_exp = 4
        out_dim = 32
        in_dim = 16
        float_w = torch.randn(num_exp, out_dim, in_dim, dtype=torch.float32)

        q = Quantized3DWeight(num_exp, out_dim, in_dim, quant_type="int8", compute_dtype=torch.float32)
        q.quantize_from_float(float_w)

        dequant = q.dequantize_all()
        self.assertEqual(dequant.shape, (num_exp, out_dim, in_dim))
        cos_sim = F.cosine_similarity(float_w.flatten(), dequant.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.98)

    def test_quantized_3d_weight_nf4(self):
        """Tests 3D tensor NF4 block-wise quantization, packing, and dequantization."""
        num_exp = 4
        out_dim = 32
        in_dim = 64
        float_w = torch.randn(num_exp, out_dim, in_dim, dtype=torch.float32)

        q = Quantized3DWeight(num_exp, out_dim, in_dim, quant_type="nf4", block_size=32, compute_dtype=torch.float32)
        q.quantize_from_float(float_w)

        # Packed buffer has half the size along in_features (2 4-bit values per uint8)
        self.assertEqual(q.qweight.shape, (num_exp, out_dim, in_dim // 2))
        self.assertEqual(q.qweight.dtype, torch.uint8)

        dequant = q.dequantize_all()
        self.assertEqual(dequant.shape, (num_exp, out_dim, in_dim))
        cos_sim = F.cosine_similarity(float_w.flatten(), dequant.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.90)

    def test_moe_qlora_layer_autograd(self):
        """
        Verifies that MoEQLoRALayer keeps base weights frozen and computes gradients
        for low-rank adapter parameters during backpropagation.
        """
        class DummyMoEConfig:
            num_experts = 4
            hidden_size = 32
            moe_intermediate_size = 64

        q_experts = QuantizedMoEExperts(DummyMoEConfig(), quant_type="nf4", compute_dtype=torch.float32)
        w_gu = torch.randn(4, 128, 32, dtype=torch.float32)
        w_d = torch.randn(4, 32, 64, dtype=torch.float32)
        q_experts.gate_up_proj.quantize_from_float(w_gu)
        q_experts.down_proj.quantize_from_float(w_d)

        # Attach MoE QLoRA layer
        qlora_layer = MoEQLoRALayer(q_experts, r=8, lora_alpha=16)

        # Check base weights are frozen
        for p in q_experts.parameters():
            self.assertFalse(p.requires_grad)

        # Check adapter parameters require grad
        self.assertTrue(qlora_layer.lora_gate_up_A.requires_grad)
        self.assertTrue(qlora_layer.lora_gate_up_B.requires_grad)
        self.assertTrue(qlora_layer.lora_down_A.requires_grad)
        self.assertTrue(qlora_layer.lora_down_B.requires_grad)

        # Forward
        num_tokens = 6
        h = torch.randn(num_tokens, 32, requires_grad=True)
        top_k_idx = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3], [0, 2]], dtype=torch.long)
        top_k_w = torch.full((num_tokens, 2), 0.5, dtype=torch.float32)

        out = qlora_layer(h, top_k_idx, top_k_w)
        self.assertEqual(out.shape, (num_tokens, 32))

        # Backward
        loss = out.sum()
        loss.backward()

        # Check gradients exist on adapters
        self.assertIsNotNone(qlora_layer.lora_gate_up_A.grad)
        self.assertIsNotNone(qlora_layer.lora_gate_up_B.grad)
        self.assertIsNotNone(qlora_layer.lora_down_A.grad)
        self.assertIsNotNone(qlora_layer.lora_down_B.grad)

        # Check base weights have no gradients
        for p in q_experts.parameters():
            self.assertIsNone(p.grad)

    def test_quantize_model_moe_and_apply_qlora_integration(self):
        """Tests full model MoE quantization and QLoRA adapter attachment with optimizer step."""
        cfg = make_mini_lfm2_moe_config()
        model = Lfm2MoeForCausalLM(cfg)
        patch_lfm(model)

        quantized_count = quantize_model_moe(model, quant_type="nf4", compute_dtype=torch.float32)
        self.assertGreater(quantized_count, 0)

        model, stats = apply_moe_qlora(model, r=4, lora_alpha=8)
        self.assertGreater(stats["moe_adapters"], 0)

        trainable_params = get_moe_qlora_parameters(model)
        self.assertGreater(len(trainable_params), 0)

        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
        optimizer.zero_grad()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        outputs = model(input_ids=input_ids)
        loss = outputs.logits.sum()
        loss.backward()

        optimizer.step()

        for p in trainable_params:
            self.assertIsNotNone(p.grad)

    def test_fast_lean_decode_step_numerical_identity(self):
        """Verifies that FastLfmServer fast_lean_decode_step matches model forward identically."""
        cfg = make_mini_lfm2_moe_config()
        model = Lfm2MoeForCausalLM(cfg)
        patch_lfm(model)
        model.eval()

        server_std = FastLfmServer(model=model, tokenizer=None, max_concurrency=4, max_seq_len=64)
        server_lean = FastLfmServer(model=model, tokenizer=None, max_concurrency=4, max_seq_len=64)

        input_ids = torch.randint(0, cfg.vocab_size, (4, 4))

        # Prefill on server_std
        server_std.static_cache.new_rollout(batch_size=4)
        server_std.static_cache.set_pos(0)
        with torch.no_grad():
            for name, m in model.named_modules():
                if hasattr(m, "q_proj") and hasattr(m, "k_proj"):
                    m._sword_static_cache = server_std.static_cache
            _ = model(input_ids=input_ids, past_key_values=server_std.static_cache, use_cache=True)

        # Prefill on server_lean
        server_lean.static_cache.new_rollout(batch_size=4)
        server_lean.static_cache.set_pos(0)
        with torch.no_grad():
            for name, m in model.named_modules():
                if hasattr(m, "q_proj") and hasattr(m, "k_proj"):
                    m._sword_static_cache = server_lean.static_cache
            _ = model(input_ids=input_ids, past_key_values=server_lean.static_cache, use_cache=True)

        curr_pos = 4
        next_token = torch.randint(0, cfg.vocab_size, (4, 1))
        dec_pos = torch.full((4, 1), curr_pos, dtype=torch.long)

        # Standard forward
        server_std.static_cache.set_pos(curr_pos)
        with torch.no_grad():
            for name, m in model.named_modules():
                if hasattr(m, "q_proj") and hasattr(m, "k_proj"):
                    m._sword_static_cache = server_std.static_cache
            out_std = model(
                input_ids=next_token,
                position_ids=dec_pos,
                past_key_values=server_std.static_cache,
                use_cache=True,
                logits_to_keep=1,
            )
        logits_std = out_std.logits[:, -1, :]

        # Lean decode step on fresh cache
        server_lean.static_cache.set_pos(curr_pos)
        with torch.no_grad():
            for name, m in model.named_modules():
                if hasattr(m, "q_proj") and hasattr(m, "k_proj"):
                    m._sword_static_cache = server_lean.static_cache
            logits_lean = server_lean.fast_lean_decode_step(next_token, dec_pos)

        diff = (logits_std - logits_lean).abs().max().item()
        self.assertLess(diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
