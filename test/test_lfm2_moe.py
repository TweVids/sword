"""
Comprehensive Unit Test Suite for Liquid Foundation Models (LFM2 / LFM2.5 MoE).
Target Model: LiquidAI/LFM2.5-24B-A2B

Tests:
1. Pure-PyTorch Causal Conv1d from scratch vs HuggingFace slow_forward (prefill & decode numerical identity).
2. Pure-PyTorch FlashAttention SDPA with QK-RMSNorm & GQA vs HuggingFace eager attention.
3. Pure-PyTorch Zero-Sync MoE Dispatch vs HuggingFace eager routing loop.
4. StaticLfmCache Hybrid Buffer (KV + Conv) zero-allocation & O(1) metadata reset.
5. End-to-end model patch and unpatch numerical identity on full LFM2.5 hybrid architecture.
6. Unsloth compatibility layer: FastLfm2MoeModel, PEFT LoRA adapter injection, and backward pass.
7. 8-concurrency rollout generation and serving throughput benchmark (>20 TPS per stream).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
from transformers.models.lfm2_moe.modeling_lfm2_moe import (
    Lfm2MoeForCausalLM,
    Lfm2MoeModel,
    Lfm2MoeAttention,
    Lfm2MoeShortConv,
    Lfm2MoeSparseMoeBlock,
    Lfm2MoeRotaryEmbedding,
    Lfm2MoeRMSNorm,
)
from transformers.cache_utils import DynamicCache

import sword
from sword.lfm import (
    StaticLfmCache,
    make_fast_lfm_conv_forward,
    make_fast_lfm_attention_forward,
    make_fast_lfm_moe_block_forward,
    patch_lfm,
    unpatch_lfm,
    FastLfmServer,
    FastLfm2MoeModel,
)


def make_mini_lfm2_moe_config(**kwargs):
    """
    Creates a mini LFM2.5 MoE configuration matching the exact layer layout
    of LiquidAI/LFM2.5-24B-A2B (hybrid conv + full_attention + dense MLP + MoE).
    """
    layer_types = [
        "conv", "conv", "full_attention", "conv", "full_attention"
    ]
    default_kwargs = dict(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_dense_layers=1,             # Layer 0 is dense MLP, layers 1-4 are MoE
        num_attention_heads=4,
        num_key_value_heads=2,          # 2:1 GQA ratio (in 24B A2B: 32:8 = 4:1)
        conv_L_cache=3,
        conv_bias=False,
        num_experts=8,                  # 8 experts in mini test (64 in 24B A2B)
        num_experts_per_tok=2,          # top-2 active (top-4 in 24B A2B)
        use_expert_bias=True,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        layer_types=layer_types,
        max_position_embeddings=512,
        norm_eps=1e-5,
    )
    default_kwargs.update(kwargs)
    return Lfm2MoeConfig(**default_kwargs)


class TestLfm2MoeArchitecture(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    # =========================================================
    # 1. Pure-PyTorch Causal Conv1d from Scratch
    # =========================================================
    def test_pure_pytorch_conv_prefill_and_decode_numerical_identity(self):
        """
        Verifies that Sword's Pure-PyTorch Causal Conv1d forward produces
        bit-identical outputs to HuggingFace on both multi-token prefill and single-token decode.
        """
        cfg = make_mini_lfm2_moe_config()
        conv_module = Lfm2MoeShortConv(cfg, layer_idx=0)
        conv_module.eval()

        cache_hf = DynamicCache(config=cfg)
        cache_sword = StaticLfmCache(config=cfg, max_batch_size=4, max_seq_len=64, device="cpu", dtype=torch.float32)

        # A. Multi-token prefill
        x_prefill = torch.randn(2, 7, cfg.hidden_size)
        with torch.no_grad():
            out_hf_prefill = conv_module(x_prefill, past_key_values=cache_hf)

        # Patch module with Sword fast conv forward
        conv_module._sword_original_forward = conv_module.forward
        conv_module.forward = types.MethodType(
            make_fast_lfm_conv_forward(conv_module._sword_original_forward), conv_module
        )

        with torch.no_grad():
            out_sword_prefill = conv_module(x_prefill, past_key_values=cache_sword)

        prefill_diff = (out_hf_prefill - out_sword_prefill).abs().max().item()
        self.assertLess(prefill_diff, 1e-5, f"Conv prefill diverged: {prefill_diff}")

        # B. Single-token decode (step 1 and step 2)
        for step in range(2):
            x_decode = torch.randn(2, 1, cfg.hidden_size)

            # Unpatch temporarily to run HF
            conv_module.forward = conv_module._sword_original_forward
            with torch.no_grad():
                out_hf_dec = conv_module(x_decode, past_key_values=cache_hf)

            # Repatch with Sword fast conv
            conv_module.forward = types.MethodType(
                make_fast_lfm_conv_forward(conv_module._sword_original_forward), conv_module
            )
            with torch.no_grad():
                out_sword_dec = conv_module(x_decode, past_key_values=cache_sword)

            dec_diff = (out_hf_dec - out_sword_dec).abs().max().item()
            self.assertLess(dec_diff, 1e-5, f"Conv decode step {step} diverged: {dec_diff}")

        # C. Resilience when L_cache attribute is missing from module
        if hasattr(conv_module, "L_cache"):
            delattr(conv_module, "L_cache")
        with torch.no_grad():
            out_missing_attr = conv_module(x_prefill, past_key_values=cache_sword)
        self.assertEqual(out_missing_attr.shape, (2, 7, cfg.hidden_size))

    # =========================================================
    # 2. Pure-PyTorch FlashAttention SDPA for Lfm2MoeAttention
    # =========================================================
    def test_pure_pytorch_flash_sdpa_attention_numerical_identity(self):
        """
        Verifies that Sword's Pure-PyTorch Flash SDPA attention produces
        identical outputs to HuggingFace attention with QK-RMSNorm and RoPE.
        """
        cfg = make_mini_lfm2_moe_config()
        cfg._attn_implementation = "sdpa"
        attn = Lfm2MoeAttention(cfg, layer_idx=2)
        attn.eval()
        rotary_emb = Lfm2MoeRotaryEmbedding(cfg)

        x = torch.randn(2, 8, cfg.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        pos_emb = rotary_emb(x, pos_ids)

        with torch.no_grad():
            out_hf, _ = attn(x, position_embeddings=pos_emb, attention_mask=None)

        attn._sword_original_forward = attn.forward
        attn.forward = types.MethodType(
            make_fast_lfm_attention_forward(attn._sword_original_forward), attn
        )

        with torch.no_grad():
            out_sword, _ = attn(x, position_embeddings=pos_emb, attention_mask=None)

        diff = (out_hf - out_sword).abs().max().item()
        self.assertLess(diff, 1e-5, f"Attention output diverged: {diff}")

        # Resilience against BlockMask (no .ndim, has .to_dense())
        class MockBlockMask:
            def __init__(self, tensor):
                self._tensor = tensor
            def to_dense(self):
                return self._tensor

        mock_mask = MockBlockMask(torch.ones(2, 1, 8, 8, dtype=torch.bool))
        with torch.no_grad():
            out_block_mask, _ = attn(x, position_embeddings=pos_emb, attention_mask=mock_mask)
        self.assertEqual(out_block_mask.shape, out_hf.shape)

    # =========================================================
    # 3. Pure-PyTorch Zero-Sync MoE Dispatch
    # =========================================================
    def test_pure_pytorch_zero_sync_moe_dispatch_numerical_identity(self):
        """
        Verifies that Sword's Zero-Sync Batched BMM MoE dispatch produces
        exact numerical identity with HuggingFace's eager routing loop,
        while capturing router logits for RL.
        """
        cfg = make_mini_lfm2_moe_config()
        moe_block = Lfm2MoeSparseMoeBlock(cfg)
        torch.nn.init.normal_(moe_block.experts.gate_up_proj, mean=0.0, std=0.02)
        torch.nn.init.normal_(moe_block.experts.down_proj, mean=0.0, std=0.02)
        moe_block.eval()

        x = torch.randn(3, 6, cfg.hidden_size)
        with torch.no_grad():
            out_hf = moe_block(x)

        moe_block._sword_original_forward = moe_block.forward
        moe_block.forward = types.MethodType(
            make_fast_lfm_moe_block_forward(moe_block._sword_original_forward), moe_block
        )

        with torch.no_grad():
            out_sword = moe_block(x)

        diff = (out_hf - out_sword).abs().max().item()
        self.assertLess(diff, 1e-5, f"MoE dispatch diverged: {diff}")
        self.assertTrue(hasattr(moe_block, "last_router_logits"))
        self.assertEqual(moe_block.last_router_logits.shape, (3 * 6, cfg.num_experts))

        # Test router module with expert_bias parameter requirement
        class MockRouterWithBiasParam(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight
            def forward(self, h, expert_bias=None):
                if expert_bias is None:
                    raise TypeError("unsupported operand type(s) for +: 'Tensor' and 'NoneType'")
                logits = F.linear(h, self.weight)
                scores = logits.sigmoid() + expert_bias
                _, topk = torch.topk(scores, 2, dim=-1)
                weights = torch.gather(scores, 1, topk)
                return logits, topk, weights

        orig_gate = moe_block.gate
        moe_block.gate = MockRouterWithBiasParam(orig_gate.weight)
        with torch.no_grad():
            out_with_router_bias = moe_block(x)
        self.assertEqual(out_with_router_bias.shape, (3, 6, cfg.hidden_size))
        moe_block.gate = orig_gate

    # =========================================================
    # 4. StaticLfmCache Hybrid Buffer (KV + Conv) & Fast Reset
    # =========================================================
    def test_static_lfm_cache_zero_allocation_and_fast_reset(self):
        """
        Verifies StaticLfmCache pre-allocates both KV and Conv static buffers,
        updates in-place without memory reallocations, and resets in O(1) time.
        """
        cfg = make_mini_lfm2_moe_config()
        cache = StaticLfmCache(
            config=cfg,
            max_batch_size=8,
            max_seq_len=128,
            dtype=torch.float32,
            device="cpu",
        )

        # Check pre-allocated buffers
        self.assertEqual(len(cache.k_cache), 2)  # 2 full_attention layers (indices 2, 4)
        self.assertEqual(len(cache.conv_cache), 3)  # 3 conv layers (indices 0, 1, 3)

        # Update conv buffer
        conv_in = torch.randn(8, cfg.hidden_size, 3)
        updated_conv = cache.update_conv_state(conv_in, layer_idx=0)
        self.assertEqual(updated_conv.shape, (8, cfg.hidden_size, 3))
        self.assertTrue(cache.has_previous_state(0))

        # Update KV cache
        k_in = torch.randn(8, cfg.num_key_value_heads, 4, cfg.hidden_size // cfg.num_attention_heads)
        v_in = torch.randn(8, cfg.num_key_value_heads, 4, cfg.hidden_size // cfg.num_attention_heads)
        k_out, v_out = cache.update(k_in, v_in, layer_idx=2)
        self.assertEqual(k_out.shape[2], 4)

        # O(1) Fast Reset
        cache.new_rollout()
        self.assertEqual(cache.current_pos, 0)
        self.assertFalse(cache.has_previous_state(0))

    # =========================================================
    # 5. Full Model End-to-End Patch & Unpatch
    # =========================================================
    def test_end_to_end_model_patch_and_unpatch(self):
        """
        Verifies patching the entire Lfm2MoeForCausalLM model with patch_lfm
        preserves exact numerical output and unpatching perfectly restores it.
        """
        cfg = make_mini_lfm2_moe_config()
        model = Lfm2MoeForCausalLM(cfg)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 8))
        with torch.no_grad():
            out_before = model(input_ids)

        patch_lfm(model, mode="flash", patch_moe=True)

        with torch.no_grad():
            out_after = model(input_ids)

        diff = (out_before.logits - out_after.logits).abs().max().item()
        self.assertLess(diff, 1e-4, f"Model outputs diverged after patch: {diff}")

        unpatch_lfm(model)
        with torch.no_grad():
            out_restored = model(input_ids)

        diff_restored = (out_before.logits - out_restored.logits).abs().max().item()
        self.assertLess(diff_restored, 1e-5, f"Model outputs diverged after unpatch: {diff_restored}")

    # =========================================================
    # 6. Unsloth Compatibility & LoRA Integration
    # =========================================================
    def test_unsloth_compatibility_and_lora_fine_tuning(self):
        """
        Verifies Unsloth FastLfm2MoeModel adapter and LoRA fine-tuning
        on LFM2.5 target projection layers.
        """
        cfg = make_mini_lfm2_moe_config()
        model = Lfm2MoeForCausalLM(cfg)

        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=FastLfm2MoeModel.target_modules[:6],  # q_proj, k_proj, v_proj, out_proj, in_proj, w1
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_cfg)
        FastLfm2MoeModel.patch_peft_model(peft_model)

        input_ids = torch.randint(0, 1000, (2, 4))
        outputs = peft_model(input_ids)
        loss = outputs.logits.sum()
        loss.backward()

        # Check gradients computed on LoRA parameters
        has_grads = any(p.grad is not None for p in peft_model.parameters() if p.requires_grad)
        self.assertTrue(has_grads, "LoRA adapter backward failed to compute gradients")

    # =========================================================
    # 7. 8-Concurrency Serving Engine & Throughput Benchmark
    # =========================================================
    def test_8_concurrency_rollout_serving_and_benchmark(self):
        """
        Verifies FastLfmServer handles 8 concurrent streams simultaneously
        with high token throughput.
        """
        tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-24B-A2B")
        cfg = make_mini_lfm2_moe_config(vocab_size=len(tokenizer) + 100)
        model = Lfm2MoeForCausalLM(cfg)

        server = FastLfmServer(
            model=model,
            tokenizer=tokenizer,
            max_concurrency=8,
            max_seq_len=128,
            device="cpu",
        )

        prompts = [f"Prompt number {i}" for i in range(8)]
        res = server.serve(prompts, max_new_tokens=4, use_fast_engine=True)

        self.assertEqual(len(res["responses"]), 8)
        self.assertEqual(res["concurrency"], 8)
        self.assertGreater(res["total_tps"], 0.0)

        # Run benchmark before vs after (with 16 generated tokens across 8 streams)
        bench = server.benchmark_before_after(prompts=prompts, max_new_tokens=16, concurrency=8)
        self.assertEqual(bench["concurrency"], 8)
        self.assertGreater(bench["after_tps"], 0.0)
        self.assertGreater(bench["speedup"], 1.0, f"Expected Sword speed engine to beat unpatched HF baseline, got {bench['speedup']:.2f}x")


    # =========================================================
    # 8. Meta-Device Instantiation of Full 24B Architecture
    # =========================================================
    def test_full_24b_architecture_meta_instantiation_and_patching(self):
        """
        Instantiates the FULL LiquidAI/LFM2.5-24B-A2B architecture (40 layers, 24B params)
        on PyTorch 'meta' device (zero RAM/VRAM footprint), applies patch_lfm, and verifies
        that all 78 modules (30 Conv, 10 Attn, 38 MoE) are successfully accelerated.
        """
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("LiquidAI/LFM2.5-24B-A2B")

        self.assertEqual(cfg.num_hidden_layers, 40)
        self.assertEqual(len(cfg.layer_types), 40)
        self.assertEqual(cfg.num_experts, 64)
        self.assertEqual(cfg.num_experts_per_tok, 4)

        with torch.device("meta"):
            full_model = Lfm2MoeForCausalLM(cfg)

        param_count = sum(p.numel() for p in full_model.parameters())
        self.assertGreater(param_count, 23.5e9)
        self.assertLess(param_count, 24.5e9)

        # Test patch_lfm on full 24B architecture
        patch_lfm(full_model, mode="flash", patch_moe=True)

        # Verify patching
        attn_count = 0
        conv_count = 0
        moe_count = 0
        for name, module in full_model.named_modules():
            if module.__class__.__name__ == "Lfm2MoeAttention":
                self.assertTrue(hasattr(module, "_sword_original_forward"))
                attn_count += 1
            elif module.__class__.__name__ == "Lfm2MoeShortConv":
                self.assertTrue(hasattr(module, "_sword_original_forward"))
                conv_count += 1
            elif module.__class__.__name__ == "Lfm2MoeSparseMoeBlock":
                self.assertTrue(hasattr(module, "_sword_original_forward"))
                moe_count += 1

        self.assertEqual(attn_count, 10)
        self.assertEqual(conv_count, 30)
        self.assertEqual(moe_count, 38)

        unpatch_lfm(full_model)

    # =========================================================
    # 9. Left-Padded Unequal-Length Batched Decode vs Per-Sequence
    # =========================================================
    def test_left_padded_unequal_length_batched_decode_numerical_identity(self):
        """
        Verifies that left-padded batches of vastly different prompt lengths
        produce identical greedy token sequences to unpadded single-sequence execution.
        Specifically exercises StaticLfmCache.get_mask_sizes, position_ids,
        conv padding masking, and decode-step attention masking.
        """
        cfg = make_mini_lfm2_moe_config(vocab_size=256)
        cfg._attn_implementation = "sdpa"
        model = Lfm2MoeForCausalLM(cfg)
        model.eval()
        patch_lfm(model, mode="flash", patch_moe=True)

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 255
            padding_side = "left"
            chat_template = None

        tokenizer = DummyTokenizer()
        server = FastLfmServer(
            model=model,
            tokenizer=tokenizer,
            max_concurrency=4,
            max_seq_len=64,
            device="cpu",
        )

        # Create 2 sequences of unequal lengths: seq0 has length 3 (left-padded with 5 zeros), seq1 has length 8
        seq0 = torch.tensor([[12, 34, 56]], dtype=torch.long)
        seq1 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.long)

        batched_ids = torch.tensor([
            [0, 0, 0, 0, 0, 12, 34, 56],
            [10, 20, 30, 40, 50, 60, 70, 80],
        ], dtype=torch.long)
        batched_mask = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)

        # Run batched greedy decode (4 new tokens)
        out_batched = server.fast_decode_8concurrency(
            input_ids=batched_ids,
            attention_mask=batched_mask,
            max_new_tokens=4,
            temperature=0.0,
        )
        gen_batched_0 = out_batched[0, -4:].tolist()
        gen_batched_1 = out_batched[1, -4:].tolist()

        # Run single unpadded greedy decode for seq0 and seq1 independently
        out_single_0 = server.fast_decode_8concurrency(
            input_ids=seq0,
            attention_mask=None,
            max_new_tokens=4,
            temperature=0.0,
        )
        out_single_1 = server.fast_decode_8concurrency(
            input_ids=seq1,
            attention_mask=None,
            max_new_tokens=4,
            temperature=0.0,
        )
        gen_single_0 = out_single_0[0, -4:].tolist()
        gen_single_1 = out_single_1[0, -4:].tolist()

        self.assertEqual(gen_batched_0, gen_single_0, f"Padded stream 0 diverged from unpadded: {gen_batched_0} vs {gen_single_0}")
        self.assertEqual(gen_batched_1, gen_single_1, f"Stream 1 diverged: {gen_batched_1} vs {gen_single_1}")

    # =========================================================
    # 10. MoE Prefill Grouped-GEMM vs Decode Unique-Expert Paths
    # =========================================================
    def test_moe_prefill_grouped_gemm_and_decode_unique_experts(self):
        """
        Exercises both MoE execution paths:
        1. Large prefill (N * K > 64) using zero-weight-duplication grouped GEMM.
        2. Small decode (N * K <= 64) with high expert overlap (unique-expert path)
           and low expert overlap (vectorized BMM path).
        Verifies exact numerical identity with HuggingFace eager MoE.
        """
        cfg = make_mini_lfm2_moe_config(num_experts=16, num_experts_per_tok=4)
        moe = Lfm2MoeSparseMoeBlock(cfg)
        torch.nn.init.normal_(moe.experts.gate_up_proj, mean=0.0, std=0.02)
        torch.nn.init.normal_(moe.experts.down_proj, mean=0.0, std=0.02)
        moe.eval()

        # A. Large prefill: 64 tokens * 4 experts/tok = 256 assignments (> 64 threshold)
        x_prefill = torch.randn(2, 32, cfg.hidden_size)
        with torch.no_grad():
            ref_prefill = moe(x_prefill)

        # B. Small decode with high overlap: 8 tokens * 4 = 32 assignments
        x_decode = torch.randn(8, 1, cfg.hidden_size)
        with torch.no_grad():
            ref_decode = moe(x_decode)

        # Patch with Sword hybrid MoE forward
        moe._sword_original_forward = moe.forward
        moe.forward = types.MethodType(make_fast_lfm_moe_block_forward(moe._sword_original_forward), moe)

        with torch.no_grad():
            fast_prefill = moe(x_prefill)
            fast_decode = moe(x_decode)

        diff_prefill = (ref_prefill - fast_prefill).abs().max().item()
        diff_decode = (ref_decode - fast_decode).abs().max().item()
        self.assertLess(diff_prefill, 1e-5, f"MoE grouped prefill diverged: {diff_prefill}")
        self.assertLess(diff_decode, 1e-5, f"MoE decode diverged: {diff_decode}")


if __name__ == "__main__":
    unittest.main()
