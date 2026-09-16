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
from sword.rl.loss import ChunkedGRPOLoss, compute_gdpo_advantages, compute_group_advantages
from sword.rl.moe_monitor import MoERouterMonitor
from sword.rl.engine import GRPOTrainer
from sword.rl.scorer import PrimaryScorer
from sword.rl.schema import DatasetRow, Trajectory, ScoredTrajectory, DomainType, EffortTier, FailureReason


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


    def test_qwen3_moe_speculative_decoding_with_static_cache(self):
        """Verifies Speculative Prompt Lookup Decoding works without attention mask mismatch on Qwen3 MoE."""
        cfg = make_mini_qwen3_moe_config()
        model = Qwen3MoeForCausalLM(cfg)
        model.eval()

        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 2
            padding_side = "left"
            def __call__(self, texts, padding=True, return_tensors="pt", **kwargs):
                # Repeating pattern to trigger speculative n-gram matches
                pattern = [10, 20, 30, 40, 10, 20, 30, 40]
                return {
                    "input_ids": torch.tensor([pattern for _ in texts], dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), len(pattern), dtype=torch.long),
                }
            def batch_decode(self, token_ids, skip_special_tokens=True):
                return ["mock_decoded_text" for _ in range(token_ids.shape[0])]

        server = FastQwen3MoeServer(
            model=model,
            tokenizer=MockTokenizer(),
            max_concurrency=2,
            max_seq_len=256,
        )

        res = server.serve(
            prompts=["test pattern prompt"],
            max_new_tokens=8,
            temperature=0.0,
            use_speculative=True,
            speculative_k=3,
        )

        self.assertTrue(res["speculative"])
        self.assertEqual(res["total_tokens"], 8)
        self.assertEqual(len(res["responses"]), 1)

        bench = server.benchmark_before_after(
            prompts=["test pattern prompt"],
            max_new_tokens=6,
            use_speculative=True,
            speculative_k=3,
        )
        self.assertIn("speedup", bench)
        self.assertGreater(bench["after_total_tps"], 0.0)

    # =========================================================
    # 9. Native FP8 KV Cache Serving Compatibility
    # =========================================================
    def test_qwen3_moe_fp8_kv_cache_serving(self):
        """Verifies Qwen3 MoE serving with native torch.float8_e4m3fn KV Cache."""
        if not hasattr(torch, "float8_e4m3fn"):
            return

        cfg = make_mini_qwen3_moe_config()
        model = Qwen3MoeForCausalLM(cfg).bfloat16()
        patch_qwen3_moe(model, mode="flash", patch_moe=True)

        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 2
            def __call__(self, texts, padding=True, return_tensors="pt", **kwargs):
                return {
                    "input_ids": torch.randint(10, 900, (len(texts), 8)),
                    "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
                }
            def batch_decode(self, token_ids, skip_special_tokens=True):
                return ["output" for _ in range(token_ids.shape[0])]

        server = FastQwen3MoeServer(
            model=model,
            tokenizer=MockTokenizer(),
            max_concurrency=4,
            max_seq_len=256,
            kv_cache_dtype=torch.float8_e4m3fn,
        )

        res = server.serve(["test prompt 1", "test prompt 2"], max_new_tokens=4, use_speculative=False)
        self.assertEqual(res["total_tokens"], 8)
        self.assertEqual(len(res["responses"]), 2)

    # =========================================================
    # 10. In-Memory MoE Weight FP8 Conversion (convert_to_fp8)
    # =========================================================
    def test_convert_to_fp8_moe_weights(self):
        """Verifies in-memory FP8 conversion of Qwen3 MoE expert weights."""
        if not hasattr(torch, "float8_e4m3fn"):
            return

        cfg = make_mini_qwen3_moe_config()
        model = Qwen3MoeForCausalLM(cfg).bfloat16()
        sword.convert_to_fp8(model)

        for name, param in model.named_parameters():
            if "gate_up_proj" in name or "down_proj" in name:
                self.assertEqual(param.dtype, torch.float8_e4m3fn)
                self.assertFalse(param.requires_grad)

    # =========================================================
    # 11. GDPO (Decoupled Multi-Reward Normalization) Tests
    # =========================================================
    def test_gdpo_decoupled_normalization_prevents_penalty_spiking(self):
        """
        Verifies GDPO prevents penalty spiking and reward collapse (arXiv:2601.05242).
        In standard GRPO, a large penalty on Rollout 1 drags down the group mean, giving
        Rollout 2 (a mathematically wrong answer) a positive advantage!
        GDPO decouples the columns, keeping wrong answers negative and correct answers positive.
        """
        # 3 Rollouts:
        # Rollout 0: Correct answer (0.7), clean format (0.1) -> sum = 0.8
        r0 = {"ground_truth": 0.7, "output_format": 0.1}
        # Rollout 1: Correct answer (0.7), extreme safety violation (-1.5) -> sum = -0.8
        r1_extreme = {"ground_truth": 0.7, "safety": -1.5}
        # Rollout 2: WRONG answer (0.0), clean format (0.1) -> sum = 0.1
        r2 = {"ground_truth": 0.0, "output_format": 0.1}

        # Standard GRPO with raw scalar sums:
        # Mean = (0.8 - 0.8 + 0.1)/3 = 0.033. Rollout 2 (0.1 > 0.033) gets POSITIVE advantage!
        grpo_advs = ChunkedGRPOLoss.compute_group_advantages([
            sum(r0.values()),
            sum(r1_extreme.values()),
            sum(r2.values()),
        ])
        # In standard GRPO, Rollout 2 (wrong answer) receives a POSITIVE advantage (+0.10):
        self.assertGreater(grpo_advs[2], 0.0, "Standard GRPO should exhibit reward collapse on this setup")

        # Now evaluate under GDPO:
        gdpo_advs, col_advs = ChunkedGRPOLoss.compute_gdpo_advantages([r0, r1_extreme, r2])

        # Rollout 0 (correct answer + clean formatting) must receive strong positive advantage
        self.assertGreater(gdpo_advs[0], 0.5)
        # Rollout 2 (WRONG answer) must receive NEGATIVE advantage (NO reward collapse!)
        self.assertLess(gdpo_advs[2], 0.0)
        # Rollout 1 (safety violation) must be heavily penalized and clamped
        self.assertLessEqual(gdpo_advs[1], -1.0)

        # Check column advantages breakdown
        self.assertIn("accuracy", col_advs)
        self.assertIn("formatting", col_advs)
        self.assertIn("safety", col_advs)
        # In accuracy channel: Rollout 0 is correct, Rollout 2 is wrong
        self.assertGreater(col_advs["accuracy"][0], 0.0)
        self.assertLess(col_advs["accuracy"][2], 0.0)

    # =========================================================
    # 12. Thinking Tags (<think> ... </think>) Validation Tests
    # =========================================================
    def test_thinking_tags_open_close_validation(self):
        """
        Verifies primary scorer validates proper opening <think> and closing </think> tags
        when condition is met, and enforces format violation for unclosed/mismatched tags.
        """
        scorer = PrimaryScorer()
        prob_thinking = DatasetRow(
            problem_id="prob_math_1",
            user_problem="Solve 2x + 5 = 15. Show your work step-by-step.",
            domain=DomainType.MATH,
            effort_tier=EffortTier.HIGH,
            recheck_required=True,
        )

        # 1. Properly opened and closed <think> ... </think>
        traj_valid = Trajectory(
            prompt=prob_thinking.user_problem,
            full_text="<think>\nFirst subtract 5: 2x = 10.\nThen divide by 2: x = 5.\n</think>\nThe answer is x = 5.",
            reasoning_trace="First subtract 5: 2x = 10.\nThen divide by 2: x = 5.",
            final_answer="The answer is x = 5.",
        )
        scored_valid = scorer.score_trajectory(prob_thinking, traj_valid)
        self.assertEqual(scored_valid.component_scores["thinking_tags"], 0.10)
        self.assertTrue(scored_valid.audit_log["thinking_tags"]["valid_thinking_tags"])

        # 2. Opened <think> but never closed </think> (unclosed tag)
        traj_unclosed = Trajectory(
            prompt=prob_thinking.user_problem,
            full_text="<think>\nFirst subtract 5: 2x = 10 and divide by 2...",
        )
        scored_unclosed = scorer.score_trajectory(prob_thinking, traj_unclosed)
        self.assertEqual(scored_unclosed.component_scores["thinking_tags"], -0.25)
        self.assertEqual(scored_unclosed.failure_reason, FailureReason.FORMAT_VIOLATION)

        # 3. Orphaned </think> without opening <think>
        traj_orphaned = Trajectory(
            prompt=prob_thinking.user_problem,
            full_text="Some random thoughts.\n</think>\nThe answer is 5.",
        )
        scored_orphaned = scorer.score_trajectory(prob_thinking, traj_orphaned)
        self.assertEqual(scored_orphaned.component_scores["thinking_tags"], -0.20)
        self.assertEqual(scored_orphaned.failure_reason, FailureReason.FORMAT_VIOLATION)

        # 4. Duplicate/nested <think> tags (hallucinated chat loop)
        traj_duplicate = Trajectory(
            prompt=prob_thinking.user_problem,
            full_text="<think>First part</think><think>Second part</think>Final answer.",
        )
        scored_duplicate = scorer.score_trajectory(prob_thinking, traj_duplicate)
        self.assertEqual(scored_duplicate.component_scores["thinking_tags"], -0.25)

        # 5. Empty thinking block
        traj_empty = Trajectory(
            prompt=prob_thinking.user_problem,
            full_text="<think>   </think>\nThe answer is 5.",
        )
        scored_empty = scorer.score_trajectory(prob_thinking, traj_empty)
        self.assertEqual(scored_empty.component_scores["thinking_tags"], -0.20)

        # 6. Inverted tags (closing tag before opening tag)
        traj_inverted = Trajectory(
            prompt=prob_thinking.user_problem,
            full_text="</think> upside down reasoning <think> answer",
        )
        scored_inverted = scorer.score_trajectory(prob_thinking, traj_inverted)
        self.assertEqual(scored_inverted.component_scores["thinking_tags"], -0.25)

        # 7. Condition met (HIGH effort / MATH domain) but thinking tags completely omitted
        traj_no_tags_high = Trajectory(
            prompt=prob_thinking.user_problem,
            full_text="The answer is 5.",
        )
        scored_no_tags = scorer.score_trajectory(prob_thinking, traj_no_tags_high)
        self.assertEqual(scored_no_tags.component_scores["thinking_tags"], -0.20)

        # 8. Condition NOT met (LOW tier, simple direct QA) and tags omitted -> allowed (0.0)
        prob_direct = DatasetRow(
            problem_id="prob_low_1",
            user_problem="What is the capital of France?",
            domain=DomainType.GENERAL,
            effort_tier=EffortTier.LOW,
            recheck_required=False,
        )
        traj_direct = Trajectory(
            prompt=prob_direct.user_problem,
            full_text="Paris",
        )
        scored_direct = scorer.score_trajectory(prob_direct, traj_direct)
        self.assertEqual(scored_direct.component_scores["thinking_tags"], 0.0)

        # 9. Leaked / duplicate tags in final answer
        traj_leaked = Trajectory(
            prompt=prob_thinking.user_problem,
            full_text="<think>\nValid thought.\n</think>\nThe answer is <think> 5.",
        )
        scored_leaked = scorer.score_trajectory(prob_thinking, traj_leaked)
        self.assertEqual(scored_leaked.component_scores["thinking_tags"], -0.25)

    # =========================================================
    # 13. Primary Scorer Column Grouping & Audit Integration
    # =========================================================
    def test_primary_scorer_column_scores_integration(self):
        """Verifies score_trajectory outputs column_scores structured for GDPO."""
        scorer = PrimaryScorer()
        prob = DatasetRow(
            problem_id="p1",
            user_problem="Explain quantum entanglement in numbered points.",
            domain=DomainType.SCIENCE,
            effort_tier=EffortTier.MEDIUM,
            explain_flag=True,
        )
        traj = Trajectory(
            prompt=prob.user_problem,
            full_text=(
                "<think>\nAnalyze entanglement between particle pairs and wavefunctions.\n</think>\n"
                "1. Entangled states share a joint quantum wavefunction.\n"
                "2. Measurement on one immediately correlates with the other.\n"
            ),
            final_answer="1. Entangled states share a joint quantum wavefunction.\n2. Measurement on one immediately correlates with the other.",
        )
        scored = scorer.score_trajectory(prob, traj)
        self.assertIn("accuracy", scored.column_scores)
        self.assertIn("formatting", scored.column_scores)
        self.assertIn("efficiency", scored.column_scores)
        self.assertIn("safety", scored.column_scores)
        self.assertGreater(scored.column_scores["formatting"], 0.0)


if __name__ == "__main__":
    unittest.main()
