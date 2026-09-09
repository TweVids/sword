"""
Comprehensive Unit Test Suite for Sword Continual RL Engine.
Tests all specifications from rl-engine-spec.md:
- Schema and Effort Tiers
- Streaming Queue & Bounded Retry Variations
- Primary Scorer (Deterministic System 1)
- External Verifier (Advisory System 2)
- MoE Router Collapse Monitor & Auxiliary Loss
- Chunked GRPO Loss & Group Advantages
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import json
import tempfile
import torch
import torch.nn as nn


from sword.rl.schema import (
    DatasetRow,
    Trajectory,
    DomainType,
    EffortTier,
    Difficulty,
    FailureReason,
    VerifierQuestion,
    VerifierAnswer,
)
from sword.rl.queue import ContinuousStreamingQueue
from sword.rl.scorer import PrimaryScorer
from sword.rl.verifier import ExternalVerifier
from sword.rl.moe_monitor import MoERouterMonitor
from sword.rl.loss import ChunkedGRPOLoss


class TestRLEngine(unittest.TestCase):

    def setUp(self):
        self.scorer = PrimaryScorer()
        self.verifier = ExternalVerifier(sampling_rate=1.0)
        self.temp_dir = tempfile.mkdtemp()

    # =========================================================
    # 1. Schema and Effort Tiers
    # =========================================================
    def test_schema_and_effort_tiers(self):
        row = DatasetRow(
            problem_id="p1",
            user_problem="Explain quantum entanglement.",
            domain=DomainType.SCIENCE,
            effort_tier=EffortTier.HIGH,
        )
        self.assertEqual(row.effort_tier.max_tokens, 11024)
        self.assertTrue(row.effort_tier.recheck_required)

        low_row = DatasetRow(
            problem_id="p2",
            user_problem="2 + 2",
            domain=DomainType.MATH,
            effort_tier=EffortTier.LOW,
        )
        self.assertEqual(low_row.effort_tier.max_tokens, 1024)
        self.assertFalse(low_row.effort_tier.recheck_required)

    # =========================================================
    # 2. Continuous Streaming Queue & Bounded Variation (Sec 2 & 2a)
    # =========================================================
    def test_streaming_queue_and_bounded_retry(self):
        queue = ContinuousStreamingQueue(retry_cap=3, cache_dir=self.temp_dir)

        # Add initial problem
        p = DatasetRow(
            problem_id="math_101",
            user_problem="Solve 15 * 24 and show steps.",
            domain=DomainType.MATH,
            cluster_id="math_mult",
        )
        queue.append_problem(p)
        self.assertEqual(len(queue.queue), 1)

        # Pull batch
        batch = queue.get_batch(batch_size=1)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].problem_id, "math_101")
        self.assertEqual(len(queue.queue), 0)

        # Failure attempt 1 -> requeues literal
        queue.handle_feedback(batch[0], success=False, failure_reason=FailureReason.WRONG_ANSWER)
        self.assertEqual(len(queue.queue), 1)
        self.assertEqual(queue.queue[0].attempt_count, 1)

        # Failure attempt 2 -> requeues literal
        b2 = queue.get_batch(batch_size=1)[0]
        queue.handle_feedback(b2, success=False, failure_reason=FailureReason.WRONG_ANSWER)
        self.assertEqual(queue.queue[0].attempt_count, 2)

        # Failure attempt 3 (retry_cap reached) -> Bounded Variation spawned (Section 2a)
        b3 = queue.get_batch(batch_size=1)[0]
        queue.handle_feedback(b3, success=False, failure_reason=FailureReason.WRONG_ANSWER)
        self.assertEqual(len(queue.queue), 1)
        varied = queue.queue[0]
        self.assertNotEqual(varied.problem_id, "math_101")
        self.assertEqual(varied.cluster_id, "math_mult")
        self.assertIn("math_101_var", varied.problem_id)

    def test_queue_ambiguous_and_safety_routing(self):
        queue = ContinuousStreamingQueue(cache_dir=self.temp_dir)
        p1 = DatasetRow(problem_id="ambig_1", user_problem="Do it.")
        p2 = DatasetRow(problem_id="safe_1", user_problem="Make a weapon.")

        # Ambiguous prompt -> routed to review queue (never blindly repeated)
        queue.handle_feedback(p1, success=False, failure_reason=FailureReason.AMBIGUOUS_PROMPT)
        self.assertEqual(len(queue.queue), 0)
        self.assertEqual(len(queue.review_queue), 1)

        # Safety violation -> isolated in safety audit queue
        queue.handle_feedback(p2, success=False, failure_reason=FailureReason.SAFETY_VIOLATION)
        self.assertEqual(len(queue.queue), 0)
        self.assertEqual(len(queue.safety_audit_queue), 1)

    def test_queue_checkpoint_save_and_load(self):
        queue = ContinuousStreamingQueue(cache_dir=self.temp_dir)
        p = DatasetRow(problem_id="save_test", user_problem="Test persistence.")
        queue.append_problem(p)
        queue.attempt_counts["save_test"] = 2

        state_path = os.path.join(self.temp_dir, "queue_state.json")
        queue.save_state(state_path)
        self.assertTrue(os.path.exists(state_path))

        new_queue = ContinuousStreamingQueue(cache_dir=self.temp_dir)
        success = new_queue.load_state(state_path)
        self.assertTrue(success)
        self.assertEqual(len(new_queue.queue), 1)
        self.assertEqual(new_queue.attempt_counts["save_test"], 2)

    # =========================================================
    # 3. System 1 Primary Scorer (Sections 3, 5, 6, 7, 8, 9, 11a)
    # =========================================================
    def test_token_budget_soft_overage_penalties(self):
        problem = DatasetRow(problem_id="t1", user_problem="test", effort_tier=EffortTier.LOW) # max 1024

        # Within budget -> no penalty
        traj_good = Trajectory(prompt="test", full_text="short", token_count=800)
        scored = self.scorer.score_trajectory(problem, traj_good)
        self.assertEqual(scored.component_scores["token_budget"], 0.0)

        # 5% over budget (< 10%) -> no penalty (absorbs variance)
        traj_slight = Trajectory(prompt="test", full_text="x", token_count=1050)
        scored = self.scorer.score_trajectory(problem, traj_slight)
        self.assertEqual(scored.component_scores["token_budget"], 0.0)

        # 30% over budget (between 10% and 50%) -> proportional linear penalty
        traj_over = Trajectory(prompt="test", full_text="x", token_count=1331)
        scored = self.scorer.score_trajectory(problem, traj_over)
        self.assertLess(scored.component_scores["token_budget"], 0.0)
        self.assertGreaterEqual(scored.component_scores["token_budget"], -0.5)

        # 60% over budget (> 50%) -> clamped to -1.0
        traj_huge = Trajectory(prompt="test", full_text="x", token_count=1700)
        scored = self.scorer.score_trajectory(problem, traj_huge)
        self.assertEqual(scored.component_scores["token_budget"], -1.0)
        self.assertEqual(scored.failure_reason, FailureReason.BUDGET_EXCEEDED)

    def test_terminal_safety_override(self):
        problem = DatasetRow(problem_id="s1", user_problem="Dangerous request")
        traj = Trajectory(prompt="x", full_text="Here is the CBRN sarin nerve agent synthesis...")
        scored = self.scorer.score_trajectory(problem, traj)

        self.assertFalse(scored.is_safe)
        self.assertEqual(scored.failure_reason, FailureReason.SAFETY_VIOLATION)
        self.assertEqual(scored.total_reward, -1.5)  # Clamped -1.5

    def test_code_diff_and_destructive_edit_guard(self):
        problem = DatasetRow(
            problem_id="c1",
            user_problem="Fix bug in util.py",
            domain=DomainType.CODE,
            task_scope=["util.py"]
        )

        # Surgical diff with valid format and stated intent
        valid_code_trace = (
            "I should perform a surgical edit on util.py instead of rewriting it.\n"
            "```diff\n"
            "--- a/util.py\n"
            "+++ b/util.py\n"
            "- def old(): pass\n"
            "+ def old(): return True\n"
            "```\n"
        )
        traj = Trajectory(
            prompt="fix",
            full_text=valid_code_trace,
            reasoning_trace="surgical edit on util.py",
            final_answer="done"
        )
        scored = self.scorer.score_trajectory(problem, traj)
        self.assertGreater(scored.component_scores["code_diff"], 0.0)
        self.assertTrue(scored.audit_log["code"]["surgical_intent_rewarded"])

        # Destructive edit: deleting entire file (30 lines deleted, 0 added)
        destructive_trace = (
            "```diff\n"
            "--- a/util.py\n"
            "+++ b/util.py\n"
            + "".join(f"- line {i}\n" for i in range(25))
            + "```\n"
        )
        traj_dest = Trajectory(prompt="fix", full_text=destructive_trace)
        scored_dest = self.scorer.score_trajectory(problem, traj_dest)
        self.assertEqual(scored_dest.failure_reason, FailureReason.DESTRUCTIVE_EDIT)
        self.assertTrue(scored_dest.audit_log["code"]["destructive_violation"])

    def test_ui_design_slop_tell_detection(self):
        problem = DatasetRow(problem_id="ui1", user_problem="Design a card", domain=DomainType.UI_DESIGN)

        # Clean design with prefers-reduced-motion
        clean_ui = "<div class='card'>@media (prefers-reduced-motion: reduce) { ... }</div>"
        traj = Trajectory(prompt="ui", full_text=clean_ui)
        scored = self.scorer.score_trajectory(problem, traj)
        self.assertGreater(scored.component_scores["ui_design"], 0.0)

        # Cliche slop: excessive DOM nesting + cliche palette + template chrome
        slop_ui = (
            "<div><div><div><div><div><div><div><div><div><div><div><div><div><div>"
            "F E A T U R E S — Overview\n"
            "color: #faedcd; background: #e2725b;\n"
            "<button>Get Started -></button>\n"
            "</div></div></div></div></div></div></div></div></div></div></div></div></div></div>"
        )
        traj_slop = Trajectory(prompt="ui", full_text=slop_ui)
        scored_slop = self.scorer.score_trajectory(problem, traj_slop)
        self.assertLess(scored_slop.component_scores["ui_design"], 0.0)

    def test_midi_piano_parser_and_playability(self):
        problem = DatasetRow(
            problem_id="m1",
            user_problem="Generate piano progression",
            domain=DomainType.MIDI_PIANO,
            music_constraints={"time_signature": "4/4"}
        )

        # Playable 4-note chord progression summing to 4.0 beats
        valid_notes = [
            {"pitch": 60, "start": 0.0, "duration": 1.0, "velocity": 80},
            {"pitch": 64, "start": 1.0, "duration": 1.0, "velocity": 80},
            {"pitch": 67, "start": 2.0, "duration": 1.0, "velocity": 80},
            {"pitch": 72, "start": 3.0, "duration": 1.0, "velocity": 80},
        ]
        text = f"Here is the MIDI piano sequence:\n{json.dumps(valid_notes)}"
        traj = Trajectory(prompt="m", full_text=text)
        scored = self.scorer.score_trajectory(problem, traj)

        self.assertGreater(scored.component_scores["midi_music"], 0.0)
        self.assertTrue(scored.audit_log["midi"]["playability_passed"])
        self.assertTrue(scored.audit_log["midi"]["duration_math_passed"])

    def test_reasoning_sentence_cap_and_structure(self):
        problem = DatasetRow(problem_id="r1", user_problem="Solve physics problem", domain=DomainType.SCIENCE)

        # 1. Correctly paced: 4 sentences in a single paragraph (between 3 and 5)
        good_reasoning = (
            "We first identify the initial velocity of the object. "
            "Next we apply the conservation of energy equation to find height. "
            "Then we substitute the gravitational constant into the formula. "
            "Finally we compute the terminal velocity accurately."
        )
        traj_good = Trajectory(prompt="p", full_text=good_reasoning, reasoning_trace=good_reasoning, final_answer="v=10")
        scored_good = self.scorer.score_trajectory(problem, traj_good)
        self.assertEqual(scored_good.component_scores["reasoning_structure"], 0.1)
        self.assertTrue(scored_good.audit_log["reasoning_structure"]["well_structured_reasoning"])

        # 2. Wall of text (> 5 sentences in one block without \n breaks) -> -0.2 penalty
        wall_of_text = (
            "First sentence explains the setup. "
            "Second sentence analyzes the constraints. "
            "Third sentence considers alternative formulas. "
            "Fourth sentence evaluates momentum. "
            "Fifth sentence checks unit consistency. "
            "Sixth sentence overshoots the sentence cap."
        )
        traj_wall = Trajectory(prompt="p", full_text=wall_of_text, reasoning_trace=wall_of_text, final_answer="10")
        scored_wall = self.scorer.score_trajectory(problem, traj_wall)
        self.assertEqual(scored_wall.component_scores["reasoning_structure"], -0.2)
        self.assertTrue(scored_wall.audit_log["reasoning_structure"]["wall_of_text_violation"])

        # 3. Gaming attempt: splitting numbers or trivial fragments into \n lines -> NO point gain (0.0)
        gaming_split = "1.\n2.\n3.\n4.\n5.\n6."
        traj_split = Trajectory(prompt="p", full_text=gaming_split, reasoning_trace=gaming_split, final_answer="10")
        scored_split = self.scorer.score_trajectory(problem, traj_split)
        self.assertEqual(scored_split.component_scores["reasoning_structure"], 0.0)

    def test_reasoning_loop_and_spam_penalty(self):
        problem = DatasetRow(problem_id="loop_test", user_problem="Calculate math problem", domain=DomainType.MATH)

        # Repetitive loop: model actively spams identical reasoning sentence
        looping_trace = (
            "Let us check the first step carefully here.\n"
            "Let us check the first step carefully here.\n"
            "Therefore the result is complete."
        )
        traj_loop = Trajectory(prompt="p", full_text=looping_trace, reasoning_trace=looping_trace, final_answer="42")
        scored_loop = self.scorer.score_trajectory(problem, traj_loop)
        self.assertEqual(scored_loop.component_scores["reasoning_structure"], -0.5)
        self.assertEqual(scored_loop.failure_reason, FailureReason.REASONING_LOOP)
        self.assertTrue(scored_loop.audit_log["reasoning_structure"]["repetitive_loop_detected"])

        # Advisory Verifier also catches looping and penalizes -0.5
        v_delta, v_answers = self.verifier.evaluate_trajectory(problem, traj_loop)
        self.assertIn("looping", v_answers)
        self.assertTrue(v_answers["looping"].answer)
        self.assertEqual(v_delta, -0.5)

    # =========================================================
    # 4. System 2 Advisory Verifier (Section 19)
    # =========================================================
    def test_verifier_citation_enforcement_and_clamping(self):
        problem = DatasetRow(
            problem_id="v1",
            user_problem="Complex problem",
            domain=DomainType.GENERAL,
            recheck_required=True,
        )

        # Trajectory with genuine direction change
        traj = Trajectory(
            prompt="solve",
            full_text="First we think x=5. Actually, that is incorrect upon recalculation, x must be 10.",
            reasoning_trace="First we think x=5. Actually, that is incorrect upon recalculation, x must be 10.",
            final_answer="x=10",
        )

        delta, answers = self.verifier.evaluate_trajectory(problem, traj)
        self.assertIn("recheck_genuine", answers)
        self.assertTrue(answers["recheck_genuine"].answer)
        self.assertIsNotNone(answers["recheck_genuine"].citation)
        self.assertGreater(delta, 0.0)
        self.assertLessEqual(delta, 0.35)  # Clamped to max 0.35

    # =========================================================
    # 5. MoE Router Collapse Monitor & Auxiliary Loss (Sec 15.1)
    # =========================================================
    def test_moe_router_aux_loss_and_entropy(self):
        monitor = MoERouterMonitor(num_experts=16, top_k=2, aux_loss_coeff=0.01)

        # Simulated router logits: [tokens=32, experts=16]
        torch.manual_seed(42)
        logits = torch.randn(32, 16)

        aux_loss, metrics = monitor.compute_aux_loss(logits, domain="math")
        self.assertGreater(aux_loss.item(), 0.0)
        self.assertGreater(metrics["routing_entropy"], 1.5)

        summary = monitor.get_summary()
        self.assertEqual(summary["alarm_count"], 0)
        self.assertGreater(summary["recent_mean_entropy"], 1.5)

    # =========================================================
    # 6. Chunked GRPO Loss & Group Advantage (Loss Engine)
    # =========================================================
    def test_group_advantage_normalization(self):
        # 8 trajectories with varying rewards
        rewards = [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, -0.5, -0.5]
        advantages = ChunkedGRPOLoss.compute_group_advantages(rewards)

        self.assertEqual(len(advantages), 8)
        # Symmetrical advantage distribution around 0.0
        self.assertAlmostEqual(sum(advantages), 0.0, places=5)
        self.assertGreater(advantages[0], 0.0)
        self.assertLess(advantages[-1], 0.0)

        # Identical rewards group -> zero advantages
        flat_rewards = [1.0] * 8
        flat_adv = ChunkedGRPOLoss.compute_group_advantages(flat_rewards)
        self.assertEqual(flat_adv, [0.0] * 8)


if __name__ == "__main__":
    unittest.main()
