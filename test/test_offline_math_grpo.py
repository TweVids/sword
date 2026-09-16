"""
Offline Simulation and Verification Test Suite for Sword Math GRPO.
Runs 100% offline without needing a 30B model or downloading large checkpoints.

Tests:
1. ThinkingFormatVerifier: all 8 tag edge cases (<think> ... </think>).
2. MathScorer Accuracy: integer, float (.0), boxed, intervals, Russian semicolons, LaTeX fractions.
3. MathScorer Formatting: natural paragraphs vs bullet/numbered/header lists in thinking.
4. MathScorer Math Equivalence: math formulas inside <think> are not penalized.
5. MathScorer Outside <think>: boxed answer bonus, paragraph answer bonus, bullet penalties.
6. Paragraph Sentence Depth: 3-5 sentence requirement on BigMath2 & xhigh/ultra/max tiers.
7. System Prompt Ordering: dynamic effort prompt FIRST, followed by paragraph and boxed directives.
8. GDPO Advantages: non-zero variance preservation, conciseness bonus tie-breaking.
9. ChunkedGRPOLoss: micro-batch forward_single and backward pass on a lightweight mock model.
10. StepAuditor: JSONL, flat table, CSV export columns, and tier summary aggregation.
11. Full End-to-End Simulation: multi-step GRPO cycle with mock model and optimizer.
"""

import sys
import os
import unittest
import json
import tempfile
import shutil
import re
import math
import torch
import torch.nn as nn

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_math_grpo import (
    ThinkingFormatVerifier,
    MathScorer,
    compute_gdpo_advantages,
    ChunkedGRPOLoss,
    StepAuditor,
    format_effort_prompt,
    PERSISTENT_PARAGRAPH_PROMPT,
    BOXED_ANSWER_PROMPT,
    EFFORT_SYSTEM_PROMPTS,
)


class MockLightweightModel(nn.Module):
    """Tiny mock model for testing ChunkedGRPOLoss without large weights."""
    def __init__(self, vocab_size: int = 128, hidden_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        hidden = self.embedding(input_ids)
        logits = self.linear(hidden)
        class Output:
            pass
        out = Output()
        out.logits = logits
        return out


class TestThinkingFormatVerifier(unittest.TestCase):
    """Tests all parsing and edge cases of <think> ... </think> tags."""

    def test_well_formed_tags(self):
        text = "<think> This is valid reasoning. </think> \boxed{42}"
        score, audit, trace, ans = ThinkingFormatVerifier.verify(text)
        self.assertEqual(score, 0.10)
        self.assertTrue(audit.get("valid_thinking_tags"))
        self.assertEqual(trace, "This is valid reasoning.")
        self.assertEqual(ans, "\boxed{42}")

    def test_unclosed_tag(self):
        text = "<think> Started reasoning but ran out of tokens"
        score, audit, trace, ans = ThinkingFormatVerifier.verify(text)
        self.assertEqual(score, -0.25)
        self.assertEqual(audit.get("error"), "unclosed_think_tag")

    def test_orphaned_close_tag(self):
        text = "Reasoning without open tag </think> The answer is 42."
        score, audit, trace, ans = ThinkingFormatVerifier.verify(text)
        self.assertEqual(score, -0.20)
        self.assertEqual(audit.get("error"), "orphaned_close_tag")

    def test_inverted_tags(self):
        text = "</think> inverted <think> 42"
        score, audit, trace, ans = ThinkingFormatVerifier.verify(text)
        self.assertEqual(score, -0.25)
        self.assertEqual(audit.get("error"), "inverted_tags_close_before_open")

    def test_duplicate_nested_tags(self):
        text = "<think> First think <think> nested think </think> 42"
        score, audit, trace, ans = ThinkingFormatVerifier.verify(text)
        self.assertEqual(score, -0.25)
        self.assertEqual(audit.get("error"), "duplicate_or_nested_tags")

    def test_empty_thinking_block(self):
        text = "<think></think> 42"
        score, audit, trace, ans = ThinkingFormatVerifier.verify(text)
        self.assertEqual(score, -0.20)
        self.assertEqual(audit.get("error"), "empty_thinking_block")

    def test_missing_tags(self):
        text = "Just direct answer 42 without thinking tags."
        score, audit, trace, ans = ThinkingFormatVerifier.verify(text)
        self.assertEqual(score, -0.20)
        self.assertEqual(audit.get("error"), "missing_required_thinking_tags")

    def test_leaked_tags_in_answer(self):
        text = "<think> Valid trace </think> Answer is 42 <think>"
        score, audit, trace, ans = ThinkingFormatVerifier.verify(text)
        self.assertEqual(score, -0.25)
        self.assertEqual(audit.get("error"), "duplicate_or_nested_tags")

    def test_quoted_closing_tag_inside_thinking_trace(self):
        # Rollout 4 scenario: model mentions '</think>' in thought sentence, followed by real closing tag
        text = (
            "<think>\n"
            "We are given 238 and 119.\n"
            "We should present reasoning inside </think> tags and the answer outside.\n"
            "</think>\n"
            "357"
        )
        score, audit, trace, ans = ThinkingFormatVerifier.verify(text)
        self.assertEqual(score, 0.10)
        self.assertTrue(audit.get("valid_thinking_tags"))
        self.assertTrue(audit.get("quoted_close_tag_in_trace"))
        self.assertIn("We are given 238 and 119", trace)
        self.assertEqual(ans, "357")


class TestMathScorerAccuracy(unittest.TestCase):
    """Tests numeric equivalence, intervals, LaTeX, and Russian notation."""

    def setUp(self):
        self.scorer = MathScorer()

    def test_exact_integer_match(self):
        res = self.scorer.score("prob", "357", "<think> Reasoning </think> 357")
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["column_scores"]["accuracy"], 0.35)

    def test_float_point_zero_equivalence(self):
        # Ground truth is "357.0", model outputs "357"
        res = self.scorer.score("prob", "357.0", "<think> Reasoning </think> 357")
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["column_scores"]["accuracy"], 0.35)

    def test_boxed_float_equivalence(self):
        res = self.scorer.score("prob", "357.0", "<think> Reasoning </think> \boxed{357}")
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["column_scores"]["accuracy"], 0.35)

    def test_russian_semicolon_intervals(self):
        # BigMath2 Russian interval notation: [4; \infty) vs English [4, \infty)
        res = self.scorer.score("prob", "[4; \\infty)", "<think> Reasoning </think> \\boxed{[4, \\infty)}")
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["column_scores"]["accuracy"], 0.35)

    def test_latex_fractions(self):
        res = self.scorer.score("prob", "\\frac{1}{2}", "<think> Reasoning </think> 1/2")
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["column_scores"]["accuracy"], 0.35)

    def test_incorrect_answer(self):
        res = self.scorer.score("prob", "357.0", "<think> Reasoning </think> \boxed{999}")
        self.assertFalse(res["is_correct"])
        self.assertEqual(res["column_scores"]["accuracy"], -0.40)


class TestMathScorerFormatting(unittest.TestCase):
    """Tests paragraph thinking, bullet penalties, math formulas, and answer structure."""

    def setUp(self):
        self.scorer = MathScorer()

    def test_natural_paragraph_rewarded(self):
        text = (
            "<think>\n"
            "We calculate the total by taking the 238 items sold in April and adding the 119 items from May. "
            "Summing these two quantities gives exactly 357 items. Everything is verified.\n"
            "</think>\n"
            "\\boxed{357}"
        )
        res = self.scorer.score("prob", "357", text, effort_tier="low")
        self.assertTrue(res["audit_log"].get("natural_paragraph_thinking_rewarded"))
        self.assertNotIn("bullet_or_list_in_thinking", res["audit_log"])

    def test_bullet_list_penalized_in_thinking(self):
        text = (
            "<think>\n"
            "Given:\n"
            "- April: 238\n"
            "- May: 119\n"
            "Calculate sum.\n"
            "</think>\n"
            "\\boxed{357}"
        )
        res = self.scorer.score("prob", "357", text, effort_tier="low")
        self.assertTrue(res["audit_log"].get("bullet_or_list_in_thinking"))
        self.assertNotIn("natural_paragraph_thinking_rewarded", res["audit_log"])

    def test_numbered_list_penalized_in_thinking(self):
        text = (
            "<think>\n"
            "1. Identify April sales: 238.\n"
            "2. Identify May sales: 119.\n"
            "3. Add them together.\n"
            "</think>\n"
            "\\boxed{357}"
        )
        res = self.scorer.score("prob", "357", text, effort_tier="low")
        self.assertTrue(res["audit_log"].get("bullet_or_list_in_thinking"))

    def test_headers_penalized_in_thinking(self):
        text = (
            "<think>\n"
            "### Step 1: Addition\n"
            "We add 238 and 119 to get 357.\n"
            "</think>\n"
            "\\boxed{357}"
        )
        res = self.scorer.score("prob", "357", text, effort_tier="low")
        self.assertTrue(res["audit_log"].get("bullet_or_list_in_thinking"))

    def test_math_formulas_not_penalized_as_bullets(self):
        text = (
            "<think>\n"
            "To solve the equation we set up the total sum.\n"
            "total = 238 + 119 = 357\n"
            "The sum matches our expectations with no errors.\n"
            "</think>\n"
            "\\boxed{357}"
        )
        res = self.scorer.score("prob", "357", text, effort_tier="low")
        self.assertNotIn("bullet_or_list_in_thinking", res["audit_log"])
        self.assertTrue(res["audit_log"].get("natural_paragraph_thinking_rewarded"))

    def test_latex_display_math_and_negative_equations_not_penalized_as_bullets(self):
        text = (
            "<think>\n"
            "We solve the projectile equation under standard gravity.\n"
            "\\[ h(t) = -16t^2 + 64t + 80 \\]\n"
            "Setting the height equation equal to zero yields the time of impact:\n"
            "- 16t^2 + 64t + 80 = 0\n"
            "Factoring out -16 gives (t - 5)(t + 1) = 0. Therefore t = 5.\n"
            "</think>\n"
            "\\boxed{5}"
        )
        res = self.scorer.score("prob", "5", text, effort_tier="low")
        self.assertNotIn("bullet_or_list_in_thinking", res["audit_log"])
        self.assertTrue(res["audit_log"].get("natural_paragraph_thinking_rewarded"))

    def test_outside_think_paragraph_rewarded(self):
        text = (
            "<think>\n"
            "We compute 238 plus 119 to arrive at 357.\n"
            "</think>\n"
            "Rosy Plascencia sold a total of \\boxed{357} air fryers altogether."
        )
        res = self.scorer.score("prob", "357", text, effort_tier="low")
        self.assertTrue(res["audit_log"].get("natural_paragraph_answer_rewarded"))
        self.assertTrue(res["audit_log"].get("boxed_answer_rewarded"))

    def test_outside_think_bullets_penalized(self):
        text = (
            "<think>\n"
            "We compute 238 plus 119 to arrive at 357.\n"
            "</think>\n"
            "- Final Answer: \\boxed{357}"
        )
        res = self.scorer.score("prob", "357", text, effort_tier="low")
        self.assertTrue(res["audit_log"].get("bullet_or_list_in_answer"))

    def test_reasoning_dump_and_headers_in_answer_penalized(self):
        # Rollout 3 scenario: dumping # Headers, **Step 1:**, and bullet lists outside <think>
        text = (
            "<think>\n"
            "We compute 238 plus 119.\n"
            "</think>\n"
            "# Solving Rosy Plascencia's Air Fryer Sales\n\n"
            "**Step 1: Identify the given information**\n"
            "- April sales: 238 air fryers\n"
            "- May sales: 119 air fryers\n\n"
            "**Step 2: Add them**\n"
            "$$238 + 119 = 357$$\n\n"
            "Rosy Plascencia sold 357 air fryers altogether."
        )
        res = self.scorer.score("prob", "357", text, effort_tier="low")
        self.assertTrue(res["audit_log"].get("reasoning_dump_in_answer"))
        self.assertNotIn("natural_paragraph_answer_rewarded", res["audit_log"])
        # Format column must reflect the strong penalty
        self.assertLess(res["column_scores"]["formatting"], 0.10)


class TestParagraphSentenceDepth(unittest.TestCase):
    """Tests 3-5 sentence paragraph depth requirement for BigMath2 and xhigh+ tiers."""

    def setUp(self):
        self.scorer = MathScorer()

    def test_template_gsm_low_effort_exempt_from_depth(self):
        # In TemplateGSM with low effort, a 1-sentence paragraph is not penalized
        text = "<think> We simply add 238 and 119 to get 357. </think> \\boxed{357}"
        res = self.scorer.score("prob", "357", text, effort_tier="low", dataset_name="math-ai/TemplateGSM")
        self.assertNotIn("shallow_paragraphs_penalty", res["audit_log"])

    def test_bigmath2_shallow_paragraph_penalized(self):
        # In BigMath2, a 1-sentence paragraph triggers penalty
        text = "<think> Only one sentence here. </think> \\boxed{10}"
        res = self.scorer.score("prob", "10", text, effort_tier="medium", dataset_name="Nihilux/BigMath2")
        self.assertTrue(res["audit_log"].get("shallow_paragraphs_penalty"))

    def test_bigmath2_deep_paragraph_rewarded(self):
        # In BigMath2, a 4-sentence paragraph triggers reward
        text = (
            "<think>\n"
            "We first consider the boundary conditions. "
            "Then we integrate the function over the given interval. "
            "Next we evaluate the constant of integration. "
            "Finally we compute the definite sum to reach the solution.\n"
            "</think>\n"
            "\\boxed{10}"
        )
        res = self.scorer.score("prob", "10", text, effort_tier="medium", dataset_name="Nihilux/BigMath2")
        self.assertTrue(res["audit_log"].get("deep_paragraph_sentences_rewarded"))

    def test_xhigh_effort_shallow_penalized_even_on_gsm(self):
        # In xhigh effort, even GSM requires deep paragraphs
        text = "<think> Just one quick sentence. </think> \\boxed{357}"
        res = self.scorer.score("prob", "357", text, effort_tier="xhigh", dataset_name="math-ai/TemplateGSM")
        self.assertTrue(res["audit_log"].get("shallow_paragraphs_penalty"))

    def test_display_math_inside_paragraph_sentence_counting(self):
        text = (
            "<think>\n"
            "We consider the indefinite integral of the exponential decay function. "
            "\\[ \\int e^{-2x} dx = -\\frac{1}{2}e^{-2x} + C \\] "
            "Evaluating this antiderivative from zero to infinity gives the total area. "
            "At infinity the exponential term approaches zero identically. "
            "Therefore the definite integral converges to exactly 0.5.\n"
            "</think>\n"
            "\\boxed{0.5}"
        )
        res = self.scorer.score("prob", "0.5", text, effort_tier="medium", dataset_name="Nihilux/BigMath2")
        self.assertTrue(res["audit_log"].get("deep_paragraph_sentences_rewarded"))
        self.assertNotIn("shallow_paragraphs_penalty", res["audit_log"])


class TestSystemPromptsAndOrdering(unittest.TestCase):
    """Verifies that effort prompt is FIRST, followed by paragraph and boxed directives."""

    def test_prompt_order_and_separation(self):
        prompt = format_effort_prompt("Find the value of x.", effort_tier="xhigh")
        lines = [line.strip() for line in prompt.split("\n") if line.strip()]

        # System start tag must be present
        self.assertIn("<|im_start|>system", prompt)

        # 1. Dynamic effort directive MUST come first
        effort_sub = EFFORT_SYSTEM_PROMPTS["xhigh"]
        self.assertTrue(effort_sub in prompt)
        pos_effort = prompt.find(effort_sub)

        # 2. Persistent paragraph prompt must come second
        pos_para = prompt.find(PERSISTENT_PARAGRAPH_PROMPT)
        self.assertGreater(pos_para, pos_effort)

        # 3. Boxed answer prompt must come third
        pos_boxed = prompt.find(BOXED_ANSWER_PROMPT)
        self.assertGreater(pos_boxed, pos_para)


class TestGDPOAdvantagesAndVariance(unittest.TestCase):
    """Verifies decoupled advantage calculation and non-zero variance preservation."""

    def setUp(self):
        self.scorer = MathScorer()

    def test_non_zero_advantages_when_all_correct_with_different_lengths(self):
        # 4 rollouts, all correct, all >= 200 min tokens, but varying in length (conciseness breaks ties)
        r1 = self.scorer.score("p", "10", "<think> Par 1. Sentence two. Sentence three. </think> \\boxed{10}", effort_tier="xhigh", token_count=250)
        r2 = self.scorer.score("p", "10", "<think> Par 2. Sentence two. Sentence three. </think> \\boxed{10}", effort_tier="xhigh", token_count=350)
        r3 = self.scorer.score("p", "10", "<think> Par 3. Sentence two. Sentence three. </think> \\boxed{10}", effort_tier="xhigh", token_count=220)
        r4 = self.scorer.score("p", "10", "<think> Par 4. Sentence two. Sentence three. </think> \\boxed{10}", effort_tier="xhigh", token_count=500)

        advs, norm_advs = compute_gdpo_advantages([r1, r2, r3, r4])
        self.assertEqual(len(advs), 4)

        # Verify variance is strictly non-zero
        self.assertTrue(any(abs(a) > 1e-4 for a in advs), f"Expected non-zero advantages, got: {advs}")
        # The shortest correct rollout (220 tokens) should have higher advantage than the longest (500 tokens)
        self.assertGreater(advs[2], advs[3])

    def test_diverse_rollouts_advantage_ranking(self):
        # 2 correct, 2 incorrect
        r_corr1 = self.scorer.score("p", "10", "<think> Correct reasoning. </think> \\boxed{10}", effort_tier="low", token_count=100)
        r_corr2 = self.scorer.score("p", "10", "<think> Correct reasoning too. </think> \\boxed{10}", effort_tier="low", token_count=120)
        r_wrong1 = self.scorer.score("p", "10", "<think> Wrong reasoning. </think> \\boxed{999}", effort_tier="low", token_count=100)
        r_wrong2 = self.scorer.score("p", "10", "<think> Wrong reasoning too. </think> \\boxed{888}", effort_tier="low", token_count=100)

        advs, _ = compute_gdpo_advantages([r_corr1, r_corr2, r_wrong1, r_wrong2])
        self.assertGreater(advs[0], 0.0)
        self.assertGreater(advs[1], 0.0)
        self.assertLess(advs[2], 0.0)
        self.assertLess(advs[3], 0.0)


class TestChunkedGRPOLossAndMicrobatch(unittest.TestCase):
    """Tests the micro-batch forward_single and backward pass using a mock PyTorch model."""

    def setUp(self):
        self.model = MockLightweightModel(vocab_size=100, hidden_dim=16)
        self.loss_fn = ChunkedGRPOLoss(chunk_size=16)

    def test_forward_single_and_backward(self):
        input_ids = torch.randint(0, 100, (1, 30))
        attention_mask = torch.ones_like(input_ids)
        prompt_length = 10
        advantage = 0.50

        self.model.train()
        loss, metrics = self.loss_fn.forward_single(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_length=prompt_length,
            advantage=advantage,
        )

        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.requires_grad)
        self.assertFalse(torch.isnan(loss))
        self.assertIn("nll", metrics)
        self.assertGreater(metrics["nll"], 0.0)
        self.assertIn("pg_loss", metrics)

        # Backward pass
        loss.backward()
        for p in self.model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertFalse(torch.isnan(p.grad).any())


class TestStepAuditor(unittest.TestCase):
    """Tests logging, JSONL/CSV exporting, dataset columns, and summary metrics."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.auditor = StepAuditor(log_dir=self.temp_dir, dataset_name="math-ai/TemplateGSM")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_and_file_exports(self):
        rollouts = [
            {
                "rollout_index": 0,
                "token_count": 150,
                "total_reward": 0.45,
                "is_correct": True,
                "final_answer": "\\boxed{357}",
                "reasoning_trace": "Paragraph reasoning trace.",
                "column_scores": {"accuracy": 0.35, "formatting": 0.10, "efficiency": 0.08},
                "component_scores": {"thinking_tags": 0.10},
                "advantage": 0.5,
                "audit_log": {},
            }
        ]

        self.auditor.record(
            step=1,
            problem="Sample question",
            reference="357",
            rollouts=rollouts,
            elapsed=1.23,
            effort_tier="low",
            dataset_name="math-ai/TemplateGSM",
            step_loss=1.4520,
            grad_norm=0.38,
        )

        # Check step JSON file
        step_file = os.path.join(self.temp_dir, "step_0001.json")
        self.assertTrue(os.path.exists(step_file))
        with open(step_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["dataset"], "math-ai/TemplateGSM")
            self.assertEqual(data["step"], 1)
            self.assertEqual(data["step_loss"], 1.4520)
            self.assertEqual(data["grad_norm"], 0.38)

        # Check CSV export
        csv_file = os.path.join(self.temp_dir, "generations_table.csv")
        self.assertTrue(os.path.exists(csv_file))
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreaterEqual(len(lines), 2)
            header = lines[0]
            self.assertIn("dataset", header)
            self.assertIn("effort_tier", header)
            self.assertIn("math-ai/TemplateGSM", lines[1])

        # Check summary
        summary = self.auditor.summarize()
        self.assertEqual(summary["total_steps"], 1)
        self.assertEqual(summary["accuracy_pct"], 100.0)


class TestFullOfflineSimulation(unittest.TestCase):
    """Simulates a complete 3-step GRPO cycle with a lightweight model, zero VRAM overhead."""

    def test_simulated_3_steps(self):
        temp_dir = tempfile.mkdtemp()
        try:
            model = MockLightweightModel(vocab_size=100, hidden_dim=16)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loss_fn = ChunkedGRPOLoss(chunk_size=16)
            scorer = MathScorer()
            auditor = StepAuditor(log_dir=temp_dir, dataset_name="math-ai/TemplateGSM")

            mock_data = [
                {"problem": "What is 2+2?", "answer": "4.0", "dataset": "math-ai/TemplateGSM", "effort_tier": "low"},
                {"problem": "What is 10*5?", "answer": "50.0", "dataset": "math-ai/TemplateGSM", "effort_tier": "medium"},
                {"problem": "Find the roots of x^2 - 4 = 0", "answer": "[-2, 2]", "dataset": "Nihilux/BigMath2", "effort_tier": "xhigh"},
            ]

            for step, item in enumerate(mock_data, start=1):
                prob = item["problem"]
                ref = item["answer"]
                tier = item["effort_tier"]
                ds = item["dataset"]

                # Simulate 4 rollouts with slight variation in lengths & formats
                simulated_texts = [
                    f"<think> Continuous reasoning for problem {step}. We proceed step by step. Everything is consistent. </think> \\boxed{{{ref.replace('.0', '')}}}",
                    f"<think> Another paragraph reasoning path. We compute carefully. The calculation yields the value. </think> \\boxed{{{ref.replace('.0', '')}}}",
                    f"<think> Third path with slightly different length. We verify the result. </think> The answer is {ref.replace('.0', '')}",
                    f"<think> Fourth path with incorrect outcome. </think> \\boxed{{999}}",
                ]

                rollout_results = []
                for idx, text in enumerate(simulated_texts):
                    token_count = 100 + idx * 30
                    res = scorer.score(
                        problem=prob,
                        reference_answer=ref,
                        full_text=text,
                        effort_tier=tier,
                        token_count=token_count,
                        dataset_name=ds,
                    )
                    res["full_text"] = text
                    res["token_count"] = token_count
                    rollout_results.append(res)

                advs, _ = compute_gdpo_advantages(rollout_results)
                for idx, a in enumerate(advs):
                    rollout_results[idx]["advantage"] = a

                # Micro-batch backward
                optimizer.zero_grad()
                total_loss = 0.0
                for idx, res in enumerate(rollout_results):
                    input_ids = torch.randint(0, 100, (1, 40))
                    mask = torch.ones_like(input_ids)
                    l_i, _ = loss_fn.forward_single(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=mask,
                        prompt_length=15,
                        advantage=advs[idx],
                    )
                    scaled_l = l_i / len(rollout_results)
                    scaled_l.backward()
                    total_loss += l_i.item()

                optimizer.step()

                auditor.record(
                    step=step,
                    problem=prob,
                    reference=ref,
                    rollouts=rollout_results,
                    elapsed=0.05,
                    effort_tier=tier,
                    dataset_name=ds,
                )

                # Verify loss is finite
                self.assertFalse(math.isnan(total_loss))

            summary = auditor.summarize()
            self.assertEqual(summary["total_steps"], 3)
            self.assertGreater(summary["accuracy_pct"], 0.0)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
