"""
Multi-Metric Reward Engine for GRPO Training.
Combines Model B correctness ratings, structural formatting checks,
Model B reasoning/answer rule evaluation, agentic tool success rates,
and penalty deductions for cheating.
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union, List

from sword.b_model import ModelBJudge, ModelBEvaluationResult, deterministic_evaluation


_DEFAULT_JUDGE = object()


@dataclass
class RewardBreakdown:
    """Detailed breakdown of multi-metric reward components."""
    truth: float = 0.0
    formatted: float = 0.0
    rule_reasoning_follow: float = 0.0
    rule_answer_follow: float = 0.0
    tools_success: float = 0.0
    cheating: bool = False
    total_score: float = 0.0
    is_agentic: bool = False
    judge_scores: Dict[str, Any] = field(default_factory=dict)
    judge_response: str = ""
    judge_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth": round(self.truth, 4),
            "formatted": round(self.formatted, 4),
            "rule_reasoning_follow": round(self.rule_reasoning_follow, 4),
            "rule_answer_follow": round(self.rule_answer_follow, 4),
            "tools_success": round(self.tools_success, 4),
            "cheating": self.cheating,
            "total_score": round(self.total_score, 4),
            "is_agentic": self.is_agentic,
        }


class MathVerifier:
    """Verifies numeric, algebraic, and root solutions for verifiable tasks."""

    @staticmethod
    def extract_final_answer(text: str) -> str:
        """Extracts the candidate final answer after think tags or boxed markers."""
        cleaned = text.strip()
        # 1. Answer after closing </think> tag
        if "</think>" in cleaned:
            parts = cleaned.split("</think>", 1)
            after_think = parts[1].strip()
            if after_think:
                cleaned = after_think

        # 2. Answer after <channel|>
        if "<channel|>" in cleaned:
            parts = cleaned.split("<channel|>", 1)
            after_ch = parts[1].strip()
            if after_ch:
                cleaned = after_ch

        # 3. Boxed answer \boxed{...}
        m_box = re.findall(r"\\boxed\{([^{}]+)\}", cleaned)
        if m_box:
            return m_box[-1].strip()

        # 4. "The answer is ..."
        m_ans = re.search(r"(?:the\s+answer\s+is\s*:?|final\s+answer\s*:?)\s*([^\n]+)", cleaned, re.IGNORECASE)
        if m_ans:
            return m_ans.group(1).strip().rstrip(".")

        return cleaned

    @classmethod
    def verify_answer(cls, candidate: str, expected_solution: str) -> bool:
        """Compares extracted candidate answer with expected solution."""
        cand_str = cls.extract_final_answer(candidate).lower().strip()
        exp_str = str(expected_solution).lower().strip()

        if cand_str == exp_str:
            return True

        # Check numeric equality
        try:
            number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
            cand_nums = [float(x) for x in re.findall(number_pattern, cand_str)]
            exp_nums = [float(x) for x in re.findall(number_pattern, exp_str)]
            if cand_nums and exp_nums and len(cand_nums) == len(exp_nums):
                if all(abs(c - e) < 1e-4 for c, e in zip(cand_nums, exp_nums)):
                    return True
                # Check sorted roots equality (e.g. x1 = 1, x2 = 2)
                if sorted(cand_nums) == sorted(exp_nums):
                    return True
            # BUG E1 fix: when the expected (or candidate) answer reduces to a
            # single value, accept a match wherever it appears — extra
            # parenthetical / unit numbers in prose must not invalidate a
            # numerically correct answer (count mismatch previously forced a
            # hard False for correct solutions).
            if cand_nums and exp_nums and (len(exp_nums) == 1 or len(cand_nums) == 1):
                if len(exp_nums) == 1:
                    e = exp_nums[0]
                    if any(abs(c - e) < 1e-4 for c in cand_nums):
                        return True
                if len(cand_nums) == 1:
                    c = cand_nums[0]
                    if any(abs(c - e) < 1e-4 for e in exp_nums):
                        return True
        except Exception:
            pass

        return False


class MultiMetricRewardScorer:
    """
    Computes unified GRPO reward using dataset attributes:
    problem, solution, reasoning_rule, answer_rule, difficulty, category, harness and code tests.
    """
    def __init__(self, b_model_judge: Optional[ModelBJudge] = _DEFAULT_JUDGE):
        # Omitted means the default judge; explicit None disables all Ollama IO.
        self.b_model_judge = ModelBJudge() if b_model_judge is _DEFAULT_JUDGE else b_model_judge

    def check_formatting(self, candidate_response: str) -> float:
        """
        Validates structural thinking and code fence formatting.
        Awards 1.0 for clean tags and balanced fences, 0.5 to 0.8 for minor flaws, 0.0 for broken output.
        """
        text = candidate_response.strip()
        if not text:
            return 0.0
        score = 1.0

        # Check thinking channel tags
        has_think_open = "<think>" in text or bool(re.search(r"<\|channel>\s*thought", text))
        has_think_close = "</think>" in text or "<channel|>" in text
        if has_think_open != has_think_close:
            score -= 0.4  # Unbalanced thinking tags

        # Check markdown code fences
        fence_count = text.count("```")
        if fence_count % 2 != 0:
            score -= 0.3  # Unbalanced code blocks
        if text.count("'''") % 2 != 0:
            score -= 0.3  # Unbalanced quoted code blocks

        # Check nested think tags
        if text.count("<think>") > 1 or text.count("</think>") > 1:
            score -= 0.3

        return max(0.0, score)

    def score(
        self,
        problem: str,
        solution: str,
        candidate_response: str,
        reasoning_rule: str = "",
        answer_rule: str = "",
        can_be_verify: Optional[bool] = None,
        difficulty: str = "medium",
        category: str = "general",
        harness_name: Optional[str] = None,
        tools_called: int = 0,
        tools_succeeded: int = 0,
        assistant_history: Optional[List[str]] = None,
        effort: str = "dynamic",
        execution_evidence: Optional[Dict[str, Any]] = None,
        dataset_row: Optional[Dict[str, Any]] = None,
        code_execution: Optional[Dict[str, Any]] = None,
    ) -> RewardBreakdown:
        """
        Computes the complete multi-metric score breakdown:
        truth, formatted, rule_reasoning_follow, rule_answer_follow, tools_success, cheating.
        The legacy can_be_verify caller argument is accepted but has no effect.
        """
        is_agentic = (tools_called > 0) or (bool(harness_name) and harness_name.lower() not in ("none", "default", "raw", ""))
        tools_success = (tools_succeeded / max(1, tools_called)) if tools_called > 0 else 0.0

        # 1. Structural formatting check
        formatted = self.check_formatting(candidate_response)

        # Model B evaluates correctness and dataset rules for all categories.
        b_res = self.b_model_judge.evaluate(
            problem=problem,
            solution=solution,
            reasoning_rule=reasoning_rule,
            answer_rule=answer_rule,
            candidate_response=candidate_response,
            difficulty=difficulty,
            category=category,
            effort=effort,
            tools_called=tools_called,
            tools_succeeded=tools_succeeded,
            **({"assistant_history": assistant_history} if assistant_history else {}),
            **({"execution_evidence": execution_evidence} if execution_evidence is not None else {}),
            **({"dataset_row": dataset_row} if dataset_row is not None else {}),
            **({"code_execution": code_execution} if code_execution else {}),
        ) if self.b_model_judge is not None else deterministic_evaluation(
            candidate_response, solution, tools_called, tools_succeeded,
        )

        rule_reasoning_follow = b_res.rule_reasoning_follow
        rule_answer_follow = b_res.rule_answer_follow
        cheating = b_res.cheating
        truth = b_res.truth

        # A successful live judge owns math correctness too: exact string/number
        # matching must not override semantic evaluation or cap it at 0.5.
        # Offline callers use the judge's deterministic fallback, without a
        # dataset flag or a separate half-credit math override.
        live_judge = self.b_model_judge is not None and getattr(
            self.b_model_judge, "last_evaluation_source", None,
        ) != "deterministic"
        if execution_evidence is not None:
            tests = execution_evidence.get("tests", {})
            total = tests.get("total", 0)
            if total > 0:
                test_truth = max(0.0, min(1.0, tests.get("passed", 0) / total))
                # A judge cannot override failing authoritative dataset checks.
                truth = min(truth, test_truth) if live_judge else test_truth
        cases = (code_execution or {}).get("cases", [])
        expected_cases = len((dataset_row or {}).get("code_tests", []))
        if cases or expected_cases:
            code_truth = sum(bool(case.get("success")) for case in cases) / max(1, expected_cases, len(cases))
            if live_judge or (execution_evidence or {}).get("tests", {}).get("total", 0):
                truth = min(truth, code_truth)
            else:
                truth = code_truth
        from sword.harness.protocol import visible_response
        if not visible_response(candidate_response).strip():
            truth = 0.0

        # 4. Total score calculation based on user specification:
        # If agentic was on: divide by 5; otherwise divide by 4.
        # If cheating was caught: subtract 0.2 penalty.
        if is_agentic:
            base_score = (truth + formatted + rule_reasoning_follow + rule_answer_follow + tools_success) / 5.0
        else:
            base_score = (truth + formatted + rule_reasoning_follow + rule_answer_follow) / 4.0

        if cheating:
            total_score = base_score - 0.2
        else:
            total_score = base_score

        # Clamp total score to valid range [-0.2, 1.0]
        total_score = max(-0.2, min(1.0, total_score))

        return RewardBreakdown(
            truth=truth,
            formatted=formatted,
            rule_reasoning_follow=rule_reasoning_follow,
            rule_answer_follow=rule_answer_follow,
            tools_success=tools_success,
            cheating=cheating,
            total_score=total_score,
            is_agentic=is_agentic,
            judge_scores=b_res.to_dict(),
            judge_response=b_res.raw_response,
            judge_source=getattr(self.b_model_judge, "last_evaluation_source", "unknown") if self.b_model_judge is not None else "disabled",
        )
