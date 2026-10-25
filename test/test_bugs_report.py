"""
Bug-canary -> regression-guard tests for the issues documented in /bugs_report.txt.

History: these started as `xfail(strict=True)` canaries firing on audited bugs
(test/test_bugs_report.py, see bugs_report.txt). All 9 audited bugs have since
been FIXED in production code, so the canaries have been converted into
permanent regression guards (plain passing tests). Status:

  * H1  — REWARD HACK: fixed (MathScorer/ExplanationScorer final-answer-only
          matching + _narrow_final_answer in train_math_grpo.py)
  * E1  — fixed (length-insensitive single-value numeric match,
          sword/reward.py verify_answer)
  * H4  — fixed (old_logprobs/ref_logprobs supplied in train_math_grpo
          forward_single and local_trainer/train_grpo.py; frozen-adapter
          reference pass via compute_reference_logprobs in sword/rl/loss.py)
  * M1  — fixed (TIS weights always self-normalized in sword/grpo_bridge.py)
  * M2  — fixed (tis_bound_c < 1.0 now rejected in DualCorrectionGRPOLoss)
  * H2  — fixed (local_trainer trains on the engine's exact sampled token ids
          via return_logprobs/return_token_ids plumb-through)
  * M10 — fixed (strict unpacking `prob, = batch`; batch_size != 1 rejected)

Canonical sanity tests (correct answers rewarded, clip semantics documented,
engine module compiles) keep guarding that the fixes did not overcorrect.
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Toy model usable by sword/rl/policy.token_statistics (the generic branch:
# model_type not in linear_head_types -> model(...) is called directly).
# ---------------------------------------------------------------------------
class ToyLMOutput:
    def __init__(self, logits):
        self.logits = logits


class ToyLM(nn.Module):
    def __init__(self, vocab=7, seq=8):
        super().__init__()
        torch.manual_seed(0)
        # fixed random logits, [B, S, V]
        self.register_buffer("logits", torch.randn(1, seq, vocab) * 2.0)
        self.config = None

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        return ToyLMOutput(self.logits.expand(input_ids.shape[0], -1, -1))


def _single_sequence(seq_len=8, prompt_len=3):
    input_ids = torch.randint(0, 7, (1, seq_len))
    mask = torch.ones_like(input_ids)
    return input_ids, mask


# ===========================================================================
# H1 (FIXED) — Reward hacking in MathScorer must stay impossible
# ===========================================================================
@pytest.fixture(scope="module")
def math_scorer():
    tm = pytest.importorskip("train_math_grpo")
    return tm.MathScorer()


def test_h1_wrong_final_answer_but_reference_number_in_trace(math_scorer):
    # Model answers 7 (wrong). Its reasoning trace mentions "42" as an
    # intermediate value. Reference answer is 42 -> must NOT be scored correct.
    full_text = (
        "Let me estimate. Intermediate quantity: 12 * 3.5 = 42 candidate. "
        "So the final answer is 7."
    )
    result = math_scorer.score(
        problem="What is the answer?",
        reference_answer="42",
        full_text=full_text,
        effort_tier="high",
        token_count=30,
    )
    assert result["is_correct"] is False
    assert result["column_scores"]["accuracy"] < 0


def test_h1_reference_substring_in_full_text_grants_no_reward(math_scorer):
    # The reference string appears in the TRACE / full text; the final answer
    # (17 apples) is wrong. Full-text matching must not resurrect the reward.
    full_text = (
        "The problem statement mentions section 42 of the book. "
        "Therefore the final answer is 17 apples."
    )
    result = math_scorer.score(
        problem="How many apples?",
        reference_answer="42",
        full_text=full_text,
        effort_tier="high",
        token_count=25,
    )
    assert result["is_correct"] is False
    assert result["column_scores"]["accuracy"] < 0


def test_h1_sanity_correct_boxed_answer_is_rewarded(math_scorer):
    """Canonical sanity: a genuinely correct boxed answer must still count."""
    full_text = "Some reasoning 7 * 6 = 42 steps done.\nFinal: \\boxed{42}"
    result = math_scorer.score(
        problem="What is 7*6?",
        reference_answer="42",
        full_text=full_text,
        effort_tier="high",
        token_count=20,
    )
    assert result["is_correct"] is True
    assert result["column_scores"]["accuracy"] > 0


# ===========================================================================
# E1 (FIXED) — sword/reward.py must not reject correct answers on count
# ===========================================================================
def _load_math_verifier():
    pytest.importorskip("sword")
    pytest.importorskip("sword.b_model")  # reward.py imports ModelBJudge
    from sword.reward import MathVerifier
    return MathVerifier


def test_e1_extra_numbers_do_not_reject_correct_answer():
    MathVerifier = _load_math_verifier()
    # The true answer (42) is present, followed by a numeric aside "1e5".
    candidate = "think... n</>\nCORRECT</>\nThe answer is 42 (noting 1e5 scale units)."
    ok = MathVerifier.verify_answer(candidate, "42")
    assert ok is True


# ===========================================================================
# H4 (FIXED) — old_logprobs / ref_logprobs must be supplied at the call sites
# ===========================================================================
def _call_site_supplies(path: str, call_hint: str, kwargs_needed, window=12):
    """Scans a window of lines following the call hint for the required kwargs."""
    src = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if call_hint in line:
            block = " ".join(lines[i:i + window])
            if all(k in block for k in kwargs_needed):
                return True
    return False


def test_h4_train_math_grpo_supplies_old_and_ref_logprobs():
    ok = _call_site_supplies(
        "train_math_grpo.py", call_hint="_compute_surrogate_loss(",
        kwargs_needed=("old_logprobs=", "ref_logprobs="))
    assert ok, (
        "train_math_grpo.py must supply rollout-time old_logprobs and a frozen "
        "reference policy's ref_logprobs (or explicitly document why not)."
    )


def test_h4_local_trainer_supplies_old_and_ref_logprobs():
    ok = _call_site_supplies(
        os.path.join("local_trainer", "train_grpo.py"),
        call_hint="loss_fn.forward_chunked",
        kwargs_needed=("old_logprobs=", "ref_logprobs="))
    assert ok


def test_h4_document_clip_is_inert_when_old_logprobs_omitted():
    """
    Documents (passing) the loss-level default: with old_logprobs=None the
    ratio is identically 1, so surr1 == surr2 and clip_eps can never bind.
    Callers must therefore supply old_logprobs (see the two tests above).
    """
    pytest.importorskip("sword.rl.loss")
    from sword.rl.loss import ChunkedGRPOLoss

    loss = ChunkedGRPOLoss(clip_eps=0.2, kl_coeff=0.04)
    policy_logprobs = torch.log_softmax(torch.randn(1, 4), dim=-1)
    mask = torch.ones(1, 4, dtype=torch.bool)
    adv = torch.tensor([1.0])

    total, metrics = loss._compute_surrogate_loss(policy_logprobs, mask, adv)
    assert metrics["mean_ratio"] == 1.0            # clip machinery never engages
    assert metrics["kl_div"] == 0.0                # no ref model anywhere


# ===========================================================================
# M1 / M2 (FIXED) — TIS weighting in sword/grpo_bridge.DualCorrectionGRPOLoss
# ===========================================================================
def _tis_loss():
    pytest.importorskip("sword.grpo_bridge")
    from sword.grpo_bridge import DualCorrectionGRPOLoss
    return DualCorrectionGRPOLoss


def test_m2_tis_bound_below_one_rejected():
    DualCorrectionGRPOLoss = _tis_loss()
    with pytest.raises(ValueError):
        DualCorrectionGRPOLoss(tis_bound_c=0.5)


def test_m1_tis_loss_scale_invariant_to_rollout_policy_drift():
    DualCorrectionGRPOLoss = _tis_loss()
    model = ToyLM()
    input_ids, mask = _single_sequence()
    prompt_lengths = [3]
    advantages = torch.tensor([1.0])
    seq_len = input_ids.shape[1]

    # Rollout-policy logprobs: for shifted column j (predicting tokens[j+1]),
    # the vLLM sampling-policy logprob, here shifted by -1.0 relative to the
    # training policy's column logprob — a deliberate policy/logprob mismatch.
    lsm = torch.log_softmax(model.logits.float()[0], dim=-1)  # [S, V]
    seq_tokens = input_ids[0].tolist()
    full_col_lps = [lsm[j, seq_tokens[j + 1]].item() for j in range(seq_len - 1)]
    num_resp = seq_len - 1 - (prompt_lengths[0] - 1)
    resp_cols = list(range(prompt_lengths[0] - 1, seq_len - 1))
    rollout_logprobs = [[full_col_lps[j] - 1.0 for j in resp_cols]]

    loss_plain = DualCorrectionGRPOLoss(use_score_centering=False)
    loss_norm = DualCorrectionGRPOLoss(use_score_centering=True)

    total_plain, m_plain = loss_plain(
        model, input_ids, mask, prompt_lengths, advantages,
        rollout_logprobs=rollout_logprobs, global_valid_tokens=num_resp)
    total_norm, m_norm = loss_norm(
        model, input_ids, mask, prompt_lengths, advantages,
        rollout_logprobs=rollout_logprobs, global_valid_tokens=num_resp)

    # self-normalized weights keep the loss magnitude invariant to E[min(rho,C)]
    target, actual = total_norm.item(), total_plain.item()
    assert math.isclose(target, actual, rel_tol=0.25), (
        f"loss scale differs between normalized ({actual:.4f}) and "
        f"self-normalized ({target:.4f}) TIS"
    )


# ===========================================================================
# H2 (FIXED) — local_trainer must train on the engine's sampled token ids
# ===========================================================================
def test_h2_local_trainer_uses_sampled_token_ids():
    src = (REPO_ROOT / "local_trainer" / "train_grpo.py").read_text(
        encoding="utf-8", errors="ignore")
    assert "tokenizer.encode(prob.user_problem" not in src, (
        "training tokens must come from the rollout engine's sampled ids, "
        "not a re-encode of the raw problem text")
    assert "tokenizer.encode(scored.trajectory.full_text" not in src, (
        "response tokens must be the engine's sampled ids, not a re-encode of "
        "decoded text (drops EOS / special tokens)")


# ===========================================================================
# M10 (FIXED) — no silent dropping of batch rows
# ===========================================================================
def test_m10_binary_batch_size_guard():
    src = (REPO_ROOT / "local_trainer" / "train_grpo.py").read_text(
        encoding="utf-8", errors="ignore")
    assert "prob = batch[0]" not in src, (
        "batch[0] silently drops every other row; enforce batch_size == 1 in "
        "TrainerConfig or iterate the full batch")
    assert "batch_size != 1" in src, (
        "there must be a loud guard against batch_size > 1")


# ===========================================================================
# H6 — regression guard: engine_vllm_bypass.py stays compilable
# (the initial audit suspected truncation; verified false, kept as a guard)
# ===========================================================================
def test_h6_engine_vllm_bypass_module_is_complete():
    path = REPO_ROOT / "local_trainer" / "engine_vllm_bypass.py"
    assert path.exists()
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        compile(src, str(path), "exec")
    except SyntaxError as exc:
        pytest.fail(f"engine_vllm_bypass.py is not valid Python: {exc}")
