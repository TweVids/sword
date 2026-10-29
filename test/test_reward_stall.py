"""Regressions for the zero-loss/zero-gradient long-trainer report."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sword.diagnostics import check_training_result, diagnostic_dataset
from sword.b_model import ModelBEvaluationResult
from sword.grpo_bridge import GRPOBridge, GRPOBridgeConfig, RolloutBatch, RolloutItem
from sword.long_trainer import run_long_train
from sword.reward import MultiMetricRewardScorer, RewardBreakdown


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, -1.0, 3.0, 1.0]))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


def batch():
    return RolloutBatch(items=[
        RolloutItem("p", [1], "r", [2], [-0.5]),
        RolloutItem("p", [1], "r2", [3], [-0.5]),
    ], concurrency_g=2)


def bridge():
    result = GRPOBridge(GRPOBridgeConfig(concurrency_g=2, device="cpu", entropy_coeff=0.0))
    result.trainer_model = TinyPolicy()
    result.tokenizer = SimpleNamespace(pad_token_id=0)
    return result


def test_first_uniform_group_reports_no_baseline_used(capsys):
    trainer = bridge()
    trainer.score_rollouts(batch(), lambda p, r: 0.225)
    assert "no reward signal" in capsys.readouterr().out
    assert trainer._reward_diagnostics["baseline_groups"] == 0


def test_disabling_baseline_ignores_existing_ema():
    trainer = bridge()
    trainer.score_rollouts(batch(), lambda p, r: 0.225)
    trainer.config.use_cross_step_baseline = False
    scored = batch()
    trainer.score_rollouts(scored, lambda p, r: 0.425)
    assert all(item.advantage == 0 for item in scored.items)


def test_zero_gradient_step_preserves_adam_momentum_and_weights():
    trainer = bridge()
    optimizer = torch.optim.AdamW(trainer.trainer_model.parameters(), lr=0.1, weight_decay=0.1)
    # Populate Adam momentum with a real update, then deliver a flat reward group.
    scored = batch()
    trainer.score_rollouts(scored, lambda p, r: 0.0 if r == "r" else 1.0)
    trainer.train_step(scored, optimizer)
    before = trainer.trainer_model.logits.detach().clone()
    param = trainer.trainer_model.logits
    moment = optimizer.state[param]["exp_avg"].clone()
    optimizer_step = optimizer.state[param]["step"].clone()
    version = trainer.weight_sync.policy_version
    trainer.config.use_cross_step_baseline = False
    scored = batch()
    scored.policy_version = version
    for item in scored.items:
        item.policy_version = version
    trainer.score_rollouts(scored, lambda p, r: 0.225)
    metrics = trainer.train_step(scored, optimizer)
    assert torch.equal(param, before)
    assert torch.equal(optimizer.state[param]["exp_avg"], moment)
    assert torch.equal(optimizer.state[param]["step"], optimizer_step)
    assert metrics["optimizer_step_skipped"] is True
    assert metrics["policy_version"] == version


def test_small_gradient_is_not_rounded_to_zero():
    trainer = bridge()
    optimizer = torch.optim.SGD(trainer.trainer_model.parameters(), lr=0.1)
    scored = batch()
    for item in scored.items:
        item.advantage = 1e-8
    metrics = trainer.train_step(scored, optimizer)
    assert 0 < metrics["grad_norm"] < 1e-6
    assert metrics["optimizer_step_skipped"] is False
    assert metrics["grpo_loss"] != 0


def test_centered_loss_can_be_zero_with_a_real_gradient():
    trainer = bridge()
    trainer.trainer_model.logits.data.zero_()
    scored = batch()
    for item in scored.items:
        item.rollout_logprobs = [-float(torch.tensor(4.0).log())]
    trainer.score_rollouts(scored, lambda p, r: 0.0 if r == "r" else 1.0)
    metrics = trainer.train_step(scored, torch.optim.SGD(trainer.trainer_model.parameters(), lr=0.1))
    assert metrics["grpo_loss"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["grad_norm"] > 0
    assert metrics["optimizer_step_skipped"] is False


@pytest.mark.parametrize("baseline_enabled", [True, False])
def test_long_trainer_reuses_advantage_handling(tmp_path, baseline_enabled, capsys):
    model = TinyPolicy()
    scores = [RewardBreakdown(total_score=r) for r in [0.225, 0.225, 0.425, 0.425]]
    with patch("sword.long_trainer.get_harness", return_value=None), \
         patch.object(GRPOBridge, "initialize_vllm"), \
         patch.object(GRPOBridge, "generate_rollouts", side_effect=lambda prompts: batch()), \
         patch.object(MultiMetricRewardScorer, "score", side_effect=scores):
        result = run_long_train(
            dataset=[{"problem": "p", "solution": "4", "harness": "opencode"}],
            model="mock", b_model=False, trainer_model=model,
            tokenizer=SimpleNamespace(pad_token_id=0), return_step_metrics=True,
            is_save_and_upload_checkpoint=False,
            Argument={"steps": 2, "concurrency_g": 2, "max_seq_len": 8,
                      "max_new_tokens": 1, "lr": 0.1, "output_dir": str(tmp_path),
                      "use_cross_step_baseline": baseline_enabled},
        )
    first, second = result["step_metrics"]
    assert first["zero_variance_groups"] == 1
    assert first["baseline_groups"] == 0
    assert first["optimizer_step_skipped"] is True
    assert second["baseline_groups"] == int(baseline_enabled)
    assert (second["grad_norm"] > 0) is baseline_enabled
    assert result["optimizer_steps"] == int(baseline_enabled)
    assert result["skipped_optimizer_steps"] == 2 - int(baseline_enabled)
    assert "RewardStd:" in capsys.readouterr().out


def test_oom_retry_does_not_use_failed_attempt_as_its_baseline(tmp_path):
    attempts = []
    original_train = GRPOBridge.train_step

    def train(trainer, scored, optimizer):
        attempts.append([item.advantage for item in scored.items])
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return original_train(trainer, scored, optimizer)

    scores = [RewardBreakdown(total_score=r) for r in [0.225, 0.225, 0.425, 0.425]]
    with patch.object(GRPOBridge, "initialize_vllm"), \
         patch.object(GRPOBridge, "generate_rollouts", side_effect=lambda prompts: batch()), \
         patch.object(GRPOBridge, "train_step", new=train), \
         patch.object(MultiMetricRewardScorer, "score", side_effect=scores):
        result = run_long_train(
            dataset=[{"problem": "p", "solution": "4"}], model="mock", b_model=False,
            trainer_model=TinyPolicy(), tokenizer=SimpleNamespace(pad_token_id=0),
            return_step_metrics=True, is_save_and_upload_checkpoint=False, show_detail_logs=False,
            Argument={"steps": 1, "concurrency_g": 2, "max_seq_len": 8,
                      "max_new_tokens": 1, "output_dir": str(tmp_path)},
        )
    assert attempts == [[0.0, 0.0], [0.0, 0.0]]
    assert result["step_metrics"][0]["baseline_groups"] == 0


def test_live_diagnostics_reject_all_zero_gradients():
    result = {
        "status": "completed", "final_step": 5, "visited_rows": 5,
        "rollout_backend": "vllm",
        "step_metrics": [{"grpo_loss": 0.0, "grad_norm": 0.0, "policy_version": i + 1}
                         for i in range(5)],
    }
    with pytest.raises(RuntimeError, match="no.*gradient|zero.*gradient"):
        check_training_result(result, 5, b_model=False, mock_test=False)


@pytest.mark.parametrize("initial_version", [0, 1])
def test_live_diagnostics_allow_skipped_first_step_and_zero_loss(initial_version):
    result = {
        "status": "completed", "final_step": 5, "visited_rows": 5,
        "rollout_backend": "vllm",
        "step_metrics": [{"grpo_loss": 0.0, "grad_norm": 0.0 if i == 0 else 0.2,
                          "policy_version": initial_version + i,
                          "optimizer_step_skipped": i == 0}
                         for i in range(5)],
    }
    check_training_result(result, 5, b_model=False, mock_test=False)


def test_live_diagnostics_reject_regularization_without_reward_signal():
    result = {
        "status": "completed", "final_step": 5, "visited_rows": 5,
        "rollout_backend": "vllm",
        "step_metrics": [{"grpo_loss": 0.1, "grad_norm": 0.2, "policy_version": i + 1,
                          "advantage_abs_max": 0.0} for i in range(5)],
    }
    with pytest.raises(RuntimeError, match="no reward-driven gradients"):
        check_training_result(result, 5, b_model=False, mock_test=False)


def test_explicitly_disabled_judge_never_constructs_or_calls_ollama():
    with patch("sword.reward.ModelBJudge") as judge:
        scorer = MultiMetricRewardScorer(b_model_judge=None)
        score = scorer.score("2+2?", "4", "The answer is 4", can_be_verify=True)
    judge.assert_not_called()
    assert score.truth == 1.0


@pytest.mark.parametrize("index", [0, 1])
def test_diagnostic_coding_truth_uses_semantic_judge(index):
    row = diagnostic_dataset()[index]
    judge = SimpleNamespace(evaluate=lambda **kwargs: ModelBEvaluationResult(truth=0.9))
    candidate = ("def is_prime(number):\n    if number < 2: return False\n"
                 "    return not any(number % divisor == 0 for divisor in range(2, number))"
                 if index == 0 else "def reverse_words(text):\n    return ' '.join(reversed(text.split()))")
    assert candidate != row["solution"]
    score = MultiMetricRewardScorer(judge).score(
        problem=row["problem"], solution=row["solution"], candidate_response=candidate,
        category=row["category"], harness_name=row["harness"],
    )
    assert score.truth == 0.9


def test_diagnostic_math_truth_uses_live_judge():
    row = diagnostic_dataset()[2]
    judge = SimpleNamespace(evaluate=lambda **kwargs: ModelBEvaluationResult(truth=0.9))
    score = MultiMetricRewardScorer(judge).score(
        problem=row["problem"], solution=row["solution"], candidate_response="The answer is 9",
        category=row["category"],
    )
    assert score.truth == 0.9


def test_diagnostic_math_without_judge_keeps_exact_verification():
    row = diagnostic_dataset()[2]
    score = MultiMetricRewardScorer(None).score(
        problem=row["problem"], solution=row["solution"], candidate_response="The answer is 9",
        category=row["category"],
    )
    assert score.truth == 1.0


def test_five_uniform_steps_are_visible_without_parameter_drift(tmp_path, capsys):
    model = TinyPolicy()
    initial = model.logits.detach().clone()
    with patch("sword.long_trainer.get_harness", return_value=None), \
         patch.object(GRPOBridge, "initialize_vllm"), \
         patch.object(GRPOBridge, "generate_rollouts", side_effect=lambda prompts: batch()), \
         patch.object(MultiMetricRewardScorer, "score", return_value=RewardBreakdown(total_score=0.225)):
        result = run_long_train(
            dataset=[{"problem": f"p{i}", "solution": "4", "harness": "opencode"} for i in range(5)],
            model="mock", b_model=False, trainer_model=model, tokenizer=SimpleNamespace(pad_token_id=0),
            return_step_metrics=True, is_save_and_upload_checkpoint=False,
            Argument={"steps": 5, "concurrency_g": 2, "max_seq_len": 8,
                      "max_new_tokens": 1, "lr": 0.1, "output_dir": str(tmp_path)},
        )
    assert result["final_step"] == 5
    assert result["optimizer_steps"] == 0
    assert result["skipped_optimizer_steps"] == 5
    assert torch.equal(model.logits, initial.to(model.logits.device))
    assert capsys.readouterr().out.count("Update: skipped") == 5
    with pytest.raises(RuntimeError, match="no nonzero gradients"):
        check_training_result(result, 5, b_model=False, mock_test=False)

