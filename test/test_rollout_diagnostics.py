"""Regressions for uniform live rewards: generation format and score evidence."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.grpo_bridge import FallbackRolloutEngine, GRPOBridge, GRPOBridgeConfig, tokenize_rollout_prompt
from sword.long_trainer import run_long_train
from sword.harness.base import BaseHarness
from sword.reward import MultiMetricRewardScorer, RewardBreakdown
from test.test_reward_stall import TinyPolicy, batch


def chat_tokenizer():
    backend = Tokenizer(WordLevel({"[BOS]": 0, "[USER]": 1, "p": 2, "[ASSISTANT]": 3, "[UNK]": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 0)])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="[BOS]", unk_token="[UNK]",
        chat_template="{{ bos_token }} [USER] {{ messages[0]['content'] }} [ASSISTANT]",
    )


def test_fallback_uses_chat_roles_and_does_not_duplicate_bos():
    result = FallbackRolloutEngine(TinyPolicy(), chat_tokenizer()).generate(
        ["p"], max_new_tokens=1, top_p=0.01, max_model_len=16,
    )[0]
    assert result["prompt_token_ids"] == [0, 1, 2, 3]
    assert result["response_token_ids"] == [2]


@pytest.mark.parametrize("fail_vllm", [False, True])
def test_vllm_and_fallback_use_same_exact_chat_prompt(fail_vllm):
    trainer = GRPOBridge(GRPOBridgeConfig(model_name="mock", concurrency_g=2, device="cpu", max_model_len=16, max_new_tokens=1))
    trainer.initialize_trainer(model=TinyPolicy(), tokenizer=chat_tokenizer())
    engine = MagicMock()

    def generate(prompts, sampling, **kwargs):
        assert prompts == [{"prompt_token_ids": [0, 1, 2, 3]}] * 2
        if fail_vllm:
            raise RuntimeError("backend failed")
        result = SimpleNamespace(prompt_token_ids=[0, 1, 2, 3], outputs=[SimpleNamespace(
            text="p", token_ids=[2], logprobs=[{2: SimpleNamespace(logprob=-0.5)}], finish_reason="stop",
        )])
        return [result, result]

    engine.generate.side_effect = generate
    trainer.vllm_engine = engine
    try:
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=lambda **kwargs: kwargs)}):
            scored = trainer.generate_rollouts(["p"])
        assert [item.prompt_token_ids for item in scored.items] == [[0, 1, 2, 3]] * 2
        assert all(item.prompt_text == "p" for item in scored.items)
    finally:
        trainer.weight_sync.cleanup()


def test_overlong_chat_prompt_is_rejected_instead_of_losing_assistant_marker():
    with pytest.raises(ValueError, match="prompt token budget"):
        tokenize_rollout_prompt(chat_tokenizer(), "p p p p", max_prompt_tokens=4)


def test_plain_tokenizer_contract_is_preserved():
    class PlainTokenizer:
        def __call__(self, prompt, **kwargs):
            assert prompt == "p"
            assert kwargs == {"return_tensors": "pt", "truncation": True, "max_length": 8}
            return {"input_ids": torch.tensor([[1, 2]])}
    inputs, is_chat = tokenize_rollout_prompt(PlainTokenizer(), "p", 8)
    assert not is_chat
    assert inputs["input_ids"].tolist() == [[1, 2]]


def test_harness_recounts_real_tokens_and_never_drops_the_task():
    class DenseTokenizer:
        def encode(self, text, **kwargs):
            return list(text)
    harness = BaseHarness()
    harness.get_system_prompt = lambda: "Instructions. " * 200 + "\n\n# Gemini CLI Tool Declarations:\n" + "long-tool-line" * 400
    result = harness.format_prompt("Solve 2+2.", tokenizer=DenseTokenizer(), max_tokens=100)
    assert len(result) <= 100
    assert result.endswith("Task:\nSolve 2+2.")


def test_harness_rejects_a_task_that_cannot_fit():
    class DenseTokenizer:
        def encode(self, text, **kwargs):
            return list(text)
    harness = BaseHarness()
    harness.get_system_prompt = lambda: "Guidance"
    with pytest.raises(ValueError, match="task alone exceeds"):
        harness.format_prompt("very long task" * 100, tokenizer=DenseTokenizer(), max_tokens=10)


def judge_with_reply(reply):
    judge = ModelBJudge(ModelBJudgeConfig(auto_setup=False, fallback_model_name=None))
    judge.ensure_ready = lambda: True
    judge._request_chat = reply
    return judge


def test_partial_live_judge_scores_retry_without_thinking():
    calls = []
    complete = {"truth": 1, "formatted": 1, "rule_reasoning_follow": 1, "rule_answer_follow": 1,
                "tools_success": 0, "cheating": False}
    def reply(payload):
        calls.append(payload)
        data = {"formatted": 1} if len(calls) == 1 else complete
        return {"message": {"content": json.dumps(data)}}
    judge = judge_with_reply(reply)
    result = judge.evaluate("2+2", "4", "explain", "answer", "4")
    assert [call["think"] for call in calls] == [True, False]
    assert result.truth == 1
    assert judge.evaluation_counts == {judge.config.model_name: 1}


def test_partial_judge_does_not_report_successful_live_evaluation():
    judge = judge_with_reply(lambda payload: {"message": {"content": '{"formatted": 1}'}})
    judge.evaluate("2+2", "4", "", "", "4")
    assert judge.last_evaluation_source == "deterministic"
    assert judge.evaluation_counts == {"deterministic": 1}
    assert "omitted required scores" in judge.last_error


def test_complete_zero_judge_scores_are_accepted_without_fabricating_signal():
    reply = {"truth": 0, "formatted": 0, "rule_reasoning_follow": 0, "rule_answer_follow": 0,
             "tools_success": 0, "cheating": False}
    judge = judge_with_reply(lambda payload: {"message": {"content": json.dumps(reply)}})
    result = judge.evaluate("2+2", "4", "", "", "bad answer")
    assert result.truth == 0
    assert not result.missing_scores
    assert judge.last_evaluation_source == judge.config.model_name


def test_blank_output_earns_no_formatting_or_fallback_correctness_reward():
    scorer = MultiMetricRewardScorer(None)
    assert scorer.check_formatting(" \n ") == 0
    result = scorer.score("2+2", "4", "", can_be_verify=False, harness_name="gemini")
    assert result.total_score == 0
    assert result.truth == result.rule_reasoning_follow == result.rule_answer_follow == 0


def test_stalled_step_prints_and_persists_candidate_and_judge_evidence(tmp_path, capsys):
    judge_scores = {"truth": 0, "formatted": 0, "rule_reasoning_follow": 0, "rule_answer_follow": 0}
    breakdown = RewardBreakdown(formatted=1, total_score=0.2, judge_scores=judge_scores,
                                judge_response=json.dumps(judge_scores), judge_source="live-judge")
    def generate(prompts):
        result = batch()
        for item in result.items:
            item.response_text = "No answer yet."
            item.is_truncated = True
        return result
    with patch("sword.long_trainer.get_harness", return_value=None), \
         patch.object(GRPOBridge, "initialize_vllm"), \
         patch.object(GRPOBridge, "generate_rollouts", side_effect=generate), \
         patch.object(MultiMetricRewardScorer, "score", return_value=breakdown):
        result = run_long_train(
            dataset=[{"problem": "p", "solution": "4", "harness": "gemini"}], model="mock", b_model=False,
            trainer_model=TinyPolicy(), tokenizer=SimpleNamespace(pad_token_id=0),
            return_step_metrics=True, trace_rollouts=True, is_save_and_upload_checkpoint=False,
            Argument={"steps": 1, "concurrency_g": 2, "max_seq_len": 8, "max_new_tokens": 1, "output_dir": str(tmp_path)},
        )
    output = capsys.readouterr().out
    assert "[ModelB JSON 0]" in output
    assert "No answer yet." in output
    assert "unique responses 1/2" in output and "truncated 2" in output
    records = [json.loads(line) for line in (tmp_path / "rollout_details.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["candidate_response"] == "No answer yet."
    assert records[0]["judge_scores"] == judge_scores
    assert records[0]["response_token_ids"] == [2]
    assert result["step_metrics"][0]["unique_responses"] == 1
