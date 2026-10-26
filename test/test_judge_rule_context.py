"""Complete judge scores, preserved rule evidence, and shared context budgets."""
import json
import math
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sword
from sword.b_model import ModelBJudge, ModelBJudgeConfig, model_b_config
from sword.defaults import DEFAULT_B_CONTEXT, DEFAULT_GRPO_CONTEXT, DEFAULT_GRPO_MAX_NEW_TOKENS
from sword.grpo_bridge import (
    GRPOBridge, GRPOBridgeConfig, RolloutBatch, RolloutItem, format_rollout_prompt,
    has_thought_section, restore_thinking_prefix, tokenize_rollout_prompt,
)
from sword.harness.protocol import visible_response, structured_calls
from sword.long_trainer import parse_dataset_row, policy_task_prompt
from sword.reward import MultiMetricRewardScorer
from test.test_runtime_setup import ready_harnesses, training_result


SCORES = {"truth": 1.0, "formatted": 1.0, "rule_reasoning_follow": 0.5,
          "rule_answer_follow": 1.0, "tools_success": 0.75, "cheating": False}


@pytest.mark.parametrize("missing", list(SCORES))
@pytest.mark.parametrize("reasoning", [True, False])
def test_incomplete_live_json_retries_with_explicit_repair_and_preserves_evidence(missing, reasoning):
    judge = ModelBJudge(ModelBJudgeConfig(auto_setup=False, fallback_model_name=None, reasoning=reasoning))
    judge.ensure_ready = lambda: True
    seen = []
    incomplete = {key: value for key, value in SCORES.items() if key != missing}
    rule = "RULE_START\n" + "Use the stated deduction and verify the result.\n" * 900 + "RULE_END"
    history = ["<think>The first step uses the dataset deduction.</think>", "I verified the intermediate result."]
    candidate = "Final Answer: " + "long answer details " * 700 + "ANSWER_END"

    def request(payload):
        seen.append(payload)
        return {"message": {"content": json.dumps(incomplete if len(seen) == 1 else SCORES)},
                "prompt_eval_count": 15000}

    with patch.object(judge, "_request_chat", side_effect=request):
        result = judge.evaluate("problem", "reference", rule, "Include ANSWER_END", candidate,
                                tools_called=4, tools_succeeded=3, assistant_history=history)
    assert len(seen) == 2 and result.to_dict() == SCORES
    assert [attempt["status"] for attempt in judge.last_evaluation_attempts] == ["failed", "scored"]
    assert judge.last_evaluation_attempts[0]["prompt_eval_count"] == 15000
    assert seen[-1]["think"] is False
    for payload in seen:
        assert payload["options"]["num_ctx"] == 32768
        assert set(payload["format"]["required"]) == set(SCORES)
        assert payload["format"]["additionalProperties"] is False
        evidence = payload["messages"][1]["content"]
        assert rule in evidence and candidate in evidence
        assert json.dumps(history) in evidence and "tools_success=0.75" in evidence
    assert missing in seen[-1]["messages"][-1]["content"]
    assert "COMPLETE JSON" in seen[-1]["messages"][-1]["content"]


@pytest.mark.parametrize("changes", [
    {"truth": None}, {"truth": "1.0"}, {"truth": True}, {"truth": float("nan")},
    {"truth": -0.1}, {"truth": 1.1}, {"cheating": 0}, {"extra": "comment"},
])
def test_invalid_live_values_cannot_be_silently_coerced_into_rewards(changes):
    judge = ModelBJudge()
    with pytest.raises(ValueError):
        judge.validate_live_scores(json.dumps({**SCORES, **changes}))


def test_policy_gets_long_rules_without_reference_solution():
    reasoning = "Keep every stated invariant.\n" * 400 + "REASONING_TAIL"
    answer = "Required answer format.\n" * 400 + "ANSWER_RULE_TAIL"
    row = parse_dataset_row({"problem": "TASK", "solution": "PRIVATE_REFERENCE_SENTINEL",
                             "reasoning_rule": reasoning, "answer_rule": answer})
    prompt = policy_task_prompt(row)
    assert prompt.startswith("TASK") and reasoning in prompt and answer in prompt
    assert "PRIVATE_REFERENCE_SENTINEL" not in prompt
    assert policy_task_prompt(parse_dataset_row({"problem": "plain", "solution": "4"})) == "plain"


def test_reward_scorer_passes_prior_reasoning_separately_from_final_answer():
    evaluate = SimpleNamespace()
    with patch.object(evaluate, "evaluate", create=True, return_value=SimpleNamespace(
        truth=1, rule_reasoning_follow=1, rule_answer_follow=1, cheating=False,
        raw_response="", to_dict=lambda: SCORES,
    )) as call:
        MultiMetricRewardScorer(evaluate).score(
            "task", "4", "Final Answer: 4", reasoning_rule="Show addition", answer_rule="A number",
            assistant_history=["<think>2 + 2 = 4</think>"], can_be_verify=False,
        )
    assert call.call_args.kwargs["candidate_response"] == "Final Answer: 4"
    assert call.call_args.kwargs["assistant_history"] == ["<think>2 + 2 = 4</think>"]
    assert call.call_args.kwargs["reasoning_rule"] == "Show addition"


def test_shared_context_defaults_and_explicit_overrides():
    config = GRPOBridgeConfig()
    assert config.max_model_len == DEFAULT_GRPO_CONTEXT == 12288
    assert config.max_new_tokens == DEFAULT_GRPO_MAX_NEW_TOKENS == 2048
    assert config.micro_batch_size == 1 and config.offload_trainer_during_rollout
    assert config.vllm_max_num_seqs == 2 and config.vllm_max_num_batched_tokens == 2048
    assert ModelBJudgeConfig().num_ctx == model_b_config().num_ctx == DEFAULT_B_CONTEXT == 32768
    assert model_b_config(B_model_num_ctx=16384).num_ctx == 16384
    assert GRPOBridgeConfig(max_model_len=8192, max_new_tokens=1024).max_model_len == 8192


def test_vllm_context_uses_bounded_concurrency_and_chunked_prefill():
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.ones(2, 2))
    bridge = GRPOBridge(GRPOBridgeConfig(device="cpu"))
    seen = []
    try:
        bridge.initialize_trainer(model=Adapter(), tokenizer=SimpleNamespace(pad_token_id=0))
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(LLM=lambda **kw: seen.append(kw) or object())}), \
             patch.object(bridge.weight_sync, "sync_to_vllm", return_value=1):
            assert bridge.initialize_vllm()
        assert seen[0]["max_model_len"] == 12288
        assert seen[0]["max_num_seqs"] == 2 and seen[0]["max_num_batched_tokens"] == 2048
        assert seen[0]["enable_chunked_prefill"] is True
    finally:
        bridge.weight_sync.cleanup()


@pytest.mark.parametrize("explicit", [False, True])
def test_diagnostics_use_shared_context_and_keep_mock_responses_short(explicit):
    with patch("sword.diagnostics.prepare_harnesses", return_value=ready_harnesses()), \
         patch("sword.diagnostics.check_harness_tools", return_value={}), \
         patch("sword.long_trainer.run_long_train", return_value=training_result()) as train:
        report = sword.train(prep_and_logs=True, test=True, mock_test=True, b_model=False,
                             Argument={"max_new_tokens": 2048} if explicit else None)
    assert report["ok"]
    args = train.call_args.kwargs["Argument"]
    assert args["max_seq_len"] == 12288 and args["max_new_tokens"] == (2048 if explicit else 64)
    assert args["micro_batch_size"] == 1


def test_judge_guide_does_not_invent_missing_rules_or_require_think_tags():
    judge = ModelBJudge()
    prompt = judge.build_user_prompt("task", "4", "", "", "Final Answer: 4")
    assert "No reasoning rule specified" in prompt and "No answer rule specified" in prompt
    assert "Follow step by step deduction" not in prompt
    assert "tags are optional unless a dataset rule requires them" in judge.SYSTEM_PROMPT
    assert "BOTH earlier assistant turns" in judge.SYSTEM_PROMPT


def native_thinking_tokenizer(channel):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    vocab = {"[BOS]":0, "[USER]":1, "p":2, "[ASSISTANT]":3, "<think>":4, "</think>":5,
             "<|channel>":6, "thought":7, "<channel|>":8, "Final":9, "Answer:":10,
             "deduction":11, "[EOS]":12, "[UNK]":13, "verification":14, "[PAD]":15}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    opener = "<|channel> thought" if channel else "<think>"
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]",
        additional_special_tokens=["<think>", "</think>", "<|channel>", "<channel|>"],
        chat_template="{{ bos_token }} [USER] {{ messages[0]['content'] }} [ASSISTANT] "
                      + "{% if enable_thinking %}" + opener + "{% endif %}",
    )


@pytest.mark.parametrize("channel", [False, True])
def test_thinking_option_reaches_native_template_and_prefix_is_restored(channel):
    tokenizer = native_thinking_tokenizer(channel)
    rendered, _ = format_rollout_prompt(tokenizer, "p")
    assert ("thought" if channel else "<think>") in rendered
    disabled, _ = format_rollout_prompt(tokenizer, "p", enable_thinking=False)
    assert "thought" not in disabled and "<think>" not in disabled
    prefix = "<|channel> thought" if channel else "<think>"
    prompt_ids = tokenize_rollout_prompt(tokenizer, "p")[0]["input_ids"][0].tolist()
    closing = "<channel|>" if channel else "</think>"
    text = restore_thinking_prefix(tokenizer, prompt_ids, f"Reasoning here {closing}\nFinal Answer: p")
    assert text.startswith(prefix) and has_thought_section(text)
    assert "Reasoning here" not in visible_response(text)
    assert "Final Answer: p" in visible_response(text)
    assert not has_thought_section(prefix + closing)


def test_prompt_opened_thought_tool_text_is_not_executed():
    tokenizer = native_thinking_tokenizer(False)
    prompt_ids = tokenize_rollout_prompt(tokenizer, "p")[0]["input_ids"][0].tolist()
    response = '<tool_call>{"name":"write_file","arguments":{"path":"unsafe.txt","content":"draft"}}</tool_call>'
    text = restore_thinking_prefix(tokenizer, prompt_ids, response + "</think>\nFinal Answer: p")
    assert structured_calls(text) == []


def test_reward_difference_in_thought_tokens_drives_actual_backward():
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(16))
        def forward(self, input_ids, **kw):
            return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))
    model = Policy()
    tokenizer = native_thinking_tokenizer(False)
    bridge = GRPOBridge(GRPOBridgeConfig(concurrency_g=2, device="cpu", entropy_coeff=0))
    try:
        bridge.initialize_trainer(model=model, tokenizer=tokenizer)
        prompt_ids = tokenize_rollout_prompt(tokenizer, "p")[0]["input_ids"][0].tolist()
        items = []
        for reward, thought_id in [(0, 11), (1, 14)]:
            ids = [thought_id, 5, 9, 10, 2, 12]  # identical final answers, differing thoughts
            text = restore_thinking_prefix(tokenizer, prompt_ids, tokenizer.decode(ids, skip_special_tokens=False))
            items.append(RolloutItem("p", prompt_ids, text, ids, [-math.log(16)] * len(ids), reward=reward))
        batch = RolloutBatch(items=items, concurrency_g=2)
        bridge.compute_advantages(batch)
        metrics = bridge.train_step(batch, torch.optim.SGD(model.parameters(), lr=0.1))
        assert metrics["grad_norm"] > 0 and metrics["reasoning_turns"] == 2
        assert model.logits[14] > model.logits[11]  # reward trains the distinguishing thought tokens
        assert visible_response(items[0].response_text) == visible_response(items[1].response_text)
    finally:
        bridge.weight_sync.cleanup()
