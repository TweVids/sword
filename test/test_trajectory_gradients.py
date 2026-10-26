"""Reward variation must train the assistant turns that actually differ."""
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sword.grpo_bridge import GRPOBridge, GRPOBridgeConfig, RolloutBatch, RolloutItem
from sword.harness.opencode import OpenCodeHarness
from sword.long_trainer import run_long_train
from sword.reward import MultiMetricRewardScorer, RewardBreakdown
from test.test_reward_stall import TinyPolicy


def trainer(model=None):
    bridge = GRPOBridge(GRPOBridgeConfig(concurrency_g=2, device="cpu", micro_batch_size=1, entropy_coeff=0))
    bridge.initialize_trainer(model=model or TinyPolicy(), tokenizer=SimpleNamespace(pad_token_id=0))
    return bridge


def identical_first_turns():
    return RolloutBatch(items=[
        RolloutItem("p", [1], "same tool request", [2], [-0.5], reward=0),
        RolloutItem("p", [1], "same tool request", [2], [-0.5], reward=1),
    ], concurrency_g=2)


def test_varying_rewards_on_identical_first_turns_can_cancel_exactly():
    bridge = trainer()
    try:
        scored = identical_first_turns()
        bridge.compute_advantages(scored)
        metrics = bridge.train_step(scored, torch.optim.SGD(bridge.trainer_model.parameters(), lr=0.1))
        assert metrics["reward_std"] == 0.5
        assert metrics["advantage_abs_max"] > 0.99
        assert metrics["grad_norm"] == 0
        assert metrics["unique_first_turns"] == metrics["unique_training_trajectories"] == 1
        assert metrics["tensors_with_grad"] == 1 and metrics["tensors_with_nonzero_grad"] == 0
    finally:
        bridge.weight_sync.cleanup()


def test_differing_followups_create_a_reward_increasing_update():
    bridge = trainer()
    try:
        scored = identical_first_turns()
        for item, token in zip(scored.items, [2, 3]):
            item.followup_turns.append(RolloutItem("tool result", [1, 2, 0], str(token), [token], [-0.5]))
        before = bridge.trainer_model.logits.softmax(-1)[3].item()
        bridge.compute_advantages(scored)
        metrics = bridge.train_step(scored, torch.optim.SGD(bridge.trainer_model.parameters(), lr=0.1))
        assert metrics["grad_norm"] > 0 and not metrics["optimizer_step_skipped"]
        assert metrics["training_turns"] == 4 and metrics["followup_training_turns"] == 2
        assert metrics["response_token_count"] == 4  # observations/context are not targets
        assert metrics["unique_first_turns"] == 1 and metrics["unique_training_trajectories"] == 2
        assert bridge.trainer_model.logits.softmax(-1)[3].item() > before
    finally:
        bridge.weight_sync.cleanup()


def test_followup_versions_and_logprob_lengths_are_validated():
    bridge = trainer()
    try:
        scored = identical_first_turns()
        followup = RolloutItem("feedback", [1], "answer", [3], [-0.5], policy_version=1)
        scored.items[0].followup_turns.append(followup)
        optimizer = torch.optim.SGD(bridge.trainer_model.parameters(), lr=0.1)
        with pytest.raises(ValueError, match="policy versions.*follow-up"):
            bridge.train_step(scored, optimizer)
        followup.policy_version = 0
        followup.rollout_logprobs = []
        with pytest.raises(ValueError, match="match every response token"):
            bridge.train_step(scored, optimizer)
    finally:
        bridge.weight_sync.cleanup()


def test_actual_peft_adapter_receives_followup_gradients():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(12)
    config = AutoConfig.for_model("qwen2", vocab_size=16, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = get_peft_model(AutoModelForCausalLM.from_config(config), LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_dropout=0))
    bridge = trainer(model)
    try:
        scored = identical_first_turns()
        for item, token in zip(scored.items, [2, 3]):
            item.followup_turns.append(RolloutItem("feedback", [1, 2, 0], "answer", [token], [-math.log(16)]))
        bridge.compute_advantages(scored)
        before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        metrics = bridge.train_step(scored, torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1))
        assert metrics["grad_norm"] > 0 and metrics["tensors_with_nonzero_grad"] > 0
        assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name in before)
    finally:
        bridge.weight_sync.cleanup()


def test_real_bitsandbytes_8bit_lora_adapter_has_reward_gradients():
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(19)
    config = AutoConfig.for_model("qwen2", vocab_size=16, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = AutoModelForCausalLM.from_config(config)
    attention = model.model.layers[0].self_attn
    for name in ("q_proj", "v_proj"):
        original = getattr(attention, name)
        quantized = bnb.nn.Linear8bitLt(16, 16, bias=original.bias is not None, has_fp16_weights=False)
        quantized.weight = bnb.nn.Int8Params(original.weight.detach().clone(), requires_grad=False, has_fp16_weights=False)
        if original.bias is not None:
            quantized.bias = torch.nn.Parameter(original.bias.detach().clone(), requires_grad=False)
        setattr(attention, name, quantized.to("cpu"))
    model.is_loaded_in_8bit = True
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    bridge = trainer(model)
    try:
        scored = identical_first_turns()
        for item, token in zip(scored.items, [2, 3]):
            item.followup_turns.append(RolloutItem("feedback", [1, 2, 0], "answer", [token], [-math.log(16)]))
        bridge.compute_advantages(scored)
        before = {name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
        metrics = bridge.train_step(scored, torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1))
        assert metrics["grad_norm"] > 0 and metrics["tensors_with_nonzero_grad"] > 0
        assert any(not torch.equal(before[name], p) for name,p in model.named_parameters() if name in before)
    finally:
        bridge.weight_sync.cleanup()


def test_zero_importance_weights_are_reported_separately_from_zero_advantages():
    model = TinyPolicy()
    model.logits.data.copy_(torch.tensor([-1000., 1000., -1000., -1000.]))
    bridge = trainer(model)
    try:
        scored = identical_first_turns()
        bridge.compute_advantages(scored)
        metrics = bridge.train_step(scored, torch.optim.SGD(model.parameters(), lr=0.1))
        assert metrics["advantage_abs_max"] > 0
        assert metrics["importance_zero_fraction"] == 1
        assert metrics["policy_logprob_min"] < -1000
        assert metrics["zero_gradient_reason"] == "zero_importance_weights"
    finally:
        bridge.weight_sync.cleanup()


def test_disconnected_trainable_parameters_have_an_explicit_diagnosis():
    class DetachedPolicy(TinyPolicy):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.logits.detach().clone().requires_grad_().expand(*input_ids.shape, -1))
    bridge = trainer(DetachedPolicy())
    try:
        scored = identical_first_turns()
        bridge.compute_advantages(scored)
        metrics = bridge.train_step(scored, torch.optim.SGD(bridge.trainer_model.parameters(), lr=0.1))
        assert metrics["tensors_with_grad"] == 0
        assert metrics["zero_gradient_reason"] == "disconnected_trainable_parameters"
    finally:
        bridge.weight_sync.cleanup()


def test_long_trainer_keeps_exact_followup_tokens_and_trains_them(tmp_path):
    model = TinyPolicy()
    initial = '<tool_call>{"action":"write","parameters":{"path":"state.txt","content":"ready"}}</tool_call>'
    calls = []
    def generate(bridge, prompts):
        calls.append(prompts)
        if len(calls) == 1:
            result = identical_first_turns()
            for item in result.items:
                item.prompt_text = prompts[0]
                item.response_text = initial
            return result
        token = 2 if len(calls) == 2 else 3
        return RolloutBatch([RolloutItem(prompts[0], [1, 2, 0], f"Final Answer: {token}", [token], [-0.5])], concurrency_g=1)
    harness = OpenCodeHarness()
    harness.get_system_prompt = lambda: "Use tools and provide a final answer."
    before = model.logits.softmax(-1)[3].item()
    with patch("sword.long_trainer.get_harness", return_value=harness), \
         patch.object(GRPOBridge, "initialize_vllm"), \
         patch.object(GRPOBridge, "generate_rollouts", autospec=True, side_effect=generate), \
         patch.object(MultiMetricRewardScorer, "score", side_effect=[RewardBreakdown(total_score=0), RewardBreakdown(total_score=1)]):
        result = run_long_train(
            dataset=[{"problem":"p", "solution":"3", "harness":"opencode"}], model="mock", b_model=False,
            trainer_model=model, tokenizer=SimpleNamespace(pad_token_id=0), trace_rollouts=True,
            return_step_metrics=True, is_save_and_upload_checkpoint=False, show_detail_logs=False,
            Argument={"steps":1, "concurrency_g":2, "max_seq_len":1024, "max_new_tokens":1, "lr":0.1,
                      "output_dir":str(tmp_path)},
        )
    metrics = result["step_metrics"][0]
    assert metrics["grad_norm"] > 0 and metrics["training_turns"] == 4
    assert model.logits.softmax(-1)[3].item() > before
    evidence = [json.loads(line) for line in (tmp_path / "rollout_details.jsonl").read_text().splitlines()]
    assert [item["training_turns"][1]["response_token_ids"] for item in evidence] == [[2], [3]]
    assert all(item["training_turns"][1]["prompt_token_ids"] == [1, 2, 0] for item in evidence)
    assert all("ready" in item["training_turns"][1]["prompt_text"] for item in evidence)
    assert evidence[0]["advantage"] < 0 < evidence[1]["advantage"]


def test_nonuniform_skipped_step_prints_gradient_diagnostics(tmp_path, capsys):
    with patch("sword.long_trainer.get_harness", return_value=None), \
         patch.object(GRPOBridge, "initialize_vllm"), \
         patch.object(GRPOBridge, "generate_rollouts", side_effect=lambda prompts: identical_first_turns()), \
         patch.object(MultiMetricRewardScorer, "score", side_effect=[RewardBreakdown(total_score=0), RewardBreakdown(total_score=1)]):
        run_long_train(
            dataset=[{"problem":"p", "solution":"3"}], model="mock", b_model=False,
            trainer_model=TinyPolicy(), tokenizer=SimpleNamespace(pad_token_id=0),
            is_save_and_upload_checkpoint=False,
            Argument={"steps":1, "concurrency_g":2, "max_seq_len":8, "max_new_tokens":1, "output_dir":str(tmp_path)},
        )
    output = capsys.readouterr().out
    assert "Zero-gradient step" in output and "unique first turns 1/2" in output
    assert "Gradient signal" in output and "Policy statistics" in output
