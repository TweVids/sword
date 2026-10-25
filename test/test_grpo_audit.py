"""Regressions for policy alignment, gradient correctness, and scorer parsing."""
import json
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.grpo_bridge import DualCorrectionGRPOLoss, FallbackRolloutEngine, VRAMLifecycleManager
from sword.rl.loss import ChunkedGRPOLoss
from sword.rl.policy import token_statistics, safe_optimizer_step
from sword.rotary_fallback import apply_rotary_fallback, homogeneous_head_dim
from sword.vllm_patch import patch_rotary_source, patch_head_dim_source, write_python_patch


class FixedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, -1.0, 3.0, 1.0]))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


class Tokenizer:
    eos_token_id = None

    def __call__(self, prompt, **kwargs):
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, ids, **kwargs):
        return str(ids)


def test_nucleus_sampling_uses_vocabulary_ids_and_raw_logprobs():
    model = FixedPolicy()
    result = FallbackRolloutEngine(model, Tokenizer()).generate(["p"], max_new_tokens=1, top_p=0.01, temperature=0.5)[0]
    assert result["response_token_ids"] == [2]
    assert result["rollout_logprobs"] == pytest.approx([model.logits.log_softmax(-1)[2].item()])


def test_entropy_floor_has_gradient_even_with_zero_advantage():
    model = FixedPolicy()
    loss_fn = DualCorrectionGRPOLoss(entropy_floor=10.0, entropy_coeff=1.0, chunk_size=1)
    loss, _ = loss_fn(model, torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.long), [1], torch.zeros(1))
    loss.backward()
    assert model.logits.grad is not None
    assert model.logits.grad.abs().sum() > 0
    # Gradient descent on the entropy penalty decreases the highest logit.
    assert model.logits.grad[2] > 0


def test_missing_rollout_logprobs_are_rejected():
    with pytest.raises(ValueError, match="every response token"):
        DualCorrectionGRPOLoss()(FixedPolicy(), torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.long), [1], torch.ones(1), [[-0.5]])


@pytest.mark.parametrize("with_lora", [False, True])
def test_chunked_decoder_matches_full_logits_and_parameter_gradients(with_lora):
    torch.manual_seed(7)
    cfg = AutoConfig.for_model("qwen2", vocab_size=17, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    if with_lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=0.01)
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    mask = torch.ones_like(ids)
    logits = model(ids, attention_mask=mask, use_cache=False).logits[:, :-1].float()
    full_lp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    full_lp.sum().backward()
    expected_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    # The causal-LM wrapper must never execute on the supported decoder path.
    with patch.object(model, "forward", side_effect=AssertionError("full logits allocated")):
        chunk_lp, _ = token_statistics(model, ids, mask, chunk_size=1)
        assert torch.allclose(chunk_lp, full_lp, atol=1e-6)
        chunk_lp.sum().backward()
    for name, param in model.named_parameters():
        if name in expected_grads:
            assert torch.allclose(param.grad, expected_grads[name], atol=2e-6, rtol=2e-5), name


def test_vram_offload_precedes_wake_and_failures_propagate():
    events = []
    model, engine = MagicMock(), MagicMock()
    model.to.side_effect = lambda device: events.append(device)
    engine.wake_up.side_effect = lambda: events.append("wake")
    mgr = VRAMLifecycleManager("cpu", offload_trainer=True)
    mgr.enter_rollout_phase(model, engine)
    assert events == ["cpu", "wake"]
    engine.sleep.side_effect = RuntimeError("sleep failed")
    with pytest.raises(RuntimeError, match="sleep failed"):
        mgr.enter_train_phase(model, engine)


@pytest.mark.parametrize("text", [
    json.dumps({"truth": 100, "formatted": -1, "rule_reasoning_follow": float("nan"), "cheating": "false"}),
    '[100, -1, NaN, 0.5, 0, "false"]',
])
def test_judge_scores_are_bounded_finite_and_parse_false(text):
    result = ModelBJudge().parse_scores(text)
    assert result.truth == 1.0
    assert result.formatted == 0.0
    assert result.rule_reasoning_follow == 0.0
    assert result.cheating is False


def test_empty_judge_response_does_not_award_default_scores():
    judge = ModelBJudge()
    with pytest.raises(ValueError):
        judge.parse_scores("thinking but no scores")
    assert judge.parse_scores('{"truth": 0.2}').formatted == 0.0


def test_memory_guard_prevents_forced_gpu_judge_request():
    judge = ModelBJudge(ModelBJudgeConfig(force_gpu=True))
    with patch.object(judge, "check_memory_guard", return_value=False), patch("sword.b_model.requests") as requests:
        assert judge.prepare_for_eval() is False
        result = judge.evaluate("p", "4", "", "", "14")
        requests.post.assert_not_called()
        assert result.truth < 1.0


def test_gdpo_column_and_component_inputs_have_identical_advantages():
    components = [{"ground_truth": 1.0, "output_format": 0.0, "safety": 0.0},
                  {"ground_truth": 0.0, "output_format": 1.0, "safety": -2.0}]
    columns = [{"accuracy": 1.0, "formatting": 0.0, "safety": 0.0},
               {"accuracy": 0.0, "formatting": 1.0, "safety": -2.0}]
    expected = ChunkedGRPOLoss.compute_gdpo_advantages(components)
    assert ChunkedGRPOLoss.compute_gdpo_advantages(columns) == expected
    assert ChunkedGRPOLoss.compute_gdpo_advantages([{"column_scores": c} for c in columns]) == expected
    assert expected[0][1] <= -2.0


def test_kl_and_old_policy_targets_are_detached_and_lengths_are_checked():
    lp = torch.tensor([[-0.5, -0.8]], requires_grad=True)
    old = torch.tensor([[-0.6, -0.9]], requires_grad=True)
    ref = torch.tensor([[-0.7, -1.0]], requires_grad=True)
    loss_fn = ChunkedGRPOLoss()
    loss, _ = loss_fn._compute_surrogate_loss(lp, torch.ones_like(lp, dtype=torch.bool), torch.ones(1), old, ref)
    loss.backward()
    assert lp.grad is not None
    assert old.grad is None and ref.grad is None
    with pytest.raises(ValueError, match="align"):
        loss_fn._compute_surrogate_loss(lp, torch.ones_like(lp, dtype=torch.bool), torch.ones(1), old[:, :1])


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_offsets_partial_dimensions_and_packed_sequences(interleaved):
    torch.manual_seed(2)
    # Distinct angles make ignoring offsets observable; only four of six dims rotate.
    angles = torch.arange(8).float()[:, None].expand(8, 2) * 0.1
    cos, sin = angles.cos(), angles.sin()
    x = torch.randn(2, 3, 2, 6)
    offsets = torch.tensor([1, 3])
    batch_result = apply_rotary_fallback(x, cos, sin, offsets, interleaved=interleaved)
    expected = x.clone()
    for b in range(2):
        for s in range(3):
            c, sn = cos[s + offsets[b]], sin[s + offsets[b]]
            if interleaved:
                a, z = x[b, s, :, :4:2], x[b, s, :, 1:4:2]
                expected[b, s, :, :4:2] = a * c - z * sn
                expected[b, s, :, 1:4:2] = a * sn + z * c
            else:
                a, z = x[b, s, :, :2], x[b, s, :, 2:4]
                expected[b, s, :, :2] = a * c - z * sn
                expected[b, s, :, 2:4] = a * sn + z * c
    assert torch.allclose(batch_result, expected)
    assert torch.equal(batch_result[..., 4:], x[..., 4:])
    packed = x.reshape(6, 2, 6)
    packed_result = apply_rotary_fallback(packed, cos, sin, offsets, torch.tensor([0, 3, 6]), interleaved=interleaved)
    assert torch.allclose(packed_result, expected.reshape_as(packed))
    recovered = apply_rotary_fallback(batch_result, cos, sin, offsets, interleaved=interleaved, conjugate=True)
    assert torch.allclose(recovered, x, atol=2e-6)
    inplace = x.clone()
    assert apply_rotary_fallback(inplace, cos, sin, offsets, interleaved=interleaved, inplace=True) is inplace
    assert torch.allclose(inplace, expected)


def test_rotary_patch_is_syntactically_valid_idempotent_and_executes_fallback():
    source = 'def run(x, cos, sin):\n    from flash_attn.ops.triton.rotary import apply_rotary\n    return apply_rotary(x, cos, sin, seqlen_offsets=1)\n'
    patched = patch_rotary_source(source)
    assert patch_rotary_source(patched) == patched
    namespace = {}
    # Explicitly force the import failure, independent of installed flash-attn.
    with patch.dict("sys.modules", {"flash_attn.ops.triton.rotary": None}):
        exec(patched, namespace)
        x = torch.ones(1, 1, 1, 4)
        angles = torch.tensor([[0., 0.], [0.5, 0.5]])
        assert torch.allclose(namespace["run"](x, angles.cos(), angles.sin()),
                              apply_rotary_fallback(x, angles.cos(), angles.sin(), 1))


def test_rotary_patch_upgrades_legacy_block_without_deleting_following_code():
    source = ('def run():\n    # marimo-rtx6000-rotary-fallback\n'
              '    try:\n        from flash_attn.ops.triton.rotary import apply_rotary\n'
              '    except ImportError:\n        def apply_rotary(x):\n            return x\n'
              '    return apply_rotary\n\ndef preserved():\n    return 7\n')
    patched = patch_rotary_source(source)
    assert 'def preserved()' in patched
    assert '    return apply_rotary\n' in patched
    compile(patched, '<test>', 'exec')


def test_head_dim_patch_preserves_indentation_and_rejects_heterogeneous_dimensions():
    from transformers.integrations.heterogeneity.configuration_utils import AmbiguousGlobalPerLayerAttributeError
    class Config:
        per_layer_config = [{"head_dim": 16}, {"head_dim": 16}]
        @property
        def head_dim(self):
            raise AmbiguousGlobalPerLayerAttributeError("per layer")
    config = Config()
    assert homogeneous_head_dim(config) == 16
    source = ('class C:\n    def get(self):\n        if True:\n'
              '            head_dim = getattr(self.hf_text_config, "head_dim", 0)\n'
              '            return head_dim\n')
    namespace = {}
    exec(patch_head_dim_source(source), namespace)
    converter = namespace['C']()
    converter.hf_text_config = config
    assert converter.get() == 16
    config.per_layer_config = [{"head_dim": 16}, {"head_dim": 32}]
    with pytest.raises(ValueError, match="Heterogeneous"):
        converter.get()


def test_invalid_patch_does_not_overwrite_file(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("value = 1\n")
    with pytest.raises(SyntaxError):
        write_python_patch(path, "def broken(:\n")
    assert path.read_text() == "value = 1\n"


def test_primary_scorer_does_not_match_numeric_substrings_or_lose_negative_sign():
    from sword.rl.scorer import PrimaryScorer
    scorer = PrimaryScorer()
    assert scorer._score_ground_truth("4", "14")[0] < 0
    assert scorer._score_ground_truth("-4", "4")[0] < 0
    assert scorer._score_ground_truth("4", "The answer is 4.")[0] > 0


def test_standalone_loss_respects_response_padding():
    from train_math_grpo import ChunkedGRPOLoss as StandaloneLoss
    model = FixedPolicy()
    ids = torch.tensor([[1, 2, 3, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    batch_loss, _ = StandaloneLoss().forward(model, ids, mask, [1], torch.ones(1))
    single_loss, _ = StandaloneLoss().forward_single(model, ids, mask, 1, 1.0)
    assert batch_loss.item() == pytest.approx(single_loss.item())


def test_math_decimal_and_boxed_answers_after_thinking():
    from sword.reward import MathVerifier
    assert MathVerifier.verify_answer("The answer is 3.14.", "3.14")
    assert not MathVerifier.verify_answer("The answer is 3.99.", "3.14")
    assert MathVerifier.verify_answer("<think>There were 8 choices.</think>After 3 steps, \\boxed{4}.", "4")
    assert MathVerifier.verify_answer("The answer is 1e-3.", "0.001")


def test_numeric_zero_dataset_solution_is_preserved():
    from sword.long_trainer import parse_dataset_row
    row = parse_dataset_row({"problem": "2 - 2", "solution": 0})
    assert row.solution == "0"
    assert row.can_be_verify is True


def test_bridge_microbatching_preserves_parameter_updates_and_loss_metrics():
    from sword.grpo_bridge import GRPOBridge, GRPOBridgeConfig, RolloutBatch, RolloutItem
    results = []
    for micro_bs in (1, 2):
        model = FixedPolicy()
        bridge = GRPOBridge(GRPOBridgeConfig(model_name="mock", device="cpu", concurrency_g=2,
                            max_model_len=16, max_new_tokens=4, micro_batch_size=micro_bs, max_grad_norm=0))
        bridge.initialize_trainer(model=model, tokenizer=Tokenizer())
        # Unequal response lengths expose incorrect averaging of normalized losses.
        batch = RolloutBatch(items=[RolloutItem("p", [1], "a", [2], [-0.8], advantage=1),
                                   RolloutItem("p", [1], "b", [3, 2], [-2.0, -0.8], advantage=-1)], concurrency_g=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        try:
            metrics = bridge.train_step(batch, optimizer)
            results.append((model.logits.detach().clone(), metrics["grpo_loss"]))
        finally:
            bridge.weight_sync.cleanup()
    assert torch.allclose(results[0][0], results[1][0])
    assert results[0][1] == pytest.approx(results[1][1], abs=2e-5)


def test_failed_adapter_export_keeps_previous_policy_version(tmp_path):
    from sword.grpo_bridge import WeightSynchronizer
    model = MagicMock()
    model.save_pretrained.side_effect = OSError("disk failure")
    sync = WeightSynchronizer()
    try:
        with pytest.raises(OSError, match="disk failure"):
            sync.sync_to_vllm(model)
        assert sync.policy_version == 0
        assert sync.current_adapter_path is None
    finally:
        sync.cleanup()


def test_optimizer_oom_restores_weights_and_existing_adam_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    safe_optimizer_step(optimizer, [parameter])
    saved_weight = parameter.detach().clone()
    saved_state = {key: value.clone() for key, value in optimizer.state[parameter].items()}
    parameter.grad = torch.ones_like(parameter)
    def partial_step():
        with torch.no_grad():
            parameter.add_(100)
        optimizer.state[parameter]["exp_avg"].add_(100)
        optimizer.state[parameter]["step"].add_(1)
        raise torch.cuda.OutOfMemoryError("out of memory during update")
    with patch.object(optimizer, "step", side_effect=partial_step):
        with pytest.raises(torch.cuda.OutOfMemoryError):
            safe_optimizer_step(optimizer, [parameter])
    assert torch.equal(parameter, saved_weight)
    for key, value in saved_state.items():
        assert torch.equal(optimizer.state[parameter][key], value)
    assert parameter.grad is None
    # Export can fail after the optimizer has successfully changed the policy.
    parameter.grad = torch.ones_like(parameter)
    def failed_export():
        raise torch.cuda.OutOfMemoryError("out of memory during adapter export")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        safe_optimizer_step(optimizer, [parameter], after_step=failed_export)
    assert torch.equal(parameter, saved_weight)
    for key, value in saved_state.items():
        assert torch.equal(optimizer.state[parameter][key], value)


def test_rollout_metadata_keeps_sampled_eos_and_prompt_tokens():
    from sword.server import FastServer
    server = SimpleNamespace(serve=MagicMock(return_value={
        "generated_texts": ["decoded without EOS", "second"],
        "prompt_token_ids": [[1, 2], [1, 2]],
        "generated_tokens": torch.tensor([[3, 0], [2, 0]]),
        "logprobs": torch.tensor([[-0.3, -0.1], [-0.4, -0.1]]),
    }))
    text, lps, metadata = FastServer.generate_rollouts(server, ["p"], 2, return_logprobs=True, return_token_ids=True, use_speculative=True)
    assert metadata == [[([1, 2], [3, 0]), ([1, 2], [2, 0])]]
    assert text == [["decoded without EOS", "second"]]
    assert lps[0][0].tolist() == pytest.approx([-0.3, -0.1])
    assert server.serve.call_args.kwargs["use_speculative"] is False


def test_legacy_rl_server_uses_shared_policy_fallback_with_exact_tokens():
    from sword.rl.engine import GRPOTrainer
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.model = FixedPolicy()
    trainer.tokenizer = Tokenizer()
    trainer.num_rollouts_per_prompt = 2
    def text_only_backend(prompts):
        raise AssertionError("decoded text backend must not be used for gradients")
    trainer.server = SimpleNamespace(generate_rollouts=text_only_backend)
    text, logprobs, metadata = trainer._generate_rollouts(["prompt"], 2)
    assert len(text[0]) == len(logprobs[0]) == len(metadata[0]) == 2
    for row, lp in zip(metadata[0], logprobs[0]):
        assert row[0] == [1]
        assert len(row[1]) == len(lp) == 2


@pytest.mark.parametrize("oom_first", [False, True])
def test_long_trainer_preserves_sampled_turn_and_retries_with_smaller_batches(tmp_path, oom_first):
    from sword.grpo_bridge import RolloutBatch, RolloutItem
    from sword.long_trainer import run_long_train
    from sword.reward import RewardBreakdown
    batch = RolloutBatch([RolloutItem("prompt", [1], "first turn", [2, 3], [-0.5, -1.0]),
                          RolloutItem("prompt", [1], "first turn", [3, 2], [-1.0, -0.5])], concurrency_g=2)
    judge = MagicMock()
    judge_loaded = [False]
    judge.prepare_for_eval.side_effect = lambda: judge_loaded.__setitem__(0, True)
    judge.release_after_eval.side_effect = lambda: judge_loaded.__setitem__(0, False)
    harness = MagicMock()
    harness.format_prompt.return_value = "prompt"
    def run_harness(**kwargs):
        assert not judge_loaded[0]
        return SimpleNamespace(final_text="final outcome", tools_called=1, tools_succeeded=1)
    harness.run.side_effect = run_harness
    def score(**kwargs):
        assert judge_loaded[0]
        assert kwargs["candidate_response"] == "final outcome"
        return RewardBreakdown(total_score=0.5)
    calls = []
    def train(bridge, supplied_batch, optimizer):
        assert not judge_loaded[0]
        assert supplied_batch.items[0].response_text == "first turn"
        assert supplied_batch.items[0].response_token_ids == [2, 3]
        assert supplied_batch.items[0].rollout_logprobs == [-0.5, -1.0]
        calls.append((bridge.config.micro_batch_size, bridge.loss_fn.chunk_size))
        if oom_first and len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return {"grpo_loss": 0.0, "step": 1}
    manager = MagicMock()
    manager.get_completed_dataset_hashes.return_value = set()
    manager.find_latest_checkpoint.return_value = None
    manager.save_checkpoint.return_value = str(tmp_path / "checkpoint")
    with patch("sword.long_trainer.CheckpointManager", return_value=manager), \
         patch("sword.long_trainer.ModelBJudge", return_value=judge), \
         patch("sword.long_trainer.get_harness", return_value=harness), \
         patch("sword.long_trainer.MultiMetricRewardScorer.score", side_effect=score), \
         patch("sword.grpo_bridge.GRPOBridge.generate_rollouts", side_effect=lambda prompts: copy.deepcopy(batch)), \
         patch("sword.grpo_bridge.GRPOBridge.train_step", new=train):
        result = run_long_train(dataset=[{"problem": "p", "solution": "4", "harness": "codex"}],
                                trainer_model=FixedPolicy(), tokenizer=Tokenizer(),
                                is_save_and_upload_checkpoint=False, show_detail_logs=False,
                                Argument={"steps": 1, "concurrency_g": 2, "micro_batch_size": 2,
                                          "chunk_size": 4, "max_seq_len": 16, "max_new_tokens": 4,
                                          "output_dir": str(tmp_path)})
    assert result["status"] == "completed"
    assert calls == ([(2, 4), (1, 2)] if oom_first else [(2, 4)])
