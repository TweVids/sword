"""Observational streaming must preserve rollouts and training updates."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sword.grpo_bridge import GRPOBridge, GRPOBridgeConfig, FallbackRolloutEngine
from sword.long_trainer import run_long_train
from sword.training_logger import TrainingLogger, observe_vllm_generation
from test.test_reward_stall import TinyPolicy, batch


def test_vllm_observer_streams_before_completion_and_preserves_objects():
    partial = SimpleNamespace(request_id="91", finished=False, outputs=[SimpleNamespace(text="partial")])
    final = SimpleNamespace(request_id="91", finished=True, outputs=[SimpleNamespace(text="complete")])
    class Engine:
        def add_request(self, request_id, prompt):
            return request_id
        def step(self):
            return [partial] if not seen else [final]
    engine = Engine()
    original_step, original_add = engine.step, engine.add_request
    seen = []
    with observe_vllm_generation(SimpleNamespace(llm_engine=engine), lambda i, text, **kw: seen.append((i, text, kw))):
        assert engine.add_request("91", "prompt") == "91"
        assert engine.step()[0] is partial
        assert seen[0][1] == "partial"
        assert engine.step()[0] is final
    assert seen[-1][2]["finished"]
    assert engine.step == original_step and engine.add_request == original_add


def test_observer_callback_failure_and_generation_failure_restore_engine():
    engine = SimpleNamespace(add_request=lambda *args: None,
                             step=lambda: [SimpleNamespace(request_id="0", finished=True, outputs=[SimpleNamespace(text="answer")])])
    original = engine.step
    with pytest.raises(RuntimeError):
        with observe_vllm_generation(SimpleNamespace(llm_engine=engine), lambda *args, **kw: 1 / 0):
            engine.add_request("0", "p")
            engine.step()
            raise RuntimeError("generation failure")
    assert engine.step is original


def test_observer_requests_live_delivery_when_offline_api_forces_final_only():
    received = []
    engine = SimpleNamespace(add_request=lambda request_id, prompt, params: received.append(params.output_kind),
                             step=lambda: [])
    fake_sampling = SimpleNamespace(RequestOutputKind=SimpleNamespace(CUMULATIVE="cumulative"))
    with patch.dict(sys.modules, {"vllm.sampling_params": fake_sampling}):
        with observe_vllm_generation(SimpleNamespace(llm_engine=engine), lambda *args, **kw: None):
            engine.add_request("0", "prompt", SimpleNamespace(output_kind="final_only"))
    assert received == ["cumulative"]


def test_logger_bounds_generation_queue_and_tracks_all_rollout_scores():
    logger = TrainingLogger(max_rollouts=2)
    for _ in range(200):
        logger.generation(0, "x" * 100000)
    assert logger.events.qsize() <= 128
    logger.generation(4, "not shown")
    logger.publish("step", step=1, scores=[1., 0., 0.5, 0.5])
    while not logger.events.empty():
        event, data = logger.events.get_nowait()
        logger._apply(event, data)
    state = logger.snapshot()
    assert len(state["rollouts"][0]["text"]) <= 32768
    assert 4 not in state["rollouts"]
    assert state["scored_rollouts"] == 4 and state["overall_mean"] == 0.5
    assert state["history"][0]["total_score"] == 2


def test_streaming_preserves_pytorch_sampling_and_random_state():
    class Tokenizer:
        eos_token_id = None
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1]])}
        def decode(self, ids, **kwargs):
            return "".join(str(i) for i in ids)
    engine = FallbackRolloutEngine(TinyPolicy(), Tokenizer())
    torch.manual_seed(77)
    plain = engine.generate(["p", "p"], max_new_tokens=4)
    plain_rng = torch.random.get_rng_state()
    torch.manual_seed(77)
    seen = []
    streamed = engine.generate(["p", "p"], max_new_tokens=4,
                               generation_callback=lambda *args, **kwargs: seen.append(args))
    assert streamed == plain and torch.equal(plain_rng, torch.random.get_rng_state())
    assert {args[0] for args in seen} == {0, 1}


def test_vllm_bridge_streaming_keeps_final_tokens_and_logprobs():
    seen = []
    outputs = [SimpleNamespace(request_id=str(i), finished=True, prompt_token_ids=[1], outputs=[SimpleNamespace(
        text=f"answer {i}", token_ids=[2], logprobs=[{2: SimpleNamespace(logprob=-0.5)}], finish_reason="stop")])
        for i in range(2)]
    class Engine:
        def __init__(self):
            self.count = 0
        def add_request(self, request_id, prompt):
            pass
        def step(self):
            self.count += 1
            if self.count % 2:
                return [SimpleNamespace(request_id=str(i), finished=False, outputs=[SimpleNamespace(text="partial")]) for i in range(2)]
            return outputs
    class LLM:
        llm_engine = Engine()
        def generate(self, prompts, sampling, **kwargs):
            for i, prompt in enumerate(prompts):
                self.llm_engine.add_request(str(i), prompt)
            self.llm_engine.step()
            return self.llm_engine.step()
    bridge = GRPOBridge(GRPOBridgeConfig(concurrency_g=2, device="cpu", log_model_outputs=False))
    bridge.trainer_model, bridge.tokenizer, bridge.vllm_engine = TinyPolicy(), SimpleNamespace(), LLM()
    with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw))}):
        plain = bridge.generate_rollouts(["p"])
        bridge.generation_callback = lambda i, text, **kw: seen.append((i, text))
        monitored = bridge.generate_rollouts(["p"])
    assert any(text == "partial" for _, text in seen)
    assert [(item.response_token_ids, item.rollout_logprobs) for item in monitored.items] == [
        (item.response_token_ids, item.rollout_logprobs) for item in plain.items]


@pytest.mark.parametrize("enabled", [False, True])
def test_long_training_executes_tests_and_logger_preserves_policy_update(tmp_path, enabled):
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1]])}
        def encode(self, text, **kwargs):
            return text.split()
    logger = TrainingLogger()
    logger._launch = lambda: None
    policy = TinyPolicy()
    before = policy.logits.detach().clone()
    def generate(bridge, prompts):
        result = batch()
        for index, item in enumerate(result.items):
            item.response_text = f"```python\ndef add(a, b):\n    return a {'+' if index == 0 else '-'} b\n```"
            if bridge.generation_callback is not None:
                bridge.generation_callback(index, item.response_text, finished=True)
        return result
    with patch.object(GRPOBridge, "initialize_vllm"), \
         patch.object(GRPOBridge, "generate_rollouts", autospec=True, side_effect=generate), \
         patch("sword.training_logger.create_training_logger", side_effect=lambda **kw: logger.start()) as create:
        result = run_long_train(dataset=[{"problem": "Implement add", "category": "new_task", "tests": "assert add(2, 3) == 5"}],
                                model="mock", b_model=False, gradio_logger=enabled,
                                trainer_model=policy, tokenizer=Tokenizer(), is_save_and_upload_checkpoint=False,
                                trace_rollouts=True,
                                show_detail_logs=False, return_step_metrics=True,
                                Argument={"steps": 1, "max_seq_len": 256, "max_new_tokens": 1,
                                          "concurrency_g": 2, "output_dir": str(tmp_path)})
    assert result["status"] == "completed" and result["optimizer_steps"] == 1
    assert not torch.equal(policy.logits.detach().cpu(), before.cpu())
    records = [json.loads(line) for line in (tmp_path / "rollout_details.jsonl").read_text().splitlines()]
    assert [r["execution_evidence"]["tests"]["passed"] for r in records] == [1, 0]
    assert [r["scores"]["truth"] for r in records] == [1., 0.]
    assert [r["test_results"][0]["success"] for r in records] == [True, False]
    assert all(r["execution_results"] for r in records)
    assert not list((tmp_path / "harness_workspaces").iterdir())
    assert create.call_count == int(enabled)
    if enabled:
        logger._worker.join(2)
        state = logger.snapshot()
        assert state["status"] == "Completed" and len(state["scores"]) == 2
        assert state["history"][0]["mean_score"] == pytest.approx(sum(r["scores"]["total_score"] for r in records) / 2, abs=1e-4)
        assert state["scores"][0]["execution_evidence"]["tests"]["passed"] == 1


def test_gradio_dashboard_is_read_only_public_and_refreshes_judge_details():
    components = []
    callbacks = []
    launch_options = {}
    class Component:
        def __init__(self, *args, **kwargs):
            components.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def tick(self, function, **kwargs):
            callbacks.append(function)
        load = change = tick
        def launch(self, **kwargs):
            launch_options.update(kwargs)
            return None, "http://localhost:7860", "https://example.gradio.live"
    fake = SimpleNamespace(**{name: Component for name in (
        "Blocks", "Markdown", "Textbox", "JSON", "Tab", "Accordion", "Dataframe", "LinePlot", "Dropdown", "Timer")})
    logger = TrainingLogger(max_rollouts=2)
    logger._apply("task", {"step": 1, "problem_id": "p", "problem": "Implement add", "rollout_count": 8})
    logger._apply("score", {"rollout": 0, "scores": {"total_score": 0.8}, "judge_scores": {"truth": 1.0},
                            "judge_response": '{"truth":1.0}', "execution_evidence": {"tests": {"passed": 2}}})
    with patch.dict(sys.modules, {"gradio": fake}):
        logger._launch()
    assert logger.error is None
    assert launch_options["share"] is True and launch_options["prevent_thread_lock"] is True
    assert logger.share_url == "https://example.gradio.live"
    assert all(value["interactive"] is False for value in components if "interactive" in value)
    values = callbacks[0](0)
    assert values[-3] == {"truth": 1.0}
    assert values[-2] == '{"truth":1.0}' and values[-1]["tests"]["passed"] == 2


def test_missing_gradio_is_nonfatal():
    logger = TrainingLogger()
    with patch.dict(sys.modules, {"gradio": None}):
        logger._launch()
    assert logger.error and logger.app is None
