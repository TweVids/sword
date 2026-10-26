"""Agent tool loops, workspace lifetime, targeted logging, and signed RL loss."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import pytest
import torch

from sword.harness import OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness, WorkspaceSandbox
from sword.harness.base import BaseHarness
from sword.logging_utils import quiet_quantization_cast_notices
from sword.grpo_bridge import DualCorrectionGRPOLoss, GRPOBridge, RolloutItem
from sword.long_trainer import run_long_train
from test.test_reward_stall import TinyPolicy, batch


HARNESSES = [OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness]


def tool_turns(cls):
    if cls is CodexCliHarness:
        return [
            "*** Begin Patch\n*** Add File: state.txt\n+draft\n*** End Patch",
            "*** Begin Patch\n*** Update File: state.txt\n@@\n-draft\n+verified\n*** End Patch",
            "```python\nfrom pathlib import Path\nassert Path('state.txt').read_text() == 'verified'\n```",
        ]
    if cls is GeminiCliHarness:
        return [
            'call:write_file{"file_path":"state.txt","content":"draft"}',
            'call:replace{"file_path":"state.txt","old_string":"draft","new_string":"verified"}',
            'call:read_file{"file_path":"state.txt"}',
        ]
    edit_args = {"path": "state.txt", "oldText": "draft", "newText": "verified"} if cls is PiHarness else {
        "path": "state.txt", "oldString": "draft", "newString": "verified"}
    return ["<tool_call>" + json.dumps(call) + "</tool_call>" for call in [
        {"action": "write", "parameters": {"path": "state.txt", "content": "draft"}},
        {"action": "edit", "parameters": edit_args},
        {"action": "read", "parameters": {"path": "state.txt"}},
    ]]


@pytest.mark.parametrize("cls", HARNESSES)
def test_tools_continue_until_final_answer_and_project_survives_each_turn(tmp_path, cls):
    harness = cls()
    harness.workspace_parent_dir = str(tmp_path)
    roots = []
    for _ in range(2):
        responses = tool_turns(cls) + ["Final Answer: The file is verified."]
        prompts = []
        def generate(prompt):
            root, = tmp_path.glob("sword_harness_workspace_*")
            index = len(prompts)
            prompts.append(prompt)
            if index == 0:
                assert not (root / "state.txt").exists()
                roots.append(root)
            else:
                expected = "draft" if index == 1 else "verified"
                assert (root / "state.txt").read_text() == expected
                assert "draft" in prompt if index == 1 else "verified" in prompt
            return responses[index]
        result = harness.run("Create, edit, and verify state.txt", generate)
        assert len(prompts) == result.turns == 4
        assert result.tools_called == result.tools_succeeded == 3
        assert result.is_completed and result.stop_reason == "final_response"
        assert result.error_message is None
        assert result.workspace_removed and not Path(result.workspace_dir).exists()
        assert len(result.assistant_turns) == 4
    assert roots[0] != roots[1]


@pytest.mark.parametrize("cls", HARNESSES)
def test_tool_only_turn_limit_is_reported_as_incomplete(tmp_path, cls):
    harness = cls()
    harness.workspace_parent_dir = str(tmp_path)
    result = harness.run("Keep working", lambda prompt: tool_turns(cls)[0], max_turns=2)
    assert result.turns == 2 and result.tools_called == 2
    assert not result.is_completed
    assert result.stop_reason == "turn_limit"
    assert "turn_limit" in result.error_message
    assert result.workspace_removed


@pytest.mark.parametrize("cls", [OpenCodeHarness, CodexCliHarness, PiHarness])
def test_explicit_final_code_answer_completes_after_validation(tmp_path, cls):
    harness = cls()
    harness.workspace_parent_dir = str(tmp_path)
    response = "Final Answer:\n```python\nprint(42)\n```"
    result = harness.run("Compute the answer", lambda prompt: response)
    assert result.is_completed and result.turns == 1
    assert result.tools_called == result.tools_succeeded == 1


def test_final_label_inside_thinking_is_not_a_terminal_answer(tmp_path):
    harness = OpenCodeHarness()
    harness.workspace_parent_dir = str(tmp_path)
    responses = iter(["<think>\nFinal Answer: still thinking\n</think>\n```python\nprint(42)\n```",
                      "Final Answer: 42"])
    result = harness.run("Calculate", lambda prompt: next(responses))
    assert result.is_completed and result.turns == 2


def test_kept_workspaces_are_separate_and_explicit_project_is_not_wiped(tmp_path):
    harness = OpenCodeHarness()
    harness.workspace_parent_dir = str(tmp_path / "runs")
    harness.keep_workspace = True
    first = harness.run("first", lambda prompt: "Final Answer: first")
    (Path(first.workspace_dir) / "previous.txt").write_text("keep me")
    second = harness.run("second", lambda prompt: "Final Answer: second")
    assert first.workspace_dir != second.workspace_dir
    assert not first.workspace_removed and not second.workspace_removed
    assert (Path(first.workspace_dir) / "previous.txt").read_text() == "keep me"
    assert not (Path(second.workspace_dir) / "previous.txt").exists()
    project = tmp_path / "project"
    project.mkdir()
    (project / "existing.txt").write_text("preserved")
    supplied = OpenCodeHarness(workspace_dir=str(project)).run("Use project", lambda prompt: "Done.")
    assert not supplied.workspace_removed
    assert (project / "existing.txt").read_text() == "preserved"


def test_python_runner_does_not_overwrite_a_project_file(tmp_path):
    sandbox = WorkspaceSandbox(base_dir=str(tmp_path))
    original = tmp_path / "_agent_runner.py"
    original.write_text("project source")
    ok, output = sandbox.run_python("print(42)")
    assert ok and output == "42"
    assert original.read_text() == "project source"
    assert not list(tmp_path.glob(".sword_agent_*.py"))


def test_workspace_cleanup_rejects_a_changed_target(tmp_path):
    sandbox = WorkspaceSandbox(parent_dir=str(tmp_path))
    original = sandbox.root_dir
    sandbox.root_dir = str(tmp_path)
    with pytest.raises(PermissionError, match="owned temporary directory"):
        sandbox.close()
    assert Path(original).exists()
    sandbox.root_dir = original
    sandbox.close()


def test_followup_preserves_task_and_recent_observation_within_budget():
    class DenseTokenizer:
        def encode(self, text, **kwargs):
            return list(text)
    harness = BaseHarness()
    harness.get_system_prompt = lambda: "Guidance " * 100
    history = "\n\nModel:\n" + "old history " * 100 + "\n\nModel:\nRead state.txt\nObservation: verified"
    prompt = harness.format_followup_prompt("Keep the original task.", history, 220, DenseTokenizer())
    assert len(prompt) <= 220
    assert "Keep the original task." in prompt and "Observation: verified" in prompt
    assert "old history" not in prompt


def test_quantization_notice_filter_is_targeted_and_idempotent(caplog):
    quiet_quantization_cast_notices()
    quiet_quantization_cast_notices()
    logger = logging.getLogger("bitsandbytes.autograd._functions")
    assert sum(getattr(item, "_sword_quantization_cast_filter", False) for item in logger.filters) == 1
    with caplog.at_level(logging.WARNING):
        logger.warning("MatMul8bitLt: inputs will be cast from %s to float16 during quantization", torch.float32)
        logger.warning("Another quantization warning")
        logger.error("A real quantization failure")
    assert "inputs will be cast" not in caplog.text
    assert "Another quantization warning" in caplog.text and "A real quantization failure" in caplog.text
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet_quantization_cast_notices()
        warnings.warn("MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization", UserWarning)
        warnings.warn("Other warning", UserWarning)
    assert [str(item.message) for item in caught] == ["Other warning"]


def test_negative_policy_loss_has_a_valid_reward_increasing_gradient():
    model = TinyPolicy()
    before = model.logits.softmax(-1)[2].item()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, metrics = DualCorrectionGRPOLoss(entropy_coeff=0.0)(
        model, torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.long), [1], torch.ones(1),
    )
    assert loss.item() == pytest.approx(-1.0)
    assert metrics["policy_loss"] < 0
    loss.backward()
    assert model.logits.grad[2] < 0
    optimizer.step()
    assert model.logits.softmax(-1)[2].item() > before


def test_long_trainer_honors_turn_budget_and_does_not_score_a_tool_call_as_final(tmp_path):
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1]])}
        def encode(self, text, **kwargs):
            return text.split()
    harness = OpenCodeHarness()
    harness.get_system_prompt = lambda: "Use tools, inspect their results, and finish the task."
    action = tool_turns(OpenCodeHarness)[0]
    def generate(prompts):
        result = batch()
        for item in result.items:
            item.response_text = action
        return result
    with patch("sword.long_trainer.get_harness", return_value=harness), \
         patch.object(GRPOBridge, "initialize_vllm"), \
         patch.object(GRPOBridge, "generate_rollouts", side_effect=generate), \
         patch.object(GRPOBridge, "generate_single", side_effect=lambda prompt, **kwargs:
                      RolloutItem(prompt, [1], action, [2], [-0.5])):
        result = run_long_train(
            dataset=[{"problem": "p", "solution": "4", "harness": "opencode"}], model="mock", b_model=False,
            trainer_model=TinyPolicy(), tokenizer=Tokenizer(), trace_rollouts=True, return_step_metrics=True,
            is_save_and_upload_checkpoint=False, show_detail_logs=False,
            Argument={"steps": 1, "max_seq_len": 256, "max_new_tokens": 1, "concurrency_g": 2,
                      "harness_max_turns": 2, "output_dir": str(tmp_path)},
        )
    records = [json.loads(line) for line in (tmp_path / "rollout_details.jsonl").read_text().splitlines()]
    assert all(record["harness"]["turns"] == 2 and not record["harness"]["is_completed"] for record in records)
    assert all(record["candidate_response"] == "" for record in records)
    assert all(record["first_response"] == action for record in records)
    assert result["step_metrics"][0]["incomplete_harness_rollouts"] == 2
    assert not list((tmp_path / "harness_workspaces").iterdir())
