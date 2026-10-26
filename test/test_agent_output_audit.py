"""Reproduce warning emissions, inspect both models, and execute complete agent loops."""
import json
import logging
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch

from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.grpo_bridge import GRPOBridge, GRPOBridgeConfig
from sword.harness import OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness
from sword.logging_utils import quiet_quantization_cast_notices, output_preview
from test.test_reward_stall import TinyPolicy
from test.test_rollout_diagnostics import chat_tokenizer


HARNESSES = [OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness]
NOTICE = "MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization"


def test_actual_matmul8bit_forward_backward_notice_is_hidden(caplog, monkeypatch):
    from bitsandbytes.autograd._functions import MatMul8bitLt, MatmulLtState
    logger = logging.getLogger("bitsandbytes.autograd._functions")
    warning = logger.warning
    while hasattr(warning, "__wrapped__"):
        warning = warning.__wrapped__
    monkeypatch.setattr(logger, "warning", warning)
    monkeypatch.setattr(logger, "filters", [])

    def run():
        inputs = torch.randn(2, 16, requires_grad=True)
        result = MatMul8bitLt.apply(inputs, torch.randn(16, 16), None, None, MatmulLtState())
        result.sum().backward()
        assert torch.isfinite(result).all() and inputs.grad.abs().max() > 0

    with caplog.at_level(logging.WARNING):
        run()
    assert NOTICE in caplog.text  # proves this is the actual emitting backend
    caplog.clear()
    quiet_quantization_cast_notices()
    with caplog.at_level(logging.WARNING):
        run()
        logger.warning("Another bitsandbytes warning")
        logger.error(NOTICE)
    assert not any(record.levelno == logging.WARNING and NOTICE in record.getMessage() for record in caplog.records)
    assert "Another bitsandbytes warning" in caplog.text
    assert any(record.levelno == logging.ERROR and NOTICE in record.getMessage() for record in caplog.records)


def test_notice_stays_hidden_after_logger_filters_are_reset(caplog, monkeypatch):
    quiet_quantization_cast_notices()
    logger = logging.getLogger("bitsandbytes.autograd._functions")
    monkeypatch.setattr(logger, "filters", [])
    with caplog.at_level(logging.WARNING):
        logger.warning("MatMul8bitLt: inputs will be cast from %s to float16 during quantization", torch.float32)
        logger.warning("Still show other warnings")
    assert NOTICE not in caplog.text and "Still show other warnings" in caplog.text


def test_old_warning_backend_survives_later_simplefilter_changes(monkeypatch):
    received = []
    monkeypatch.setattr(warnings, "showwarning", lambda message, category, *args, **kwargs: received.append((str(message), category)))
    with warnings.catch_warnings():
        quiet_quantization_cast_notices()
        warnings.simplefilter("always")
        warnings.warn(NOTICE, UserWarning)
        warnings.warn(NOTICE + ".", UserWarning)
        warnings.warn("An unrelated warning", UserWarning)
        warnings.warn(NOTICE, RuntimeWarning)
    assert received == [("An unrelated warning", UserWarning), (NOTICE, RuntimeWarning)]


def test_model_a_output_is_printed_without_waiting_for_reward_stall(capsys):
    bridge = GRPOBridge(GRPOBridgeConfig(model_name="mock", concurrency_g=2, device="cpu",
                                        max_model_len=16, max_new_tokens=1, top_p=0.01))
    bridge.initialize_trainer(model=TinyPolicy(), tokenizer=chat_tokenizer())
    try:
        batch = bridge.generate_rollouts(["p"])
        output = capsys.readouterr().out
        assert output.count("[ModelA Output]") == 2
        assert "backend=pytorch" in output and "response_tokens=1" in output
        assert all(item.response_text in output for item in batch.items)
    finally:
        bridge.weight_sync.cleanup()


def test_bounded_output_preserves_answer_tail_and_marks_empty():
    assert output_preview("") == "<empty>"
    text = "thinking " * 100 + "Final Answer: 42"
    preview = output_preview(text, 80)
    assert preview.startswith("thinking") and preview.endswith("Final Answer: 42")
    assert "characters omitted" in preview


def test_real_http_judge_outputs_and_thinking_only_retry_are_visible(capsys):
    requests_seen = []
    scores = {"truth":1, "formatted":1, "rule_reasoning_follow":1, "rule_answer_follow":1,
              "tools_success":0, "cheating":False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests_seen.append(payload)
            message = {"content":"", "thinking":"Checking the candidate implementation"} if len(requests_seen) == 1 else {"content":json.dumps(scores)}
            body = json.dumps({"message":message, "eval_count":24, "done_reason":"length" if len(requests_seen) == 1 else "stop"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        judge = ModelBJudge(ModelBJudgeConfig(port=server.server_port, auto_setup=False, fallback_model_name=None))
        judge.ensure_ready = lambda: True
        result = judge.evaluate("p", "42", "", "", "Final Answer: 42")
        assert result.truth == 1
        assert [item["think"] for item in requests_seen] == [True, False]
        output = capsys.readouterr().out
        assert output.count("[ModelB Request]") == output.count("[ModelB Output]") == 2
        assert "Checking the candidate implementation" in output
        assert "<empty>" in output and json.dumps(scores) in output
        assert "generated_tokens=24" in output and "[ModelB Scores]" in output
        assert [item["status"] for item in judge.last_evaluation_attempts] == ["failed", "scored"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("cls", HARNESSES)
def test_agent_retries_reasoning_then_writes_edits_verifies_and_finishes(tmp_path, cls, capsys):
    harness = cls(workspace_dir=str(tmp_path))
    harness.log_tool_outputs = True
    if cls is CodexCliHarness:
        create = json.dumps({"name":"apply_patch", "arguments":"*** Begin Patch\n*** Add File: state.txt\n+draft\n*** End Patch"})
        edit = json.dumps({"name":"apply_patch", "arguments":{"patch":"*** Begin Patch\n*** Update File: state.txt\n@@\n-draft\n+verified\n*** End Patch"}})
    else:
        create = json.dumps({"function":{"name":"write_file", "arguments":json.dumps({"path":"state.txt", "content":"draft"})}})
        edit = json.dumps({"name":"edit", "arguments":{"path":"state.txt", "old_str":"draft", "new_str":"verified"}})
    command = f'"{sys.executable}" -c "from pathlib import Path; assert Path(\'state.txt\').read_text() == \'verified\'; print(\'VERIFIED_FROM_TOOL\')"'
    verify = json.dumps({"function":{"name":"exec_command" if cls is CodexCliHarness else "bash", "arguments":json.dumps({"cmd":command})}})
    responses = ["<think>I need to plan before acting.</think>", "<tool_call>{broken request", create, edit, verify, "Final Answer: Created, edited, and verified."]
    prompts = []
    def generate(prompt):
        index = len(prompts)
        prompts.append(prompt)
        if index in (1, 2):
            assert "No complete tool call" in prompt
        if index == 3:
            assert (tmp_path / "state.txt").read_text() == "draft"
        if index == 5:
            assert "VERIFIED_FROM_TOOL" in prompt
        return responses[index]
    result = harness.run("Implement and verify", generate, max_turns=8)
    assert result.is_completed and result.turns == 6
    assert result.tools_called == result.tools_succeeded == 3
    assert (tmp_path / "state.txt").read_text() == "verified"
    assert "[Agent Tool]" in capsys.readouterr().out


@pytest.mark.parametrize("cls", HARNESSES)
def test_plan_or_partial_tool_call_is_never_a_completed_answer(tmp_path, cls):
    harness = cls(workspace_dir=str(tmp_path))
    for text in ("I will inspect the files first.", "<think>Final Answer: unfinished", '<tool_call>{"name":"write_file","arguments":'):
        result = harness.run("Work", lambda prompt: text, max_turns=2)
        assert not result.is_completed and result.turns == 2
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cls", [OpenCodeHarness, GeminiCliHarness, PiHarness])
def test_tool_calls_inside_thinking_are_not_executed(tmp_path, cls):
    hidden = json.dumps({"name":"write_file", "arguments":{"path":"should_not_exist.txt", "content":"hidden"}})
    responses = iter([f"<think>{hidden}</think>", "Final Answer: Done."])
    result = cls(workspace_dir=str(tmp_path)).run("Work", lambda prompt: next(responses))
    assert result.is_completed and result.turns == 2 and result.tools_called == 0
    assert not (tmp_path / "should_not_exist.txt").exists()


def test_gemini_calls_with_braces_in_content_and_pi_last_line_action(tmp_path):
    content = '{"nested":{"value":1}}'
    calls = iter(["call:write_file" + json.dumps({"file_path":"nested.json", "content":content}), "Final Answer: Written."])
    result = GeminiCliHarness(workspace_dir=str(tmp_path)).run("Write JSON", lambda prompt: next(calls))
    assert result.is_completed and result.tools_succeeded == 1
    assert (tmp_path / "nested.json").read_text() == content
    actions = iter(["Action: read nested.json", "Final Answer: Read."])
    result = PiHarness(workspace_dir=str(tmp_path)).run("Read JSON", lambda prompt: next(actions))
    assert result.is_completed and result.tools_succeeded == 1

