"""An actual sampled policy/tool/HTTP-judge/backward loop with controlled outputs."""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.grpo_bridge import GRPOBridge
from sword.harness import OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness
from sword.long_trainer import run_long_train


class CharacterTokenizer:
    pad_token_id = 130
    eos_token_id = 129

    def encode(self, text, add_special_tokens=True):
        ids = [min(ord(char), 127) for char in text]
        return ids + [128] if add_special_tokens else ids

    def __call__(self, text, **options):
        ids = self.encode(text)
        if options.get("truncation"):
            ids = ids[-options.get("max_length", len(ids)):]
        return {"input_ids":torch.tensor([ids])}

    def decode(self, ids, **options):
        return "".join(chr(int(value)) for value in ids if int(value) < 128)


class ControlledAgentPolicy(torch.nn.Module):
    """A tiny causal policy follows observations; its final digit is sampled.

    No generation function is mocked. Raw logprobs and backward use this same
    model, so the test verifies training dataflow, not real coding capability.
    """
    def __init__(self, cls):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(131))
        if cls is CodexCliHarness:
            create = json.dumps({"name":"apply_patch", "arguments":"*** Begin Patch\n*** Add File: state.txt\n+draft\n*** End Patch"})
            edit = json.dumps({"name":"apply_patch", "arguments":"*** Begin Patch\n*** Update File: state.txt\n@@\n-draft\n+verified\n*** End Patch"})
            read_command = f'"{sys.executable}" -c "from pathlib import Path; print(Path(\'state.txt\').read_text())"'
            verify_command = f'"{sys.executable}" -c "from pathlib import Path; assert Path(\'state.txt\').read_text() == \'verified\'; print(\'CHECKED\')"'
            read = json.dumps({"name":"exec_command", "arguments":{"cmd":read_command}})
            verify = json.dumps({"name":"exec_command", "arguments":{"cmd":verify_command}})
        else:
            create = json.dumps({"name":"write_file", "arguments":{"path":"state.txt", "content":"draft"}})
            edit = json.dumps({"name":"edit", "arguments":{"path":"state.txt", "old_str":"draft", "new_str":"verified"}})
            read = json.dumps({"name":"read_file", "arguments":{"path":"state.txt"}})
            verify = json.dumps({"name":"python", "arguments":{"code":"from pathlib import Path\nassert Path('state.txt').read_text() == 'verified'\nprint('CHECKED')"}})
        self.scripts = [create, edit, read, verify, "Final Answer: 0"]

    def forward(self, input_ids, **kwargs):
        rows = []
        for ids in input_ids.tolist():
            boundary = ids.index(128)
            prompt = "".join(chr(value) for value in ids[:boundary])
            matches = list(re.finditer(r"(?:OpenCode Environment|Gemini Environment Feedback|Codex CLI Execution Feedback):\n|(?:^|\n)Observation:\s*", prompt))
            feedback = prompt[matches[-1].end():] if matches else ""
            if "CHECKED" in feedback:
                stage = 4
            elif "edited" in feedback or "updated" in feedback:
                stage = 2
            elif "verified" in feedback:
                stage = 3
            elif feedback:
                stage = 1
            else:
                stage = 0
            script = self.scripts[stage]
            target_ids = [ord(char) for char in script] + [129]
            positions = torch.arange(len(ids), device=input_ids.device)
            relative = (positions - boundary).clamp(min=0, max=len(target_ids)-1)
            peaks = torch.tensor(target_ids, device=input_ids.device)[relative]
            logits = torch.zeros(len(ids), 131, device=input_ids.device)
            logits.scatter_(1, peaks[:, None], 24.)
            if stage == 4:
                final_position = boundary + len(script) - 1
                if final_position < len(ids):
                    logits[final_position, ord("1")] = 24.
            rows.append(logits + self.bias)
        return SimpleNamespace(logits=torch.stack(rows))


@pytest.mark.parametrize("cls", [OpenCodeHarness, CodexCliHarness, GeminiCliHarness, PiHarness])
def test_sampled_policy_executes_full_loop_and_all_turns_get_trained(tmp_path, cls, capsys):
    torch.manual_seed(17)
    judge_requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            judge_requests.append(data)
            query = data["messages"][-1]["content"]
            assert "VERIFY_AFTER_EDIT_RULE" in query and "FINAL_DIGIT_RULE" in query
            earlier = query.split("Earlier Assistant Turns (reasoning and action evidence):\n", 1)[1]
            earlier = json.loads(earlier.split("\n\nCandidate Model Response to Evaluate:", 1)[0])
            assert len(earlier) == 4
            assert "state.txt" in earlier[0] and "CHECKED" in earlier[-1]
            candidate = data["messages"][-1]["content"].split("Candidate Model Response to Evaluate:\n", 1)[1]
            truth = float("Final Answer: 1" in candidate)
            scores = {"truth":truth, "formatted":1, "rule_reasoning_follow":1, "rule_answer_follow":1,
                      "tools_success":1, "cheating":False}
            body = json.dumps({"message":{"content":json.dumps(scores)}, "eval_count":32, "done_reason":"stop"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    judge = ModelBJudge(ModelBJudgeConfig(model_name="controlled-http-judge", port=server.server_port,
                                          auto_setup=False, force_gpu=False, auto_unload=False, reasoning=False))
    judge.ensure_ready = lambda: True
    model = ControlledAgentPolicy(cls)
    harness = cls()
    harness.get_system_prompt = lambda: "Implement the task with tools."
    try:
        with patch("sword.long_trainer.ModelBJudge", return_value=judge), \
             patch("sword.long_trainer.get_harness", return_value=harness), \
             patch.object(GRPOBridge, "initialize_vllm", return_value=False):
            result = run_long_train(
                dataset=[{"problem":"Write draft, change it to verified, read it, verify it, then give a final digit.",
                          "reasoning_rule":"VERIFY_AFTER_EDIT_RULE: read and verify the edited file before answering.",
                          "answer_rule":"FINAL_DIGIT_RULE: give a final digit.",
                          "solution":"Final Answer: 1", "harness":harness.name, "can_be_verify":False}],
                model="controlled-agent-policy", trainer_model=model, tokenizer=CharacterTokenizer(),
                b_model=True, B_model_auto_setup=False, trace_rollouts=True, return_step_metrics=True,
                is_save_and_upload_checkpoint=False, auto_unload_B_model=False,
                Argument={"steps":1, "concurrency_g":4, "max_seq_len":4096, "max_new_tokens":256,
                          "micro_batch_size":1, "temperature":0.7, "top_p":1., "lr":0.05,
                          "output_dir":str(tmp_path)},
            )
        records = [json.loads(line) for line in (tmp_path / "rollout_details.jsonl").read_text().splitlines()]
        assert len(records) == len(judge_requests) == 4
        assert all(record["harness"]["is_completed"] and record["harness"]["turns"] == 5 for record in records)
        assert all(record["tools_called"] == record["tools_succeeded"] == 4 for record in records)
        assert all(record["training_turn_count"] == len(record["training_turns"]) == 5 for record in records)
        for record in records:
            assert all("VERIFY_AFTER_EDIT_RULE" in turn["prompt_text"] and "FINAL_DIGIT_RULE" in turn["prompt_text"]
                       for turn in record["training_turns"])
            assert all(len(turn["response_token_ids"]) == len(turn["rollout_logprobs"]) for turn in record["training_turns"])
            assert record["training_response_tokens"] == sum(len(turn["response_token_ids"]) for turn in record["training_turns"])
            assert record["judge_attempts"][0]["status"] == "scored"
        metrics = result["step_metrics"][0]
        assert metrics["training_turns"] == 20 and metrics["followup_training_turns"] == 16
        assert metrics["grad_norm"] > 0 and metrics["advantage_abs_max"] > 0
        assert model.bias[ord("1")] > model.bias[ord("0")]
        assert not any((tmp_path / "harness_workspaces").iterdir())
        output = capsys.readouterr().out
        assert output.count("[ModelA Output]") == 20 and output.count("[ModelB Output]") == 4
        assert output.count("[Agent Tool]") == 16
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
