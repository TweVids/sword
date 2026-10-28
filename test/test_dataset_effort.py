"""Dataset effort reaches both generation backends and the judge's full evidence."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.grpo_bridge import (
    GRPOBridge, GRPOBridgeConfig, RolloutBatch, RolloutItem,
    THINKING_EFFORTS, format_rollout_prompt, rollout_prompt_budget, tokenize_rollout_prompt,
)
from sword.harness import OpenCodeHarness
from sword.long_trainer import DatasetProblemRow, load_dataset_records, parse_dataset_row, run_long_train
from test.test_judge_rule_context import native_thinking_tokenizer
from test.test_reward_stall import TinyPolicy


def effort_tokenizer():
    tokenizer = native_thinking_tokenizer(False)
    tokenizer.chat_template = (Path(__file__).parents[1] / "chat_template.jinja").read_text(encoding="utf-8")
    return tokenizer


@pytest.mark.parametrize("value,expected", [
    (None, "dynamic"), ("", "dynamic"), ("  ", "dynamic"), (" DYNAMIC ", "dynamic"),
    ("low", "low"), (" Medium ", "medium"), ("HIGH", "high"),
])
def test_effort_is_normalized_and_serialized(value, expected):
    row = parse_dataset_row({"problem": "p", "solution": "4", "effort": value})
    assert row.effort == row.to_dict()["effort"] == expected
    assert DatasetProblemRow("p", "4", effort=value).effort == expected


@pytest.mark.parametrize("value", ["off", "xhigh", "max", False, 0, [], {}])
def test_invalid_dataset_effort_is_rejected(value):
    with pytest.raises(ValueError, match="dynamic, low, medium, high"):
        parse_dataset_row({"problem": "p", "effort": value})


def test_effort_affects_row_and_dataset_hashes():
    rows, _ = load_dataset_records([{"problem": "p", "solution": "4", "effort": e} for e in THINKING_EFFORTS])
    assert len({row.row_hash for row in rows}) == 4
    hashes = [load_dataset_records([row.to_dict()])[1] for row in rows]
    assert len(set(hashes)) == 4
    assert parse_dataset_row({"problem": "p", "solution": "4"}).row_hash == rows[0].row_hash


@pytest.mark.parametrize("extension", ["csv", "tsv", "json", "jsonl"])
def test_local_dataset_columns_include_effort_and_quoted_rules(tmp_path, extension):
    records = [{"problem": "p, with a comma", "solution": "4", "effort": e,
                "reasoning_rule": "Decide whether to think.\nKeep it short.", "answer_rule": "A number."}
               for e in THINKING_EFFORTS]
    path = tmp_path / ("data." + extension)
    with path.open("w", encoding="utf-8", newline="") as stream:
        if extension in ("csv", "tsv"):
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter="\t" if extension == "tsv" else ",")
            writer.writeheader()
            writer.writerows(records)
        elif extension == "json":
            json.dump(records, stream)
        else:
            stream.write("\n".join(json.dumps(row) for row in records))
    rows, dataset_hash = load_dataset_records(str(path))
    assert [row.effort for row in rows] == list(THINKING_EFFORTS)
    assert all(row.problem == records[0]["problem"] and row.reasoning_rule == records[0]["reasoning_rule"] for row in rows)
    assert dataset_hash == load_dataset_records(records)[1]


@pytest.mark.parametrize("effort", THINKING_EFFORTS)
def test_supplied_template_enables_thinking_without_forcing_a_thought_prefix(effort):
    tokenizer = effort_tokenizer()
    rendered, is_chat = format_rollout_prompt(tokenizer, "p", effort=effort)
    assert is_chat and "<|turn>system\n<|think|>\nEffort: " + effort in rendered
    assert rendered.endswith("<|turn>model\n")
    assert "<|channel>thought" not in rendered  # dynamic can emit either thoughts or a direct answer
    disabled, _ = format_rollout_prompt(tokenizer, "p", enable_thinking=False, effort=effort)
    assert "<|think|>" not in disabled and "Effort:" not in disabled
    inputs, _ = tokenize_rollout_prompt(tokenizer, "", effort=effort)
    assert rollout_prompt_budget(tokenizer, 1024, 64, effort=effort) == 1024 - 64 - len(inputs["input_ids"][0]) - 8


@pytest.mark.parametrize("effort", THINKING_EFFORTS)
@pytest.mark.parametrize("backend", ["pytorch", "vllm", "vllm_failure"])
def test_generation_backends_receive_identical_effort_prompts(effort, backend):
    tokenizer = effort_tokenizer()
    expected_ids = tokenize_rollout_prompt(tokenizer, "p", effort=effort)[0]["input_ids"][0].tolist()
    bridge = GRPOBridge(GRPOBridgeConfig(concurrency_g=2, device="cpu", max_model_len=128,
                                        max_new_tokens=1, top_p=0.01, effort=effort))
    bridge.initialize_trainer(model=TinyPolicy(), tokenizer=tokenizer)
    engine = MagicMock()

    def generate(prompts, sampling, **kwargs):
        assert prompts == [{"prompt_token_ids": expected_ids}] * 2
        assert sampling["skip_special_tokens"] is False
        if backend == "vllm_failure":
            raise RuntimeError("switch to PyTorch")
        return [SimpleNamespace(prompt_token_ids=expected_ids, outputs=[SimpleNamespace(
            text="p", token_ids=[2], logprobs=[{2: SimpleNamespace(logprob=-0.5)}], finish_reason="stop",
        )])] * 2

    if backend != "pytorch":
        engine.generate.side_effect = generate
        bridge.vllm_engine = engine
    try:
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=lambda **kw: kw)}):
            batch = bridge.generate_rollouts(["p"])
        assert [item.prompt_token_ids for item in batch.items] == [expected_ids] * 2
    finally:
        bridge.weight_sync.cleanup()


@pytest.mark.parametrize("candidate,answer", [
    ("<think>2 + 2 = 4</think>4", "4"),
    ("<|channel>thought\n2 + 2 = 4\n<channel|>4", "4"),
    ("<think>2 + 2 = 4", ""),
    ("<|channel>thought\n2 + 2 = 4", ""),
    ("4", "4"),
])
def test_judge_sees_original_thoughts_and_separate_answer(candidate, answer):
    judge = ModelBJudge()
    prompt = judge.build_user_prompt("p", "4", "Choose whether to think.", "A number.", candidate)
    assert "Candidate Thinking Effort: dynamic" in prompt
    assert 'Candidate Model Response to Evaluate:\n"""\n' + candidate + '\n"""' in prompt
    assert 'Candidate Answer Outside Thought Channels:\n"""\n' + answer + '\n"""' in prompt
    assert "candidate decides whether to emit thoughts" in judge.SYSTEM_PROMPT
    assert "a correct result appearing only in thoughts is not a final answer" in judge.SYSTEM_PROMPT


@pytest.mark.parametrize("completed", [False, True])
def test_row_effort_persists_through_harness_followups_and_judge_sees_every_turn(tmp_path, completed):
    tokenizer = effort_tokenizer()
    harness = OpenCodeHarness()
    harness.get_system_prompt = lambda: "Use tools and give a final answer."
    judge = ModelBJudge(ModelBJudgeConfig(auto_setup=False, auto_unload=False, reasoning=False))
    judge.ensure_ready = lambda **kw: True
    judge.prepare_for_eval = lambda: True
    queries, requests = [], []
    action = '<think>THOUGHT_BEFORE_TOOL</think>' + json.dumps({"name": "python", "arguments": {"code": "print(4)"}})
    last = "<|channel>thought\nLAST_TURN_THOUGHT\n<channel|>\n" + ("Final Answer: 4" if completed else action)
    model = TinyPolicy()
    model.logits = torch.nn.Parameter(torch.zeros(len(tokenizer)))

    def generate(bridge, prompts):
        initial = bridge.config.concurrency_g == 2
        requests.append((bridge.config.effort, initial))
        prompt_ids = tokenize_rollout_prompt(tokenizer, prompts[0], effort=bridge.config.effort)[0]["input_ids"][0].tolist()
        return RolloutBatch(items=[RolloutItem(prompts[0], prompt_ids, action if initial else last, [2], [-0.5])
                                  for _ in range(bridge.config.concurrency_g)], concurrency_g=bridge.config.concurrency_g)

    def request(payload):
        queries.append(payload["messages"][1]["content"])
        scores = {"truth": float(completed), "formatted": 1, "rule_reasoning_follow": 1,
                  "rule_answer_follow": float(completed), "tools_success": 1, "cheating": False}
        return {"message": {"content": json.dumps(scores)}}

    with patch("torch.cuda.is_available", return_value=False), \
         patch("sword.long_trainer.ModelBJudge", return_value=judge), \
         patch("sword.long_trainer.get_harness", return_value=harness), \
         patch.object(GRPOBridge, "initialize_vllm", return_value=False), \
         patch.object(GRPOBridge, "generate_rollouts", autospec=True, side_effect=generate), \
         patch.object(judge, "_request_chat", side_effect=request):
        run_long_train(
            dataset=[{"problem": "p", "solution": "4", "harness": "opencode", "effort": e,
                      "reasoning_rule": "Choose whether to think.", "answer_rule": "A number."}
                     for e in THINKING_EFFORTS],
            model="controlled", trainer_model=model, tokenizer=tokenizer,
            b_model=True, B_model_auto_setup=False, auto_unload_B_model=False,
            is_save_and_upload_checkpoint=False, show_detail_logs=False, trace_rollouts=True,
            Argument={"steps": 4, "concurrency_g": 2, "max_seq_len": 1024, "max_new_tokens": 1,
                      "harness_max_turns": 2, "output_dir": str(tmp_path)},
        )
    assert requests == [(e, initial) for e in THINKING_EFFORTS for initial in (True, False, False)]
    assert len(queries) == 8
    for effort, query in zip([e for e in THINKING_EFFORTS for _ in range(2)], queries):
        assert "Candidate Thinking Effort: " + effort in query
        assert "THOUGHT_BEFORE_TOOL" in query and "LAST_TURN_THOUGHT" in query
        assert "Choose whether to think." in query and "A number." in query
    records = [json.loads(line) for line in (tmp_path / "rollout_details.jsonl").read_text().splitlines()]
    assert [row["effort"] for row in records] == [e for e in THINKING_EFFORTS for _ in range(2)]
    assert all(row["candidate_response"] == (last if completed else "") for row in records)
