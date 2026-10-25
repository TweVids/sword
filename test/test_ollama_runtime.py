"""Offline regressions for automatic provisioning and live-judge fallback."""

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from sword import ollama_runtime as runtime
from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.reward import MultiMetricRewardScorer


PRIMARY = "qwen3.6:35b"
FALLBACK = "qwen3.5:9b"
SCORES = {"truth": 1, "formatted": 1, "rule_reasoning_follow": 0.9,
          "rule_answer_follow": 1, "tools_success": 0, "cheating": False}
SCORE_CONTENT = json.dumps(SCORES)


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = "http://127.0.0.1:11343/api/chat"
    result._content = json.dumps(data).encode()
    return result


def tags(*models):
    return response({"models": [{"name": model} for model in models]})


def stream(*events):
    result = MagicMock()
    result.__enter__.return_value = result
    result.iter_lines.return_value = [json.dumps(event).encode() for event in events]
    return result


def test_existing_binary_is_reused_without_download(tmp_path):
    binary = tmp_path / "ollama"
    binary.touch()
    with patch.object(runtime.shutil, "which", return_value=str(binary)), patch.object(runtime.requests, "get") as get:
        assert runtime.ensure_ollama_binary() == str(binary)
        get.assert_not_called()


@pytest.mark.parametrize("machine,arch", [("x86_64", "amd64"), ("aarch64", "arm64")])
def test_installs_official_archive_in_user_prefix(tmp_path, machine, arch):
    prefix = tmp_path / "install"
    download = MagicMock()
    download.__enter__.return_value = download
    download.iter_content.return_value = [b"archive"]

    def extract(command, **kwargs):
        assert command[:3] == ["tar", "--zstd", "-xf"]
        assert command[-2:] == ["-C", str(prefix)]
        assert Path(command[3]).read_bytes() == b"archive"
        (prefix / "bin").mkdir()
        (prefix / "bin" / "ollama").touch()

    with patch.object(runtime, "_find_binary", side_effect=[None, str(prefix / "bin" / "ollama")]), \
         patch.object(runtime, "_install_dir", return_value=prefix), \
         patch.object(runtime.platform, "system", return_value="Linux"), \
         patch.object(runtime.platform, "machine", return_value=machine), \
         patch.object(runtime.shutil, "which", return_value="tool"), \
         patch.object(runtime.requests, "get", return_value=download) as get, \
         patch.object(runtime.subprocess, "run", side_effect=extract):
        assert runtime.ensure_ollama_binary() == str(prefix / "bin" / "ollama")
    assert get.call_args.args[0] == f"https://ollama.com/download/ollama-linux-{arch}.tar.zst"
    assert get.call_args.kwargs["timeout"] == (10, 120)


def test_remote_server_reused_without_local_binary():
    with patch.object(runtime, "_is_up", return_value=True), \
         patch.object(runtime, "ensure_ollama_binary") as install:
        assert runtime.start_ollama_server(host="https://judge.example:443")
        install.assert_not_called()


def test_unavailable_remote_server_does_not_install_locally():
    with patch.object(runtime, "_is_up", return_value=False), \
         patch.object(runtime, "ensure_ollama_binary") as install:
        assert not runtime.start_ollama_server(host="judge.example:11434")
        install.assert_not_called()


def test_starts_server_on_configured_port_and_storage(tmp_path):
    proc = MagicMock(pid=123)
    proc.poll.return_value = None
    with patch.object(runtime, "_is_up", side_effect=[False, True]), \
         patch.object(runtime, "ensure_ollama_binary", return_value="ollama"), \
         patch.object(runtime.subprocess, "Popen", return_value=proc) as popen:
        assert runtime.start_ollama_server(models_dir=str(tmp_path / "models"), host="127.0.0.1:11343",
                                           log_path=str(tmp_path / "ollama.log"))
        env = popen.call_args.kwargs["env"]
        assert env["OLLAMA_HOST"] == "127.0.0.1:11343"
        assert env["OLLAMA_MODELS"] == str(tmp_path / "models")
        assert env["OLLAMA_NUM_PARALLEL"] == "1"
        assert env["OLLAMA_MAX_LOADED_MODELS"] == "1"
        proc.terminate.assert_not_called()


def test_startup_timeout_terminates_only_spawned_process(tmp_path):
    proc = MagicMock()
    proc.poll.return_value = None
    with patch.object(runtime, "_is_up", return_value=False), \
         patch.object(runtime, "ensure_ollama_binary", return_value="ollama"), \
         patch.object(runtime.subprocess, "Popen", return_value=proc), \
         patch.object(runtime.time, "monotonic", side_effect=[0, 61]):
        assert not runtime.start_ollama_server(models_dir=str(tmp_path / "models"),
                                               log_path=str(tmp_path / "ollama.log"))
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)


def test_stream_pull_tracks_separate_layers_and_success(capsys):
    events = stream({"status": "pulling A", "digest": "A", "total": 100, "completed": 100},
                    {"status": "pulling B", "digest": "B", "total": 50, "completed": 0},
                    {"status": "pulling B", "digest": "B", "total": 50, "completed": 50},
                    {"status": "success"})
    with patch.object(runtime.requests, "post", return_value=events) as post:
        assert runtime.pull_ollama_model(PRIMARY, host="127.0.0.1:11343")
        assert post.call_args.args[0] == "http://127.0.0.1:11343/api/pull"
        assert post.call_args.kwargs["json"] == {"model": PRIMARY, "stream": True}
    assert "ready after" in capsys.readouterr().out


@pytest.mark.parametrize("events", [[], [{"error": "manifest not found"}], [{"status": "verifying"}]])
def test_pull_requires_explicit_success(events):
    with patch.object(runtime.requests, "post", return_value=stream(*events)):
        assert not runtime.pull_ollama_model(PRIMARY)


def test_pull_detects_byte_stall_even_when_events_keep_arriving(capsys):
    stalled = stream(*[{"status": "pulling A", "digest": "A", "total": 100, "completed": 5}] * 2)
    with patch.object(runtime.requests, "post", return_value=stalled), \
         patch.object(runtime.time, "monotonic", side_effect=[0, 1, 122]):
        assert not runtime.pull_ollama_model(PRIMARY, stall_s=120)
    assert "without new bytes" in capsys.readouterr().out


def test_pull_reports_read_timeout():
    with patch.object(runtime.requests, "post", side_effect=requests.exceptions.ReadTimeout("stalled")):
        assert not runtime.pull_ollama_model(PRIMARY)


def test_judge_auto_starts_and_pulls_on_its_own_port():
    judge = ModelBJudge(ModelBJudgeConfig(port=11343, models_dir="/marimo/.ollama"))
    with patch.object(judge, "is_available", return_value=False), \
         patch.object(judge, "list_models", return_value=[]), \
         patch("sword.b_model.start_ollama_server", return_value=True) as start, \
         patch("sword.b_model.pull_ollama_model", return_value=True) as pull:
        assert judge.ensure_ready()
        assert start.call_args.kwargs["host"] == "http://127.0.0.1:11343"
        assert pull.call_args.args == (PRIMARY,)
        assert pull.call_args.kwargs["models_dir"] == "/marimo/.ollama"


def test_missing_primary_tries_fallback_pull():
    judge = ModelBJudge()
    with patch.object(judge, "is_available", return_value=True), \
         patch.object(judge, "list_models", return_value=[]), \
         patch.object(judge, "unload"), \
         patch("sword.b_model.pull_ollama_model", side_effect=[False, True]) as pull:
        assert judge.ensure_ready()
        assert [call.args[0] for call in pull.call_args_list] == [PRIMARY, FALLBACK]
        assert judge.active_model_name == FALLBACK


@pytest.mark.parametrize("failure", [requests.exceptions.ReadTimeout("load timed out"),
                                   response({"error": "model requires more memory"}, 500),
                                   response({"message": {"content": "no scores"}})])
def test_reward_retries_smaller_judge_and_unloads_active_model(failure):
    judge = ModelBJudge()
    calls = []

    def post(url, json, **kwargs):
        calls.append((url, copy.deepcopy(json)))
        if url.endswith("/api/chat") and json["model"] == PRIMARY:
            # failure2 (unparseable content) must also fail for the no-think
            # re-request of the same model before escalation to the fallback.
            if isinstance(failure, Exception):
                raise failure
            return failure
        if url.endswith("/api/chat"):
            return response({"message": {"content": SCORE_CONTENT}})
        return response({"done": True})

    with patch("sword.b_model.requests.get", return_value=tags(PRIMARY, FALLBACK)), \
         patch("sword.b_model.requests.post", side_effect=post):
        score = MultiMetricRewardScorer(judge).score("2 + 2?", "4", "<think>2+2=4</think>4")
        assert score.rule_reasoning_follow == 0.9
        assert judge.active_model_name == FALLBACK
        assert judge.last_evaluation_source == FALLBACK
        # Keep using the successful fallback for subsequent trajectories.
        judge.evaluate("p", "4", "", "", "4")
        judge.release_after_eval()
    chat_models = [payload["model"] for url, payload in calls if url.endswith("/api/chat")]
    parse_failure = isinstance(failure, requests.Response) and \
        "error" not in failure.json()  # unparseable content, not an HTTP/API error
    if not parse_failure:
        # Transport/load failures escalate to the fallback immediately.
        assert chat_models == [PRIMARY, FALLBACK, FALLBACK]
    else:
        # Parse failure: the primary gets a think + no-think chance first.
        assert chat_models == [PRIMARY, PRIMARY, FALLBACK, FALLBACK]
    assert calls[-1][1] == {"model": FALLBACK, "keep_alive": 0}
    assert judge.evaluation_counts == {FALLBACK: 2}


def test_preload_failure_uses_fallback_with_same_inference_options():
    judge = ModelBJudge(ModelBJudgeConfig(num_ctx=2048))
    calls = []

    def post(url, json, **kwargs):
        calls.append(copy.deepcopy(json))
        if json.get("keep_alive") != 0 and json["model"] == PRIMARY:
            return response({"error": "OOM"}, 500)
        return response({"done": True})

    with patch("sword.b_model.requests.get", return_value=tags(PRIMARY, FALLBACK)), \
         patch("sword.b_model.requests.post", side_effect=post):
        assert judge.ensure_ready(load_model=True)
    assert [call["model"] for call in calls] == [PRIMARY, PRIMARY, FALLBACK]
    assert calls[-1]["options"]["num_ctx"] == 2048
    assert calls[-1]["options"]["num_gpu"] == 999


@pytest.mark.parametrize("reasoning", [True, False])
def test_unsupported_thinking_retries_primary_without_switching(reasoning):
    judge = ModelBJudge(ModelBJudgeConfig(reasoning=reasoning))
    seen = []

    def post(url, json, **kwargs):
        seen.append(copy.deepcopy(json))
        if "think" in json:
            return response({"error": "does not support thinking"}, 400)
        return response({"message": {"content": SCORE_CONTENT}})

    with patch("sword.b_model.requests.get", return_value=tags(PRIMARY)), \
         patch("sword.b_model.requests.post", side_effect=post):
        result = judge.evaluate("p", "4", "", "", "4")
    assert result.truth == 1
    assert [call["model"] for call in seen] == [PRIMARY, PRIMARY]
    assert seen[0]["think"] is reasoning
    assert "think" not in seen[-1]


@pytest.mark.parametrize("reasoning", [False, None])
def test_disabled_thinking_overrides_model_default(reasoning):
    judge = ModelBJudge(ModelBJudgeConfig(auto_setup=False, reasoning=reasoning, num_predict=128,
                                        num_predict_no_think=512))
    seen = []

    def post(url, json, **kwargs):
        seen.append(copy.deepcopy(json))
        if json.get("think", True):
            return response({"message": {"thinking": "Still evaluating...", "content": ""}})
        return response({"message": {"content": SCORE_CONTENT}})

    with patch("sword.b_model.requests.get", return_value=tags(PRIMARY)), \
         patch("sword.b_model.requests.post", side_effect=post):
        result = judge.evaluate("p", "4", "", "", "4")
    assert result.truth == 1
    assert judge.last_evaluation_source == PRIMARY
    assert len(seen) == 1
    assert seen[0]["think"] is False
    assert seen[0]["options"]["num_predict"] == 512


def test_both_judges_failing_reports_deterministic_and_limits_retries():
    judge = ModelBJudge()
    with patch("sword.b_model.requests.get", return_value=tags(PRIMARY, FALLBACK)), \
         patch("sword.b_model.requests.post", return_value=response({"error": "OOM"}, 500)) as post:
        judge.evaluate("p", "4", "", "", "14")
        calls = post.call_count
        result = judge.evaluate("p", "4", "", "", "14")
        assert post.call_count == calls
    assert result.truth < 1
    assert judge.last_evaluation_source == "deterministic"
    assert judge.evaluation_counts == {"deterministic": 2}


def test_mock_training_disables_automatic_provisioning():
    # Explicit opt-out must leave offline tests and caller-managed hosts alone.
    judge = ModelBJudge(ModelBJudgeConfig(auto_setup=False))
    with patch.object(judge, "is_available", return_value=False), \
         patch("sword.b_model.start_ollama_server") as start, \
         patch("sword.b_model.pull_ollama_model") as pull:
        assert not judge.ensure_ready()
        start.assert_not_called()
        pull.assert_not_called()
