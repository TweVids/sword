"""Offline regressions for the live setup/test API and truthful readiness reports."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests
import torch

import sword
from sword import diagnostics, runtime_setup
from sword.b_model import ModelBJudgeConfig
from sword.defaults import DEFAULT_GRPO_MODEL, DEFAULT_B_MODEL, DEFAULT_B_MODEL_FALLBACK
from sword.grpo_bridge import GRPOBridge, GRPOBridgeConfig
from sword.harness import prompts
from sword import vllm_patch


def training_result():
    return {
        "status": "completed", "final_step": 5, "visited_rows": 5,
        "rollout_backend": "vllm", "dataset_sha256": "dataset-hash",
        "step_metrics": [{"grpo_loss": 0.1, "grad_norm": 0.2, "policy_version": i + 1}
                         for i in range(5)],
        "b_model": {"evaluation_counts": {DEFAULT_B_MODEL: 10}},
    }


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data).encode()
    return result


def ready_harnesses():
    return {name: {"ok": True, "repo_ready": True, "prompt_chars": 100, "initialized": True}
            for name in runtime_setup.HARNESS_NAMES}


@pytest.fixture
def provisioned():
    with patch.object(runtime_setup.platform, "system", return_value="Linux"), \
         patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(runtime_setup, "loaded_package_versions", return_value={}), \
         patch.object(runtime_setup, "changed_loaded_packages", return_value=[]), \
         patch.object(runtime_setup, "provision_dependencies", return_value={"ok": True}) as dependencies, \
         patch.object(runtime_setup, "vllm_patcher", return_value={"ok": True, "restart_kernel_required": False}) as vllm, \
         patch.object(runtime_setup, "prepare_harnesses", return_value=ready_harnesses()) as harnesses, \
         patch.object(runtime_setup, "download_target_model", return_value={"ok": True, "path": "/cached/model"}) as download, \
         patch.object(runtime_setup, "provision_model_b", return_value={"ok": True}) as judge:
        yield SimpleNamespace(dependencies=dependencies, vllm=vllm, harnesses=harnesses,
                              download=download, judge=judge)


def test_train_selects_gemma_grpo_defaults_and_exposes_new_flags():
    with patch("sword.train_entry._run_prep_and_logs", return_value={"ok": True}) as runner:
        sword.train(prep_and_logs=True, setup=True, test=True, b_model=True)
    options = runner.call_args.kwargs
    assert options["model"] == DEFAULT_GRPO_MODEL
    assert options["load_in_8bit"] is True
    assert options["vllm_quantization"] == "fp8"
    assert options["setup"] is True and options["b_model"] is True
    assert ModelBJudgeConfig().model_name == DEFAULT_B_MODEL
    assert ModelBJudgeConfig().fallback_model_name == DEFAULT_B_MODEL_FALLBACK


def test_train_preserves_explicit_target_quantization_and_authentication():
    with patch("sword.train_entry._run_prep_and_logs", return_value={}) as runner:
        sword.train(prep_and_logs=True, model="custom/model", load_in_4bit=True,
                    HF_TOKEN="private-token", model_revision="pinned", b_model=False)
    options = runner.call_args.kwargs
    assert options["model"] == "custom/model"
    assert options["load_in_4bit"] and not options["load_in_8bit"]
    assert options["HF_TOKEN"] == "private-token"
    assert options["model_revision"] == "pinned"
    assert options["b_model"] is False


def test_setup_provisions_each_component_on_the_training_port(provisioned):
    result = sword.setup_environment(B_model_port=12456, HF_TOKEN="private-token",
                                    B_model_models_dir="/persistent/models", model_revision="pinned")
    assert result["ok"] and not result["restart_kernel_required"]
    provisioned.vllm.assert_called_once()
    assert provisioned.download.call_args.args == (DEFAULT_GRPO_MODEL, "private-token", None, "pinned")
    assert provisioned.judge.call_args.args == (12456,)
    assert provisioned.judge.call_args.kwargs["B_model_models_dir"] == "/persistent/models"
    assert "private-token" not in json.dumps(result)


def test_setup_vllm_failure_blocks_large_downloads_and_judge_start(provisioned):
    provisioned.vllm.return_value = {"ok": False, "error": "wheel unavailable"}
    result = sword.setup_environment()
    assert not result["ok"]
    assert result["steps"]["vllm"]["error"] == "wheel unavailable"
    provisioned.download.assert_not_called()
    provisioned.judge.assert_not_called()


def test_setup_b_model_false_avoids_ollama(provisioned):
    result = sword.setup_environment(b_model=False)
    assert result["ok"]
    provisioned.judge.assert_not_called()
    assert result["steps"]["b_model"]["skipped"] == "b_model=False"


def test_mock_setup_performs_no_installs_downloads_or_server_calls(provisioned):
    result = sword.setup_environment(mock_test=True)
    assert result["ok"] and result["mode"] == "mock"
    assert result["environment_validated"] is False
    for tool in vars(provisioned).values():
        tool.assert_not_called()


def test_setup_restart_stops_training_after_preparation():
    with patch.object(runtime_setup, "setup_environment", return_value={
        "ok": True, "restart_kernel_required": True,
    }), patch("sword.long_trainer.run_long_train") as train:
        result = sword.train(setup=True, long_train=True)
    assert not result["ok"] and result["status"] == "restart_required"
    train.assert_not_called()


def test_dependencies_use_current_interpreter_pip_when_uv_is_missing():
    def which(name):
        return None if name == "uv" else "/usr/bin/" + name
    with patch.object(runtime_setup.shutil, "which", side_effect=which), \
         patch.object(runtime_setup.importlib.util, "find_spec",
                      side_effect=lambda name: None if name == "peft" else object()), \
         patch.object(runtime_setup.subprocess, "run",
                      return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
        assert runtime_setup.provision_dependencies()["ok"]
    assert run.call_args.args[0] == [sys.executable, "-m", "pip", "install", "peft"]


def test_setup_upgrades_transformers_when_gemma4_architecture_is_missing():
    with patch.object(runtime_setup.shutil, "which", return_value="/tool"), \
         patch.object(runtime_setup.importlib.util, "find_spec", return_value=object()), \
         patch("transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES", {}), \
         patch.object(runtime_setup.subprocess, "run",
                      return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
        assert runtime_setup.provision_dependencies()["ok"]
    assert run.call_args.args[0][-2:] == ["--upgrade", "transformers"]


def test_target_download_forwards_gated_model_access_and_validates_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").touch()
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as download:
        result = runtime_setup.download_target_model(DEFAULT_GRPO_MODEL, "private-token", "/cache", "revision")
    assert result["ok"]
    assert download.call_args.kwargs["token"] == "private-token"
    assert download.call_args.kwargs["revision"] == "revision"
    assert "private-token" not in json.dumps(result)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="No model weights"):
        runtime_setup.download_target_model(str(tmp_path))


@pytest.mark.parametrize("name", runtime_setup.HARNESS_NAMES)
def test_harness_probe_executes_real_native_tools(name):
    result = diagnostics.check_harness_tools(name)
    assert result["workspace_verified"]
    assert result["tools_called"] >= 3
    assert result["tools_succeeded"] == result["tools_called"]


def test_empty_harness_execution_cannot_pass():
    harness = MagicMock()
    harness.run.return_value = sword.HarnessResult()
    with patch.object(diagnostics, "get_harness", return_value=harness), \
         pytest.raises(RuntimeError, match="tool execution failed"):
        diagnostics.check_harness_tools("opencode")


def test_harness_shell_executes_a_real_command():
    assert diagnostics.check_sandbox_shell()["shell_execution"]


def test_suite_aggregates_failures_and_never_calls_mock_live_checks():
    suite = diagnostics.RuntimeTestSuite(mock_test=True)
    live = MagicMock()
    suite.check("live", live, live=True)
    suite.check("bad", lambda: {"ok": False, "error": "not ready"})
    suite.check("good", lambda: {"ok": True})
    report = suite.report()
    assert not report["ok"]
    assert report["passed"] == report["failed"] == report["skipped"] == 1
    assert report["checks"]["bad"]["details"]["ok"] is False
    live.assert_not_called()


def test_missing_target_environment_blocks_training_and_does_not_install():
    with patch.object(diagnostics, "check_cuda", side_effect=RuntimeError("no CUDA")), \
         patch.object(diagnostics, "check_dependencies", return_value={}), \
         patch.object(diagnostics, "check_vllm", return_value={"ok": True}), \
         patch.object(diagnostics, "check_target_model", return_value={}), \
         patch.object(diagnostics, "prepare_harnesses", return_value=ready_harnesses()), \
         patch.object(diagnostics, "check_harness_tools", return_value={}), \
         patch.object(diagnostics, "setup_environment") as setup, \
         patch("sword.long_trainer.run_long_train") as train:
        result = sword.train(prep_and_logs=True, b_model=False)
    assert not result["ok"]
    assert result["test_suite"]["checks"]["cuda_forward_backward"]["status"] == "failed"
    train.assert_not_called()
    setup.assert_not_called()


def test_mock_diagnostics_passes_settings_and_cannot_claim_live_validation():
    result = training_result()
    result["rollout_backend"] = "pytorch"
    with patch.object(diagnostics, "prepare_harnesses", return_value=ready_harnesses()), \
         patch.object(diagnostics, "check_harness_tools", return_value={}), \
         patch("sword.long_trainer.run_long_train", return_value=result) as train, \
         patch.object(diagnostics, "requests") as network:
        report = sword.train(prep_and_logs=True, setup=True, mock_test=True, b_model=False,
                             HF_TOKEN="private-token", concurrent=2, B_model_keep_alive="2m")
    assert report["ok"] and report["environment_validated"] is False
    assert report["test_suite"]["mode"] == "mock"
    options = train.call_args.kwargs
    assert options["model"] == DEFAULT_GRPO_MODEL
    assert options["HF_TOKEN"] == "private-token"
    assert options["B_model_keep_alive"] == "2m"
    assert options["load_in_8bit"] and options["b_model"] is False
    assert options["is_save_and_upload_checkpoint"] is False and options["upload_id"] is None
    network.get.assert_not_called()
    network.post.assert_not_called()


@pytest.mark.parametrize("exhaust_thinking_budget", [False, True])
def test_live_fallback_probe_scores_twice_with_the_real_fallback_request_path(exhaust_thinking_budget):
    active = set()
    chats = []
    scores = {"truth": 1, "formatted": 1, "rule_reasoning_follow": 1,
              "rule_answer_follow": 1, "tools_success": 0, "cheating": False}
    score_content = json.dumps(scores)

    def get(url, **_):
        if url.endswith("/api/ps"):
            return response({"models": [{"name": name, "size_vram": 123} for name in active]})
        return response({"models": [{"name": name} for name in (DEFAULT_B_MODEL, DEFAULT_B_MODEL_FALLBACK)]})

    def post(url, json, **_):
        if url.endswith("/api/chat"):
            chats.append((json["model"], json.get("think", "default")))
            active.add(json["model"])
            if exhaust_thinking_budget and json.get("think", True):
                return response({"message": {"thinking": "Still evaluating...", "content": ""},
                                 "done_reason": "length"})
            return response({"message": {"content": score_content}})
        active.discard(json["model"])
        return response({"done": True})

    with patch("sword.b_model.requests.get", side_effect=get), \
         patch("sword.b_model.requests.post", side_effect=post), \
         patch("sword.b_model.ModelBJudge.prepare_for_eval", return_value=True):
        result = diagnostics.check_live_fallback(ModelBJudgeConfig(auto_setup=False))
    assert result["failure_injected"]
    expected_thinking = [True, False, True, False] if exhaust_thinking_budget else [True, True]
    assert chats == [(DEFAULT_B_MODEL_FALLBACK, thinking) for thinking in expected_thinking]
    assert result["evaluation_counts"] == {DEFAULT_B_MODEL_FALLBACK: 2}
    assert not active


def test_missing_fallback_cannot_pass_live_scorer_probe():
    with patch("sword.b_model.requests.get", side_effect=lambda url, **_: response({
        "models": [] if url.endswith("/api/ps") else [{"name": DEFAULT_B_MODEL}],
    })), patch("sword.b_model.requests.post", return_value=response({"done": True})), \
         patch("sword.b_model.ModelBJudge.prepare_for_eval", return_value=True), \
         pytest.raises(RuntimeError, match="simulated primary|retry"):
        diagnostics.check_live_fallback(ModelBJudgeConfig(auto_setup=False))


def test_scorer_restarts_crashed_ollama_then_uses_the_smaller_model():
    from sword.b_model import ModelBJudge
    from sword.reward import MultiMetricRewardScorer
    crashed = False
    models = []
    scores = json.dumps({"truth": 1, "rule_reasoning_follow": 1, "formatted": 1})

    def get(url, **_):
        if crashed:
            raise requests.ConnectionError("Ollama process exited")
        return response({"models": [{"name": name} for name in (DEFAULT_B_MODEL, DEFAULT_B_MODEL_FALLBACK)]})

    def post(url, json, **_):
        nonlocal crashed
        models.append(json["model"])
        if json["model"] == DEFAULT_B_MODEL:
            crashed = True
            raise requests.ConnectionError("Ollama process exited during primary load")
        return response({"message": {"content": scores}})

    def restart(**_):
        nonlocal crashed
        crashed = False
        return True

    judge = ModelBJudge(ModelBJudgeConfig(port=12456))
    with patch("sword.b_model.requests.get", side_effect=get), \
         patch("sword.b_model.requests.post", side_effect=post), \
         patch("sword.b_model.start_ollama_server", side_effect=restart) as start:
        MultiMetricRewardScorer(judge).score("2+2?", "4", "<think>2+2=4</think>4")
    assert judge.last_evaluation_source == DEFAULT_B_MODEL_FALLBACK
    assert models == [DEFAULT_B_MODEL, DEFAULT_B_MODEL_FALLBACK]
    assert start.call_args.kwargs["host"] == "http://127.0.0.1:12456"


def test_b_model_false_runs_real_mock_grpo_without_creating_a_judge(tmp_path):
    with patch("sword.long_trainer.ModelBJudge") as judge:
        result = sword.run_long_train(
            dataset=[{"problem": "2+2?", "solution": "4"}, {"problem": "3+3?", "solution": "6"}],
            model="mock", b_model=False, is_save_and_upload_checkpoint=False,
            show_detail_logs=False,
            Argument={"steps": 2, "concurrency_g": 2, "max_seq_len": 64,
                      "max_new_tokens": 2, "output_dir": str(tmp_path)},
        )
    assert result["status"] == "completed"
    assert result["b_model"]["enabled"] is False
    judge.assert_not_called()


@pytest.mark.parametrize("failure", ["backend", "deterministic", "loss", "metrics", "versions", "rows"])
def test_live_training_report_rejects_incomplete_or_synthetic_success(failure):
    result = training_result()
    if failure == "backend":
        result["rollout_backend"] = "pytorch"
    elif failure == "deterministic":
        result["b_model"]["evaluation_counts"]["deterministic"] = 1
    elif failure == "loss":
        result["step_metrics"][0]["grpo_loss"] = float("nan")
    elif failure == "metrics":
        result["step_metrics"] = []
    elif failure == "versions":
        result["step_metrics"][1]["policy_version"] = 1
    elif failure == "rows":
        result["visited_rows"] = 2
    with pytest.raises(RuntimeError):
        diagnostics.check_training_result(result, 5, b_model=True, mock_test=False)


def test_checkpoint_readback_rejects_corrupt_weights_and_optimizer(tmp_path):
    from safetensors.torch import save_file
    result = training_result()
    result["checkpoint_dir"] = str(tmp_path)
    (tmp_path / "training_state.json").write_text(json.dumps({"step": 5}))
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "data_complete.logs").write_text("dataset-hash")
    torch.save({"state": {}, "param_groups": [{}]}, tmp_path / "optimizer.pt")
    weights = tmp_path / "adapter_model.safetensors"
    save_file({"adapter": torch.tensor([float("nan")])}, str(weights))
    with pytest.raises(RuntimeError, match="nonfinite"):
        diagnostics.check_checkpoint(result)
    save_file({"adapter": torch.tensor([1.0])}, str(weights))
    assert diagnostics.check_checkpoint(result)["step"] == 5
    torch.save({}, tmp_path / "optimizer.pt")
    with pytest.raises(RuntimeError, match="Optimizer"):
        diagnostics.check_checkpoint(result)


def test_vllm_patch_reuses_installed_package_and_supports_patch_only(tmp_path):
    root = tmp_path / "vllm"
    common = root / "model_executor/layers/rotary_embedding/common.py"
    common.parent.mkdir(parents=True)
    common.write_text("def rotary():\n    from flash_attn.ops.triton.rotary import apply_rotary\n    return apply_rotary\n")
    spec = SimpleNamespace(origin=str(root / "__init__.py"))
    with patch.object(vllm_patch.importlib.util, "find_spec", return_value=spec), \
         patch.object(vllm_patch.metadata, "version", return_value="0.23.0"), \
         patch.object(vllm_patch, "loaded_package_versions", return_value={}), \
         patch.object(vllm_patch.subprocess, "run") as run:
        assert vllm_patch.bootstrap_environment(install=False)["ok"]
        source = common.read_text()
        assert vllm_patch.bootstrap_environment()["ok"]
        assert common.read_text() == source
        assert vllm_patch.inspect_vllm_installation()["ok"]
    run.assert_not_called()


def test_harness_checkout_failure_is_not_reported_as_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "THIRD_PARTY_DIR", str(tmp_path))
    monkeypatch.setenv("SWORD_AUTO_CLONE_HARNESS", "1")
    with patch("subprocess.run", side_effect=[
        SimpleNamespace(returncode=0), SimpleNamespace(returncode=0),
        subprocess.CalledProcessError(1, ["git", "checkout"]),
    ]) as run, pytest.warns(RuntimeWarning):
        assert not prompts.ensure_harness_repo("opencode", verify=True)
    assert run.call_args.kwargs["check"] is True


def test_mock_trainer_never_attempts_the_real_unsloth_target_load():
    with patch("sword.grpo_bridge.HAS_UNSLOTH", True), \
         patch("sword.grpo_bridge.FastLanguageModel", create=True) as unsloth:
        bridge = GRPOBridge(GRPOBridgeConfig(device="cpu", max_new_tokens=8, max_model_len=64))
        bridge.initialize_trainer(mock_test=True)
        assert bridge.initialize_vllm() is False
    unsloth.from_pretrained.assert_not_called()
    bridge.weight_sync.cleanup()


@pytest.mark.parametrize("update", [False, True])
def test_existing_harness_revision_is_verified_or_safely_updated(tmp_path, monkeypatch, update):
    monkeypatch.setattr(prompts, "THIRD_PARTY_DIR", str(tmp_path))
    monkeypatch.setenv("SWORD_AUTO_CLONE_HARNESS", "1")
    (tmp_path / "opencode").mkdir()
    pinned = prompts.HARNESS_GIT_COMMITS["opencode"]
    replies = [SimpleNamespace(stdout="other-revision")]
    if update:
        replies += [SimpleNamespace(), SimpleNamespace(), SimpleNamespace(stdout=pinned)]
    with patch("subprocess.run", side_effect=replies) as run:
        assert prompts.ensure_harness_repo("opencode", verify=True, update=update) is update
    if update:
        assert "--force" not in run.call_args_list[2].args[0]
        assert run.call_args_list[2].args[0][-2:] == ["checkout", pinned]
    else:
        assert run.call_count == 1


def test_trainer_loading_uses_quantization_authentication_and_the_requested_revision():
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.for_model("qwen2", vocab_size=32, hidden_size=8, intermediate_size=16,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = AutoModelForCausalLM.from_config(config)
    model.is_loaded_in_8bit = True
    bridge_config = GRPOBridgeConfig(
        device="cpu", load_in_8bit=True, hf_token="private-token", model_revision="pinned",
        model_cache_dir="/cache", local_files_only=True, max_model_len=64, max_new_tokens=8,
    )
    with patch("sword.grpo_bridge.HAS_UNSLOTH", False), \
         patch("sword.grpo_bridge.AutoTokenizer.from_pretrained", return_value=MagicMock()) as tokenizer, \
         patch("sword.grpo_bridge.AutoModelForCausalLM.from_pretrained", return_value=model) as load, \
         patch("peft.prepare_model_for_kbit_training", return_value=model) as prepare, \
         patch("peft.get_peft_model", return_value=model):
        bridge = GRPOBridge(bridge_config)
        bridge.initialize_trainer()
    options = load.call_args.kwargs
    assert options["quantization_config"].load_in_8bit
    assert options["token"] == "private-token"
    assert options["revision"] == "pinned" and options["cache_dir"] == "/cache"
    assert options["local_files_only"] is True
    assert tokenizer.call_args.kwargs["token"] == "private-token"
    assert "private-token" not in repr(bridge_config)
    prepare.assert_called_once_with(model)
    bridge.weight_sync.cleanup()
