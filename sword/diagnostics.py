"""Live component checks for the environment in which Sword will train."""

import importlib
import json
import math
import platform
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import requests
import torch

from .b_model import ModelBJudge, model_b_config
from .defaults import DEFAULT_GRPO_MODEL
from .harness import get_harness, WorkspaceSandbox
from .reward import MultiMetricRewardScorer
from .runtime_setup import (
    HARNESS_NAMES, configure_training_environment, prepare_harnesses, setup_environment,
)
from .vllm_patch import inspect_vllm_installation


class RuntimeTestSuite:
    def __init__(self, mock_test: bool = False):
        self.mock_test = mock_test
        self.checks = {}

    def skip(self, name: str, reason: str):
        self.checks[name] = {"status": "skipped", "reason": reason}

    def check(self, name: str, operation, *, live: bool = False):
        if live and self.mock_test:
            self.skip(name, "mock_test=True; requires the real target environment")
            return {}
        print(f"[Sword Test] {name}...", flush=True)
        started = time.monotonic()
        try:
            result = operation() or {}
            if result.get("ok") is False:
                entry = {"status": "failed", "details": result,
                         "error": result.get("error") or f"{name} failed: {result}"}
            else:
                entry = {"status": "passed", "details": result}
        except Exception as exc:
            entry = {"status": "failed", "error": str(exc)}
        entry["elapsed_s"] = round(time.monotonic() - started, 3)
        self.checks[name] = entry
        print(f"[Sword Test] {name}: {entry['status'].upper()}", flush=True)
        if entry.get("error"):
            print(f"    {entry['error']}", flush=True)
        return entry.get("details", {})

    @property
    def ok(self):
        return not any(check["status"] == "failed" for check in self.checks.values())

    def report(self):
        return {
            "mode": "mock" if self.mock_test else "real", "ok": self.ok,
            **{status: sum(check["status"] == status for check in self.checks.values())
               for status in ("passed", "failed", "skipped")},
            "checks": self.checks,
        }


def check_cuda() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable in the target environment.")
    x = torch.randn(16, 16, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(x.grad).all()):
        raise RuntimeError("CUDA forward/backward returned nonfinite values.")
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count(),
        "free_vram_gb": round(free / 1024**3, 2), "total_vram_gb": round(total / 1024**3, 2),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def check_dependencies(load_in_4bit=False, load_in_8bit=False) -> Dict[str, Any]:
    names = ["transformers", "peft", "accelerate", "datasets", "huggingface_hub", "requests"]
    if load_in_4bit or load_in_8bit:
        names.append("bitsandbytes")
    versions = {}
    for name in names:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")
    return {"python": platform.python_version(), "system": platform.system(), "versions": versions}


def check_vllm() -> Dict[str, Any]:
    if platform.system() != "Linux":
        raise RuntimeError("The vLLM target-environment suite requires Linux or WSL2.")
    result = inspect_vllm_installation()
    if not result["ok"]:
        return result
    module = importlib.import_module("vllm")
    if not callable(getattr(module, "LLM", None)):
        raise RuntimeError("vLLM imported without a usable LLM class.")
    return {**result, "version": getattr(module, "__version__", "unknown")}


def check_target_model(model: str, HF_TOKEN=None, model_cache_dir=None,
                       model_revision=None) -> Dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    options = {
        "token": HF_TOKEN, "cache_dir": model_cache_dir, "revision": model_revision,
        "trust_remote_code": True, "local_files_only": True,
    }
    config = AutoConfig.from_pretrained(model, **options)
    # Resolve the actual architecture, rather than accepting an arbitrary config file.
    model_class = AutoModelForCausalLM._model_mapping[type(config)]
    tokenizer = AutoTokenizer.from_pretrained(model, **options)
    ids = tokenizer.encode("Sword target-environment tokenization check.", add_special_tokens=True)
    if not ids or not tokenizer.decode(ids):
        raise RuntimeError("Target tokenizer failed to encode/decode a nonempty prompt.")
    return {
        "model": model, "config_class": type(config).__name__,
        "model_class": model_class.__name__, "tokenizer_class": type(tokenizer).__name__,
        "prompt_tokens": len(ids), "revision": model_revision,
    }


def _harness_turns(name: str):
    path = "sword_probe.txt"
    code = f"from pathlib import Path\nassert Path('{path}').read_text() == 'beta'\nprint('SWORD_TOOL_OK')"
    fence = chr(96) * 3
    python_block = f"{fence}python\n{code}\n{fence}"
    if name == "codex":
        return ["\n".join([
            f"*** Begin Patch\n*** Add File: {path}\n+alpha\n*** End Patch",
            f"*** Begin Patch\n*** Update File: {path}\n@@\n-alpha\n+beta\n*** End Patch",
            python_block,
        ])]
    if name == "gemini_cli":
        return [
            "call:write_file" + json.dumps({"file_path": path, "content": "alpha"}),
            "call:replace" + json.dumps({"file_path": path, "old_string": "alpha", "new_string": "beta"}),
            "call:read_file" + json.dumps({"file_path": path}),
            "call:python" + json.dumps({"code": code}),
        ]
    calls = [
        {"action": "write", "parameters": {"path": path, "content": "alpha"}},
        {"action": "edit", "parameters": {"path": path, "oldString": "alpha", "newString": "beta"}},
        {"action": "read", "parameters": {"path": path}},
    ]
    if name == "pi":
        calls[1]["parameters"] = {"path": path, "oldText": "alpha", "newText": "beta"}
    return ["<tool_call>" + json.dumps(call) + "</tool_call>" for call in calls] + [python_block]


def check_harness_tools(name: str) -> Dict[str, Any]:
    # Fixed tool requests test the parser and real sandbox, without relying on sampled text.
    turns = _harness_turns(name)
    expected_tools = 3 if name == "codex" else len(turns)
    with tempfile.TemporaryDirectory(prefix=f"sword-test-{name}-") as directory:
        harness = get_harness(name, workspace_dir=directory)
        if harness is None:
            raise RuntimeError(f"Harness could not initialize: {name}")
        iterator = iter(turns)
        result = harness.run(
            prompt="Write, edit, read, and execute in the diagnostic workspace.",
            model_generate_fn=lambda _: next(iterator, "Complete."),
            max_turns=len(turns), task_id=f"diagnostic-{name}",
        )
        if (result.tools_called != expected_tools or result.tools_succeeded != expected_tools
                or result.tools_failed or result.error_message):
            raise RuntimeError(f"Harness tool execution failed: {result.to_dict()}")
        if (Path(directory) / "sword_probe.txt").read_text() != "beta":
            raise RuntimeError("Harness write/edit did not produce the expected file contents.")
        return {**result.to_dict(), "workspace_verified": True}


def check_sandbox_shell():
    with tempfile.TemporaryDirectory(prefix="sword-test-shell-") as directory:
        sandbox = WorkspaceSandbox(base_dir=directory)
        ok, output = sandbox.run_bash("echo SWORD_SHELL_OK")
        if not ok or "SWORD_SHELL_OK" not in output:
            raise RuntimeError(f"Harness shell could not execute a command: {output}")
        return {"shell_execution": True}


def _model_matches(name: str, requested: str) -> bool:
    def tag(value):
        return value if ":" in value.rsplit("/", 1)[-1] else f"{value}:latest"
    return tag(name) == tag(requested)


def _running_models(judge):
    response = requests.get(f"{judge.url}/api/ps", timeout=10)
    response.raise_for_status()
    return response.json().get("models", [])


def check_judge_gpu(judge) -> Dict[str, Any]:
    active = [model for model in _running_models(judge)
              if _model_matches(model.get("name") or model.get("model", ""), judge.active_model_name)]
    if not active or not any(model.get("size_vram", 0) > 0 for model in active):
        raise RuntimeError(f"Ollama is not using GPU VRAM for {judge.active_model_name}.")
    return {"model": judge.active_model_name, "size_vram": sum(m.get("size_vram", 0) for m in active)}


def check_judge_unloaded(judge, timeout_s: float = 10) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        if not any(_model_matches(m.get("name") or m.get("model", ""), judge.active_model_name)
                   for m in _running_models(judge)):
            return {"model": judge.active_model_name, "unloaded": True}
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Ollama judge remained loaded after release: {judge.active_model_name}")
        time.sleep(0.25)


def check_live_scoring(judge) -> Dict[str, Any]:
    if not judge.prepare_for_eval() or not judge.ensure_ready(load_model=True):
        raise RuntimeError(judge.last_error or "Insufficient VRAM to load the judge.")
    scorer = MultiMetricRewardScorer(judge)
    result = scorer.score(
        problem="What is 2 + 2?", solution="4", reasoning_rule="Show deduction.",
        answer_rule="Answer 4.", candidate_response="<think>2 plus 2 equals 4.</think>4",
    )
    if judge.last_evaluation_source in (None, "deterministic"):
        raise RuntimeError(judge.last_error or "Scoring did not use a live Ollama model.")
    return {"evaluation_source": judge.last_evaluation_source,
            "scores": result.to_dict(), "thinking_received": judge.last_thinking_received}


class _FallbackProbeJudge(ModelBJudge):
    """Fail only the primary request; fallback requests still use the real Ollama API."""
    failure_injected = False

    def _request_chat(self, payload):
        if payload["model"] == self.config.model_name:
            self.failure_injected = True
            raise RuntimeError("Diagnostic: simulated primary model load failure during scoring")
        return super()._request_chat(payload)


def check_live_fallback(config) -> Dict[str, Any]:
    judge = _FallbackProbeJudge(replace(config, auto_unload=True))
    fallback = config.fallback_model_name
    try:
        if not judge.prepare_for_eval():
            raise RuntimeError("Insufficient VRAM for the fallback probe.")
        # Use the scorer itself so the same reward is retried after the injected failure.
        scorer = MultiMetricRewardScorer(judge)
        options = dict(problem="What is 2 + 2?", solution="4", candidate_response="<think>2+2=4</think>4")
        scorer.score(**options)
        if judge.last_evaluation_source != fallback:
            raise RuntimeError(judge.last_error or "The scorer did not retry with the configured fallback.")
        scorer.score(**options)
        if judge.active_model_name != fallback or judge.evaluation_counts.get(fallback) != 2:
            raise RuntimeError("The successful fallback did not remain active for the next reward.")
        gpu = check_judge_gpu(judge) if config.force_gpu else None
        return {"model": fallback, "failure_injected": judge.failure_injected,
                "evaluation_counts": dict(judge.evaluation_counts), "gpu": gpu}
    finally:
        judge.release_after_eval()
        check_judge_unloaded(judge)


def check_training_result(result: Dict[str, Any], steps: int, b_model: bool, mock_test: bool):
    if result.get("status") != "completed" or result.get("final_step") != steps:
        raise RuntimeError(f"GRPO did not complete {steps} steps: {result.get('status')}")
    if result.get("visited_rows") != 5:
        raise RuntimeError("GRPO did not visit every diagnostic dataset row.")
    metrics = result.get("step_metrics", [])
    if len(metrics) != steps:
        raise RuntimeError("GRPO is missing per-step loss and gradient measurements.")
    for metric in metrics:
        if any(not math.isfinite(float(metric[key])) for key in ("grpo_loss", "grad_norm")):
            raise RuntimeError("GRPO loss or gradient norm is nonfinite.")
    versions = [int(metric["policy_version"]) for metric in metrics]
    if not versions or versions[0] < 1 or any(b <= a for a, b in zip(versions, versions[1:])):
        raise RuntimeError("GRPO did not publish a new LoRA policy version after every update.")
    if not mock_test:
        if result.get("rollout_backend") != "vllm":
            raise RuntimeError(result.get("vllm_initialization_error") or "GRPO used the PyTorch fallback.")
        if b_model:
            counts = result.get("b_model", {}).get("evaluation_counts", {})
            if not counts or counts.get("deterministic") or not any(counts.values()):
                raise RuntimeError("GRPO reward scoring fell back to deterministic scores.")
    return {"steps": steps, "visited_rows": 5, "rollout_backend": result.get("rollout_backend"),
            "losses": [m["grpo_loss"] for m in metrics], "gradient_norms": [m["grad_norm"] for m in metrics]}


def check_checkpoint(result):
    directory = Path(result["checkpoint_dir"])
    state = json.loads((directory / "training_state.json").read_text(encoding="utf-8"))
    if state["step"] != result["final_step"]:
        raise RuntimeError("Saved checkpoint step does not match the trained step.")
    json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    if (directory / "adapter_model.safetensors").is_file():
        from safetensors.torch import load_file
        weights = load_file(str(directory / "adapter_model.safetensors"))
    else:
        weights = torch.load(directory / "adapter_model.bin", map_location="cpu", weights_only=True)
    if not weights or any(not bool(torch.isfinite(weight).all()) for weight in weights.values()):
        raise RuntimeError("Checkpoint weights are missing or nonfinite.")
    optimizer = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
    if not isinstance(optimizer, dict) or not optimizer.get("param_groups"):
        raise RuntimeError("Optimizer state is missing or unreadable.")
    hashes = (directory / "data_complete.logs").read_text(encoding="utf-8").split()
    if result["dataset_sha256"] not in hashes:
        raise RuntimeError("The completed diagnostic dataset is missing from the checkpoint log.")
    return {"directory": str(directory), "step": state["step"], "adapter_tensors": len(weights)}


def diagnostic_dataset():
    return [
        {
            "problem": "Write a Python function is_prime(n: int) -> bool that checks primality.",
            "solution": "def is_prime(n):\n    return n > 1 and all(n % i for i in range(2, int(n**0.5) + 1))",
            "reasoning_rule": "Explain the algorithm inside <think> tags.",
            "answer_rule": "Provide the Python function in a code block.",
            "can_be_verify": True, "difficulty": "medium", "category": "code", "harness": "opencode",
        },
        {
            "problem": "Write reverse_words(s: str) -> str to reverse the order of words.",
            "solution": "def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
            "reasoning_rule": "Explain whitespace handling.",
            "answer_rule": "Return one Python function.",
            "can_be_verify": True, "difficulty": "easy", "category": "code", "harness": "codex",
        },
        {
            "problem": "Solve for x: 3*x + 15 = 42.", "solution": "9",
            "reasoning_rule": "Isolate x step by step.", "answer_rule": "Give the exact integer.",
            "can_be_verify": True, "difficulty": "easy", "category": "math", "harness": "gemini",
        },
        {
            "problem": "All roses are flowers and some flowers fade quickly. Must all roses fade quickly?",
            "solution": "No", "reasoning_rule": "Check the categorical syllogism.",
            "answer_rule": "Answer Yes or No.", "can_be_verify": False,
            "difficulty": "medium", "category": "reasoning", "harness": "pi",
        },
        {
            "problem": "What is the primary gas in Earth's atmosphere?", "solution": "Nitrogen",
            "reasoning_rule": "Identify the atmospheric composition.", "answer_rule": "Name the gas.",
            "can_be_verify": False, "difficulty": "easy", "category": "science", "harness": None,
        },
    ]


def run_target_diagnostics(
    *, test: bool = True, setup: bool = False, b_model: bool = True,
    model: str = DEFAULT_GRPO_MODEL, B_model_port: int = 11343, concurrent: int = 8,
    load_in_4bit: bool = False, load_in_8bit: bool = False, lora_cfg=None, Argument=None,
    HF_TOKEN=None, mock_test: bool = False, auto_unload_B_model: bool = True,
    B_model_keep_alive: str = "5m", long_train: bool = False, dataset=None, **options,
) -> Dict[str, Any]:
    print("=" * 72 + "\n SWORD TARGET ENVIRONMENT SETUP & TEST SUITE\n" + "=" * 72)
    print(f"Target GRPO model: {model} | Model B enabled: {b_model} | Mock: {mock_test}", flush=True)
    environment = configure_training_environment()
    suite = RuntimeTestSuite(mock_test)
    setup_result = None
    patch_result = {}
    harness_status = {}
    b_diag = {"enabled": b_model, "status": "not_tested"}
    train_result = {"status": "not_requested" if not test and not long_train else "preflight_failed", "final_step": 0}

    def finish():
        report = suite.report()
        print(f"[Sword Test] Passed: {report['passed']} | Failed: {report['failed']} | Skipped: {report['skipped']}", flush=True)
        return {
            "ok": suite.ok, "mode": "mock" if mock_test else "real",
            "environment_validated": suite.ok and not mock_test,
            "model": model, "environment": environment, "setup_result": setup_result,
            "patch_result": patch_result, "harness_status": harness_status,
            "b_model_diagnostic": b_diag, "train_result": train_result, "test_suite": report,
        }

    if setup:
        setup_result = setup_environment(
            model=model, b_model=b_model, B_model_port=B_model_port, HF_TOKEN=HF_TOKEN,
            load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, mock_test=mock_test,
            auto_unload_B_model=auto_unload_B_model, B_model_keep_alive=B_model_keep_alive,
            **options,
        )
        patch_result = setup_result["steps"].get("vllm", {})
        for name, result in setup_result["steps"].items():
            if "skipped" in result:
                suite.skip(f"setup.{name}", result["skipped"])
            else:
                suite.check(f"setup.{name}", lambda result=result: result)
        if setup_result.get("restart_kernel_required"):
            suite.check("setup.restart", lambda: {
                "ok": False, "error": "Imported packages changed. Restart the Python kernel and rerun this call.",
            })
            train_result["status"] = "restart_required"
        if not setup_result["ok"] or setup_result.get("restart_kernel_required"):
            suite.skip("grpo_training", "Setup did not produce a ready runtime")
            return finish()

    suite.check("cuda_forward_backward", check_cuda, live=True)
    suite.check("dependencies", lambda: check_dependencies(load_in_4bit, load_in_8bit), live=True)
    patch_result = suite.check("vllm_import_and_patches", check_vllm, live=True) or patch_result
    suite.check("target_model_and_tokenizer", lambda: check_target_model(
        model, HF_TOKEN, options.get("model_cache_dir"), options.get("model_revision"),
    ), live=True)
    harness_status = prepare_harnesses(auto_clone=False, mock_test=mock_test)
    if test:
        suite.check("harness.shell", check_sandbox_shell)
    for name in HARNESS_NAMES:
        if mock_test:
            suite.skip(f"harness.{name}.repository", "mock_test=True; no repositories downloaded")
        else:
            suite.check(f"harness.{name}.repository", lambda name=name: harness_status[name])
        if test:
            suite.check(f"harness.{name}.tools", lambda name=name: check_harness_tools(name))

    b_options = dict(options)
    b_options.setdefault("B_model_auto_setup", setup)
    config = model_b_config(
        B_model_port, mock_test=mock_test, auto_unload=auto_unload_B_model,
        keep_alive=B_model_keep_alive, **b_options,
    )
    if b_model and not mock_test and suite.ok:
        # Diagnostic probes own their temporary GPU loads and always release them.
        judge = ModelBJudge(replace(config, auto_unload=True))
        if setup_result:
            judge.active_model_name = setup_result["steps"]["b_model"].get(
                "active_model_name", judge.active_model_name,
            )
        try:
            if test:
                suite.check("b_model.live_scoring", lambda: check_live_scoring(judge))
                if config.force_gpu and suite.checks["b_model.live_scoring"]["status"] == "passed":
                    suite.check("b_model.gpu", lambda: check_judge_gpu(judge))
                elif not config.force_gpu:
                    suite.skip("b_model.gpu", "B_model_force_gpu=False")
                judge.release_after_eval()
                suite.check("b_model.unload", lambda: check_judge_unloaded(judge))
                if config.fallback_model_name and config.fallback_model_name != config.model_name:
                    suite.check("b_model.live_fallback", lambda: check_live_fallback(config))
                else:
                    suite.skip("b_model.live_fallback", "No distinct fallback model configured")
            else:
                suite.check("b_model.connection", lambda: {
                    "ok": judge.ensure_ready(), "error": judge.last_error,
                })
            b_diag = {"enabled": True, **judge.check_connection()}
        finally:
            judge.release_after_eval()
    else:
        reason = "b_model=False" if not b_model else ("mock_test=True" if mock_test else "Preflight checks failed")
        for name in ("live_scoring", "gpu", "unload", "live_fallback"):
            suite.skip(f"b_model.{name}", reason)
        b_diag = {"enabled": b_model, "status": "skipped", "reason": reason,
                  "model_name": config.model_name, "fallback_model_name": config.fallback_model_name}

    if (test or long_train) and suite.ok:
        from .long_trainer import run_long_train
        arguments = dict(Argument or {})
        if test:
            arguments.setdefault("steps", 5)
            arguments.setdefault("concurrency_g", concurrent)
            arguments.setdefault("max_seq_len", 2048)
            arguments.setdefault("max_new_tokens", 128)
            arguments.setdefault("micro_batch_size", 1)
            arguments.setdefault("output_dir", "checkpoints/diagnostics")
            arguments.setdefault("allow_retest", True)
        hf_overrides = options.get("hf_overrides") or {
            "allow_global_per_layer_attribute_access": True,
            "text_config": {"allow_global_per_layer_attribute_access": True},
        }
        train_options = dict(b_options)
        train_options.update(
            dataset=diagnostic_dataset() if test else dataset, model=model, b_model=b_model,
            B_model_port=B_model_port, HF_TOKEN=HF_TOKEN,
            auto_unload_B_model=auto_unload_B_model, B_model_keep_alive=B_model_keep_alive,
            custom_checkpoint_save=True, is_save_and_upload_checkpoint=False, upload_id=None,
            resume_from_checkpoint=None, Argument=arguments,
            lora_cfg=lora_cfg or {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
                                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                                    "gate_proj", "up_proj", "down_proj"]},
            show_detail_logs=True, score_logs=True, return_step_metrics=True,
            load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
            require_vllm=not mock_test, local_files_only=True, mock_test=mock_test,
            hf_overrides=hf_overrides,
        )
        train_options.setdefault("vllm_quantization", "fp8" if load_in_8bit else None)
        train_options.setdefault("kv_cache_dtype", "fp8" if load_in_8bit else "auto")
        train_options.setdefault("speculative_ngram", True)
        train_options.setdefault("offload_trainer_during_rollout", True)

        def train_and_validate():
            nonlocal train_result
            train_result = run_long_train(**train_options)
            if test:
                return check_training_result(train_result, int(arguments["steps"]), b_model, mock_test)
            return {"ok": train_result.get("status") == "completed", "status": train_result.get("status")}

        suite.check("grpo_training", train_and_validate)
        if train_result.get("checkpoint_dir"):
            suite.check("checkpoint_readback", lambda: check_checkpoint(train_result))
    elif test or long_train:
        suite.skip("grpo_training", "Preflight checks failed")
        suite.skip("checkpoint_readback", "Training did not run")
    return finish()
