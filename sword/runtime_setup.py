"""Explicit, repeatable provisioning for a Sword Linux GPU training environment."""

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .defaults import DEFAULT_GRPO_MODEL
from .b_model import ModelBJudge, model_b_config
from .harness import ensure_harness_repo, get_harness, get_harness_system_prompt
from .harness.prompts import harness_repo_policy
from .vllm_patch import vllm_patcher, loaded_package_versions, changed_loaded_packages


HARNESS_NAMES = ("opencode", "codex", "gemini_cli", "pi")


# ---------------------------------------------------------------------------
# Self-update: keep the INSTALLED sword package in sync with the source repo.
# The runtime imports sword from site-packages, so a plain `git pull` cannot
# change behavior. Setup running with setup=True therefore grafts the repo
# code into the installed package (reinstall, or byte-copy if that fails).
# Disable with SWORD_SELF_UPDATE=0. Point at a nonstandard checkout with
# SWORD_REPO_DIR=/path/to/sword.
# ---------------------------------------------------------------------------

def _package_hashes(sword_dir: Path) -> Dict[str, str]:
    import hashlib
    return {path.relative_to(sword_dir).as_posix(): hashlib.sha256(path.read_bytes())
            for path in sorted(sword_dir.rglob("*.py"))}


def _find_sword_source_repo() -> Optional[Path]:
    env_dir = os.environ.get("SWORD_REPO_DIR")
    candidates: list = []
    if env_dir:
        candidates.append(Path(env_dir))
    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.extend(cwd.parents)
    for candidate in candidates:
        try:
            inside = candidate / "sword" / "grpo_bridge.py"
            is_repo = (candidate / ".git").exists()
            if inside.exists() and is_repo:
                return candidate
        except OSError:
            continue
    return None


def self_update() -> Dict[str, Any]:
    """
    BUG-FIX rollout helper (user-requested 2026-10-25): auto-update the installed
    sword package from the source repo inside the running process.
    """
    from . import sword_installed_dir
    installed_dir = Path(sword_installed_dir())
    if os.environ.get("SWORD_SELF_UPDATE", "1").strip().lower() in ("0", "false", "no"):
        return {"ok": True, "skipped": "SWORD_SELF_UPDATE=0"}

    repo = _find_sword_source_repo()
    if repo is None:
        installed_hashes = _package_hashes(installed_dir)
        return {"ok": True, "skipped": "No source repo found (SWORD_REPO_DIR unset)",
                "installed_from": str(installed_dir), "installed_files": len(installed_hashes)}

    if repo.resolve() == installed_dir.parent.resolve():
        return {"ok": True, "skipped": "Running from the source repo; nothing to graft",
                "source_repo": str(repo), "installed_from": str(installed_dir)}

    before = {name: mod for name, mod in list(sys.modules.items())
              if name == "sword" or name.startswith("sword.")}
    before_hashes = _package_hashes(installed_dir)

    # 1. Pull the source repo so the graft contains the remote change set.
    pull_tail = ""
    try:
        run = subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"],
                             text=True, capture_output=True, timeout=90)
        pull_tail = ((run.stdout or "") + (run.stderr or ""))[-500:]
    except Exception as exc:
        pull_tail = f"git pull failed: {exc} (using the existing checkout)"

    repo_hashes = _package_hashes(Path(repo) / "sword")
    already_current = repo_hashes == before_hashes
    changed_files = []
    for rel, digest in repo_hashes.items():
        if before_hashes.get(rel) != digest:
            changed_files.append(rel)

    results: Dict[str, Any] = {
        "ok": True, "source_repo": str(repo), "installed_from": str(installed_dir),
        "changed_files": [] if already_current else changed_files,
    }
    if already_current:
        results["skipped"] = (f"Installed package already matches {repo.resolve()}; "
                              f"{'pull said: ' + pull_tail.strip() if pull_tail.strip() else 'no new commits'}")
        return results

    # 2. Preferred path: reinstall the repo as a wheel into the live venv.
    reinstalled = False
    command: list = []
    if shutil.which("uv"):
        command = ["uv", "pip", "install", "--python", sys.executable,
                   "--force-reinstall", "--no-deps", "--no-build-isolation", str(repo)]
    else:
        command = [sys.executable, "-m", "pip", "install", "--force-reinstall",
                   "--no-deps", str(repo)]
    try:
        run = subprocess.run(command, text=True, capture_output=True, timeout=600)
        results["graft_command"] = " ".join(command)
        results["graft_exit_code"] = run.returncode
        results["graft_tail"] = ((run.stdout or "") + (run.stderr or ""))[-1500:]
        reinstalled = run.returncode == 0
    except Exception as exc:
        results["graft_tail"] = f"reinstall failed: {exc}"

    # 3. Fallback ("self-patch"): byte-copy the repo's python sources over the
    #    installed package when wheel building/reinstalling is not viable.
    if not reinstalled:
        copied = 0
        try:
            for rel, _digest in repo_hashes.items():
                src = Path(repo) / "sword" / rel
                dst = installed_dir / rel
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    copied += 1
            results["self_patch"] = f"copied {copied} files into {installed_dir}"
        except OSError as exc:
            results["ok"] = False
            results["error"] = f"reinstall and self-patch both failed: {exc}"
            return results

    # 4. Reload already-imported sword modules whose bytes changed so the fix is
    #    live in this process (lazily imported modules do not need this).
    import importlib
    importlib.invalidate_caches()
    after_hashes = _package_hashes(installed_dir)
    reloaded, reload_errors = [], []
    stale = sorted((name for name, mod in list(before.items())
                    if getattr(mod, "__file__", None)), key=lambda n: n.count("."))
    for name in stale:
        mod = sys.modules.get(name)
        file = getattr(mod, "__file__", None)
        path = Path(file) if file else None
        rel = str(path.relative_to(installed_dir)) if path and installed_dir in path.parents else None
        if rel and after_hashes.get(rel) != before_hashes.get(rel) and name != "sword":
            try:
                importlib.reload(sys.modules[name])
                reloaded.append(name)
            except Exception as exc:
                reload_errors.append(f"{name}: {exc}")
    results["reloaded_modules"] = reloaded
    results["reload_errors"] = reload_errors
    if reload_errors:
        results["restart_kernel_required_hint"] = (
            f"Some modules failed to hot-reload: {reload_errors}. A kernel restart "
            "guarantees the updated code is live.")
    return results


def configure_training_environment() -> Dict[str, str]:
    values = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_FLASHINFER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    os.environ.update(values)
    try:
        from transformers.configuration_utils import PretrainedConfig
        PretrainedConfig.allow_global_per_layer_attribute_access = True
    except ImportError:
        pass
    return values


def provision_dependencies(load_in_4bit: bool = False, load_in_8bit: bool = False,
                           model: str = DEFAULT_GRPO_MODEL) -> Dict[str, Any]:
    missing_tools = [tool for tool in ("git", "bash") if not shutil.which(tool)]
    if missing_tools:
        if platform.system() != "Linux" or os.geteuid() != 0 or not shutil.which("apt-get"):
            raise RuntimeError(f"Required harness tools are missing: {', '.join(missing_tools)}")
        subprocess.run(["apt-get", "update"], check=True, timeout=300)
        subprocess.run(["apt-get", "install", "-y", *missing_tools], check=True, timeout=300)
    dependencies = {
        "transformers": "transformers", "accelerate": "accelerate", "peft": "peft",
        "datasets": "datasets", "huggingface_hub": "huggingface_hub",
        "requests": "requests>=2.31.0", "sentencepiece": "sentencepiece",
        "safetensors": "safetensors",
    }
    if load_in_4bit or load_in_8bit:
        dependencies["bitsandbytes"] = "bitsandbytes"
    missing = [package for module, package in dependencies.items()
               if importlib.util.find_spec(module) is None]
    upgrade_transformers = False
    if "/gemma-4-" in model.lower() and "transformers" not in missing:
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
            upgrade_transformers = "gemma4" not in CONFIG_MAPPING_NAMES
        except ImportError:
            upgrade_transformers = True
        if upgrade_transformers:
            missing.append("transformers")
    if not missing:
        return {"ok": True, "installed": [], "reused": list(dependencies)}
    command = (["uv", "pip", "install", "--python", sys.executable]
               if shutil.which("uv") else [sys.executable, "-m", "pip", "install"])
    if upgrade_transformers:
        command.append("--upgrade")
    result = subprocess.run([*command, *missing], text=True, capture_output=True)
    importlib.invalidate_caches()
    return {"ok": result.returncode == 0, "installed": missing,
            "returncode": result.returncode, "tail": (result.stdout + result.stderr)[-4000:]}


def prepare_harnesses(auto_clone: bool = False, mock_test: bool = False) -> Dict[str, Any]:
    statuses = {}
    with harness_repo_policy(auto_clone=auto_clone and not mock_test):
        for name in HARNESS_NAMES:
            try:
                ready = ensure_harness_repo(name, verify=not mock_test,
                                            update=auto_clone and not mock_test)
                with harness_repo_policy(auto_clone=False):
                    prompt = get_harness_system_prompt(name, use_cache=False)
                instance = get_harness(name)
                statuses[name] = {
                    "ok": bool(ready and prompt and instance),
                    "repo_ready": ready, "prompt_chars": len(prompt),
                    "initialized": instance is not None,
                    "class": type(instance).__name__ if instance else None,
                }
                if not ready:
                    statuses[name]["error"] = (
                        "Pinned harness repository is missing or has a different revision; run setup=True."
                    )
            except Exception as exc:
                statuses[name] = {"ok": False, "repo_ready": False, "error": str(exc)}
    return statuses


def download_target_model(model: str, HF_TOKEN: Optional[str] = None,
                          model_cache_dir: Optional[str] = None,
                          model_revision: Optional[str] = None) -> Dict[str, Any]:
    if Path(model).is_dir():
        path = str(Path(model).resolve())
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=model, token=HF_TOKEN, cache_dir=model_cache_dir, revision=model_revision,
            ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot"],
        )
    if not (Path(path) / "config.json").is_file():
        raise RuntimeError(f"Model config.json is missing in {path}")
    if not any(Path(path).glob("*.safetensors")) and not any(Path(path).glob("*.bin")):
        raise RuntimeError(f"No model weights downloaded in {path}")
    return {"ok": True, "model": model, "path": path, "revision": model_revision}


def provision_model_b(B_model_port: int = 11343, mock_test: bool = False,
                      auto_unload_B_model: bool = True, B_model_keep_alive: str = "5m",
                      **options) -> Dict[str, Any]:
    judge = ModelBJudge(model_b_config(
        B_model_port, mock_test=mock_test, auto_unload=auto_unload_B_model,
        keep_alive=B_model_keep_alive, **options,
    ))
    if not judge.ensure_ready():
        raise RuntimeError(judge.last_error or f"Ollama judge is unavailable at {judge.url}")
    fallback = judge.config.fallback_model_name
    if fallback and not judge._ensure_model(fallback):
        raise RuntimeError(f"Ollama fallback model is unavailable: {fallback}")
    # Provisioning downloads both judges; scoring owns their later GPU loading.
    return {"ok": True, "url": judge.url, "model_name": judge.config.model_name,
            "active_model_name": judge.active_model_name, "fallback_model_name": fallback,
            "fallback_ready": bool(fallback), "last_error": judge.last_error}


def setup_environment(
    model: str = DEFAULT_GRPO_MODEL, *, b_model: bool = True, B_model_port: int = 11343,
    HF_TOKEN: Optional[str] = None, load_in_4bit: bool = False, load_in_8bit: bool = False,
    mock_test: bool = False, **options,
) -> Dict[str, Any]:
    """Install/patch vLLM, prepare four harnesses, cache the target, and provision Ollama.

    Mock mode skips installations, downloads, and server operations. Real setup reports
    whether already-imported packages changed so training can wait for a kernel restart.
    """
    environment = configure_training_environment()
    names = ("self_update", "platform", "dependencies", "vllm", "harnesses", "target_model", "b_model")
    if mock_test:
        return {
            "ok": True, "mode": "mock", "environment_validated": False,
            "restart_kernel_required": False, "environment": environment,
            "steps": {name: {"ok": True, "skipped": "mock_test=True"} for name in names},
        }
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose either load_in_4bit or load_in_8bit.")
    before = loaded_package_versions()
    steps = {}

    def run(name, operation):
        print(f"[Sword Setup] {name}...", flush=True)
        try:
            result = operation()
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        steps[name] = result
        print(f"[Sword Setup] {name}: {'PASS' if result.get('ok') else 'FAIL'}", flush=True)
        if result.get("error"):
            print(f"    {result['error']}", flush=True)
        return bool(result.get("ok"))

    import torch
    run("self_update", self_update)
    steps["platform"] = {
        "ok": platform.system() == "Linux" and torch.cuda.is_available(),
        "system": platform.system(), "cuda_available": torch.cuda.is_available(),
    }
    if not steps["platform"]["ok"]:
        steps["platform"]["error"] = "vLLM setup requires Linux (including WSL2) and working PyTorch CUDA."
    ready = steps["platform"]["ok"]
    if ready:
        ready = run("dependencies", lambda: provision_dependencies(load_in_4bit, load_in_8bit, model))
    if ready:
        ready = run("vllm", lambda: vllm_patcher(
            version=options.get("vllm_version", "vllm==0.23.0"), verbose=True,
            sync_project=bool(options.get("setup_sync_project", False)),
        ))
    if ready:
        statuses = prepare_harnesses(auto_clone=True)
        steps["harnesses"] = {"ok": all(s["ok"] for s in statuses.values()), "statuses": statuses}
        # Independent model preparation still proceeds when one harness is unavailable.
        ready = run("target_model", lambda: download_target_model(
            model, HF_TOKEN, options.get("model_cache_dir"), options.get("model_revision"),
        ))
    if ready and b_model:
        run("b_model", lambda: provision_model_b(B_model_port, **options))
    elif not b_model:
        steps["b_model"] = {"ok": True, "skipped": "b_model=False"}
    for name in names:
        steps.setdefault(name, {"ok": False, "skipped": "An earlier setup step failed"})
    changed = changed_loaded_packages(before)
    restart = bool(changed or steps["vllm"].get("restart_kernel_required"))
    return {
        "ok": all(step["ok"] for step in steps.values()),
        "mode": "real", "environment_validated": False, "environment": environment,
        "restart_kernel_required": restart, "changed_loaded_packages": changed,
        "steps": steps,
    }
