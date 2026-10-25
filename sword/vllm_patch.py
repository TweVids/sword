import os
import ast
import sys
import re
import subprocess
import importlib
import importlib.util
import inspect
import tempfile
import shutil
from importlib import metadata

from .rotary_fallback import apply_rotary_fallback, homogeneous_head_dim
from pathlib import Path
from typing import Dict, Any, List



def write_python_patch(path: Path, source: str):
    """Validate syntax and atomically replace a patched Python file."""
    compile(source, str(path), "exec")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(source)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def patch_rotary_source(source: str) -> str:
    marker = "# sword-rotary-fallback-v2"
    if marker in source:
        compile(source, "<vllm rotary patch>", "exec")
        return source
    old = re.search(r"(?m)^([ \t]*)# marimo-rtx6000-rotary-fallback[ \t]*$", source)
    if old:
        indent = old.group(1)
        start = old.start()
        marker_line = source[:old.start()].count("\n") + 1
        tree = ast.parse(source)
        legacy_try = next((node for node in ast.walk(tree)
                           if isinstance(node, ast.Try) and node.lineno == marker_line + 1), None)
        if legacy_try is None:
            raise ValueError("Unsupported legacy rotary patch layout")
        end = sum(len(line) for line in source.splitlines(keepends=True)[:legacy_try.end_lineno])
    else:
        match = re.search(r"(?m)^([ \t]*)from flash_attn\.ops\.triton\.rotary import apply_rotary[ \t]*$", source)
        if not match:
            raise ValueError("Unsupported vLLM rotary source: flash-attn import not found")
        indent, start, end = match.group(1), match.start(), match.end()
    function = inspect.getsource(apply_rotary_fallback).replace("def apply_rotary_fallback(", "def apply_rotary(", 1).rstrip()
    lines = [marker, "try:", "    from flash_attn.ops.triton.rotary import apply_rotary",
             "except (ImportError, ModuleNotFoundError):"]
    lines += ["    " + line if line else "" for line in function.splitlines()]
    replacement = "\n".join(indent + line if line else line for line in lines) + "\n"
    patched = source[:start] + replacement + source[end:]
    compile(patched, "<vllm rotary patch>", "exec")
    return patched


def patch_head_dim_source(source: str) -> str:
    if "# sword-homogeneous-head-dim" in source:
        compile(source, "<vllm head-dimension patch>", "exec")
        return source
    target = 'getattr(self.hf_text_config, "head_dim", 0)'
    if target not in source:
        return source
    source = source.replace(target, "homogeneous_head_dim(self.hf_text_config)", 1)
    source += "\n# sword-homogeneous-head-dim\n" + inspect.getsource(homogeneous_head_dim)
    compile(source, "<vllm head-dimension patch>", "exec")
    return source


def loaded_package_versions() -> Dict[str, str]:
    """Record versions already imported by a notebook before package installation.

    Reads from importlib.metadata so that the snapshot uses the SAME source as
    `changed_loaded_packages()`'s comparison. Previously this used
    `mod.__version__`, which can differ from the metadata version (e.g.
    "2.9.0+cu130" vs "2.9.0"), causing `changed_loaded_packages` to report a
    false positive on every run and demand a kernel restart in an infinite loop.
    """
    names = ("torch", "transformers", "numpy", "vllm", "peft", "accelerate",
             "huggingface_hub", "tokenizers", "triton", "safetensors")
    snapshot: Dict[str, str] = {}
    for name in names:
        if (mod := sys.modules.get(name)) is None:
            continue
        try:
            snapshot[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # Package metadata missing (e.g. vendored/embedded install):
            # fall back to the module attribute so a real change still trips.
            version = getattr(mod, "__version__", None)
            if isinstance(version, str):
                snapshot[name] = f"__module__:{version}"
    return snapshot


def changed_loaded_packages(before: Dict[str, str]) -> List[str]:
    changed = []
    for name, version in before.items():
        if version.startswith("__module__:"):
            try:
                mod = sys.modules.get(name)
                current = getattr(mod, "__version__", "") if mod is not None else ""
            except Exception:
                current = ""
            if current != version.split(":", 1)[1]:
                changed.append(name)
            continue
        try:
            if metadata.version(name) != version:
                changed.append(name)
        except metadata.PackageNotFoundError:
            changed.append(name)
    return changed


def inspect_vllm_installation() -> Dict[str, Any]:
    """Validate installed patch files without installing or modifying packages."""
    steps = []
    try:
        spec = importlib.util.find_spec("vllm")
        if not spec or not spec.origin:
            raise RuntimeError("vLLM is not installed; run sword.train(setup=True, ...).")
        root = Path(spec.origin).parent
        for relative, marker in (
            ("model_executor/layers/rotary_embedding/common.py", "# sword-rotary-fallback-v2"),
            ("transformers_utils/model_arch_config_convertor.py", "# sword-homogeneous-head-dim"),
        ):
            path = root / relative
            if not path.exists():
                if relative.startswith("transformers_utils/"):
                    continue
                raise RuntimeError(f"Unsupported vLLM rotary layout: {path}")
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            # A convertor without the old scalar head_dim lookup needs no patch.
            patched = marker in source or (
                relative.startswith("transformers_utils/")
                and 'getattr(self.hf_text_config, "head_dim", 0)' not in source
                and "# marimo-safe-heterogeneous-head-dim" not in source
            )
            steps.append({"patch_file": str(path), "patched": patched})
            if not patched:
                raise RuntimeError(f"vLLM patch missing in {path}; run setup=True.")
        return {"ok": True, "restart_kernel_required": False, "steps": steps}
    except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
        return {"ok": False, "error": str(exc), "steps": steps}


def bootstrap_environment(
    version: str = "vllm==0.23.0", *, install: bool = True, sync_project: bool = False,
) -> Dict[str, Any]:
    """
    Bootstraps the local python environment:
    - Optionally syncs a uv project when sync_project=True
    - Installs missing/mismatched vllm via uv or this interpreter's pip
    - Patches vllm rotary embedding common.py with pure-PyTorch fallback
    """
    results = []
    before = loaded_package_versions()
    patched_loaded_modules = []
    cwd = Path.cwd()
    if install and sync_project and ((cwd / "pyproject.toml").exists() or (cwd / "uv.lock").exists()):
        try:
            run = subprocess.run(["uv", "sync", "--exact", "--active"], text=True, capture_output=True)
        except OSError as exc:
            return {"ok": False, "steps": results, "error": str(exc)}
        results.append({
            "command": "uv sync --exact --active",
            "returncode": run.returncode,
            "tail": (run.stdout + run.stderr)[-4000:],
        })
        if run.returncode:
            return {"ok": False, "steps": results}
    else:
        results.append({
            "command": "uv sync --exact --active",
            "skipped": "Project sync is opt-in; existing runtime packages are retained",
        })
    pkg = "vllm" if version in ("latest", "vllm") else version
    try:
        installed = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        installed = None
    needs_install = installed is None or version in ("latest", "vllm") or pkg != f"vllm=={installed}"
    try:
        if install and needs_install:
            command = (["uv", "pip", "install", "--python", sys.executable]
                       if shutil.which("uv") else [sys.executable, "-m", "pip", "install"])
            if version in ("latest", "vllm"):
                command.append("--upgrade")
            run = subprocess.run([*command, pkg], text=True, capture_output=True)
            results.append({
                "command": " ".join([*command, pkg]), "returncode": run.returncode,
                "tail": (run.stdout + run.stderr)[-4000:],
            })
            if run.returncode:
                return {"ok": False, "steps": results}
        else:
            results.append({"command": f"install {pkg}", "skipped": (
                f"Already installed: vllm=={installed}" if installed else "Installation disabled"
            )})
    except OSError as exc:
        return {"ok": False, "steps": results, "error": str(exc)}
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("vllm")
    if not spec or not spec.origin:
        return {"ok": False, "steps": results, "error": "vLLM installed but not importable"}
    common = Path(spec.origin).parent / "model_executor/layers/rotary_embedding/common.py"
    if not common.exists():
        return {"ok": False, "steps": results, "error": f"Unsupported vLLM rotary layout: {common}"}
    try:
        source = common.read_text(encoding="utf-8")
        patched = patch_rotary_source(source)
        if patched != source:
            write_python_patch(common, patched)
            if "vllm.model_executor.layers.rotary_embedding.common" in sys.modules:
                patched_loaded_modules.append("vllm.model_executor.layers.rotary_embedding.common")
        results.append({"patch_file": str(common), "patched": "# sword-rotary-fallback-v2" in patched})

        convertor = Path(spec.origin).parent / "transformers_utils/model_arch_config_convertor.py"
        if convertor.exists():
            source = convertor.read_text(encoding="utf-8")
            if "# marimo-safe-heterogeneous-head-dim" in source:
                raise ValueError("Legacy head-dimension patch detected; restore this vLLM file before patching")
            patched = patch_head_dim_source(source)
            if patched != source:
                write_python_patch(convertor, patched)
                if "vllm.transformers_utils.model_arch_config_convertor" in sys.modules:
                    patched_loaded_modules.append("vllm.transformers_utils.model_arch_config_convertor")
            results.append({"patch_file": str(convertor), "patched": "# sword-homogeneous-head-dim" in patched})
    except (OSError, ValueError, SyntaxError) as exc:
        return {"ok": False, "steps": results, "error": str(exc)}

    changed = changed_loaded_packages(before) + patched_loaded_modules
    return {"ok": True, "restart_kernel_required": bool(changed),
            "changed_loaded_packages": changed, "steps": results}


def vllm_patcher(
    verbose: bool = True, version: str = "vllm==0.23.0", *,
    install: bool = True, sync_project: bool = False,
) -> Dict[str, Any]:
    """
    Executes bootstrap_environment() and prints detailed step execution logs.
    Can be imported and called via:
        from sword import vllm_patcher
        vllm_patcher()             # Installs and patches vllm==0.23.0
        vllm_patcher(version="latest")  # Upgrades to latest vLLM release and patches it
    """
    if verbose:
        print(f"[Sword] Starting vLLM patcher and environment bootstrap ({version})...")
    outcome = bootstrap_environment(version=version, install=install, sync_project=sync_project)

    if verbose:
        print(f"[Sword] Bootstrap overall status: ok={outcome.get('ok')}")
        steps = outcome.get("steps", [])
        for idx, step in enumerate(steps, 1):
            if "command" in step:
                status = f"code={step.get('returncode')}" if "returncode" in step else f"skipped ({step.get('skipped')})"
                print(f"[Sword] Step {idx}: command '{step['command']}' -> {status}")
                tail = step.get("tail", "").strip()
                if tail:
                    tail_lines = tail.splitlines()[-5:]
                    for line in tail_lines:
                        print(f"    | {line}")
            elif "patch_file" in step:
                print(f"[Sword] Step {idx}: patch file '{step['patch_file']}' -> patched={step.get('patched')}")
        if outcome.get("error"):
            print(f"[Sword] Error encountered: {outcome.get('error')}")
        if outcome.get("restart_kernel_required"):
            print("[Sword] Environment updated successfully. Note: Kernel restart may be required if vLLM was previously loaded.")

    return outcome


vllm_patcher.bootstrap_environment = bootstrap_environment

