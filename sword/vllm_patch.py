import os
import sys
import re
import subprocess
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, Any


def bootstrap_environment() -> Dict[str, Any]:
    """
    Bootstraps the local python environment:
    - Runs uv sync if pyproject.toml or uv.lock exists
    - Installs vllm==0.23.0 via uv pip
    - Patches vllm rotary embedding common.py with pure-PyTorch fallback
    """
    results = []
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists() or (cwd / "uv.lock").exists():
        run = subprocess.run(["uv", "sync", "--exact", "--active"], text=True, capture_output=True)
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
            "skipped": "No pyproject.toml/uv.lock in current directory",
        })
    run = subprocess.run(["uv", "pip", "install", "vllm==0.23.0"], text=True, capture_output=True)
    results.append({
        "command": "uv pip install vllm==0.23.0",
        "returncode": run.returncode,
        "tail": (run.stdout + run.stderr)[-4000:],
    })
    if run.returncode:
        return {"ok": False, "steps": results}
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("vllm")
    if not spec or not spec.origin:
        return {"ok": False, "steps": results, "error": "vLLM installed but not importable"}
    common = Path(spec.origin).parent / "model_executor/layers/rotary_embedding/common.py"
    source = common.read_text()
    needle = "from flash_attn.ops.triton.rotary import apply_rotary"
    marker = "# marimo-rtx6000-rotary-fallback"
    if needle in source and marker not in source:
        match = re.search(r"(?m)^([ \t]*)from flash_attn\.ops\.triton\.rotary import apply_rotary\s*$", source)
        if not match:
            return {"ok": False, "steps": results, "error": f"Could not locate import line in {common}"}
        indent = match.group(1)
        body = [
            marker,
            "try:",
            "    from flash_attn.ops.triton.rotary import apply_rotary",
            "except (ImportError, ModuleNotFoundError):",
            "    import torch as _torch",
            "    def apply_rotary(x, cos, sin, seqlen_offsets=0, cu_seqlens=None,",
            "                     max_seqlen=None, interleaved=False, inplace=False, conjugate=False):",
            "        d = x.shape[-1]",
            "        x1, x2 = x[..., :d//2], x[..., d//2:]",
            "        c = cos[..., :d//2] if cos.shape[-1] >= d//2 else cos",
            "        s = sin[..., :d//2] if sin.shape[-1] >= d//2 else sin",
            "        if conjugate: s = -s",
            "        out = _torch.cat([x1*c - x2*s, x1*s + x2*c], dim=-1).to(x.dtype)",
            "        if inplace: x.copy_(out); return x",
            "        return out",
        ]
        replacement = "\n".join(indent + line if line else line for line in body)
        common.write_text(source[:match.start()] + replacement + source[match.end():])
    results.append({"patch_file": str(common), "patched": marker in common.read_text()})
    return {"ok": True, "restart_kernel_required": True, "steps": results}


def vllm_patcher(verbose: bool = True) -> Dict[str, Any]:
    """
    Executes bootstrap_environment() and prints detailed step execution logs.
    Can be imported and called via:
        from sword import vllm_patcher
        vllm_patcher()
    """
    if verbose:
        print("[Sword] Starting vLLM patcher and environment bootstrap...")
    outcome = bootstrap_environment()

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

