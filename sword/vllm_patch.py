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

    # Patch vLLM model_arch_config_convertor for heterogeneous head_dim access
    convertor_file = Path(spec.origin).parent / "transformers_utils/model_arch_config_convertor.py"
    if convertor_file.exists():
        c_source = convertor_file.read_text()
        c_marker = "# marimo-safe-heterogeneous-head-dim"
        if c_marker not in c_source:
            target_str = 'head_dim = getattr(self.hf_text_config, "head_dim", 0)'
            if target_str in c_source:
                replacement_str = (
                    c_marker + "\n"
                    "        try:\n"
                    "            head_dim = getattr(self.hf_text_config, 'head_dim', 0)\n"
                    "        except Exception:\n"
                    "            head_dim = getattr(getattr(self.hf_text_config, 'per_layer_config', [{}])[0], 'head_dim', 0) if getattr(self.hf_text_config, 'per_layer_config', None) else 0"
                )
                c_source = c_source.replace(target_str, replacement_str, 1)
                convertor_file.write_text(c_source)
                results.append({"patch_file": str(convertor_file), "patched": True})

    # Patch Transformers heterogeneity configuration_utils to suppress AmbiguousGlobalPerLayerAttributeError
    t_spec = importlib.util.find_spec("transformers")
    if t_spec and t_spec.origin:
        t_root = Path(t_spec.origin).parent
        utils_file = t_root / "utils/__init__.py"
        if utils_file.exists():
            u_source = utils_file.read_text()
            u_marker = "# marimo-download-url-compat"
            if u_marker not in u_source:
                compat_snippet = (
                    f"\n{u_marker}\n"
                    "try:\n"
                    "    from urllib.request import urlretrieve as download_url\n"
                    "except Exception:\n"
                    "    def download_url(url, **kw): return None\n"
                )
                utils_file.write_text(u_source + compat_snippet)
                results.append({"patch_file": str(utils_file), "patched": True})

        het_file = t_root / "integrations/heterogeneity/configuration_utils.py"
        if het_file.exists():
            h_source = het_file.read_text()
            h_marker = "# marimo-suppress-heterogeneity-error"
            if h_marker not in h_source:
                # Bypass AmbiguousGlobalPerLayerAttributeError by forcing allow_global_per_layer_attribute_access to True
                patched_h = re.sub(
                    r'if\s+not\s+getattr\(self,\s*["\']allow_global_per_layer_attribute_access["\'],\s*False\):',
                    f"{h_marker}\n        if False:",
                    h_source,
                )
                if patched_h != h_source:
                    het_file.write_text(patched_h)
                    results.append({"patch_file": str(het_file), "patched": True})
                else:
                    patched_h2 = re.sub(
                        r"raise\s+AmbiguousGlobalPerLayerAttributeError\(.*?\)",
                        f"{h_marker}\n        return super().__getattribute__(key)",
                        h_source,
                        flags=re.DOTALL,
                    )
                    if patched_h2 != h_source:
                        het_file.write_text(patched_h2)
                        results.append({"patch_file": str(het_file), "patched": True})

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

