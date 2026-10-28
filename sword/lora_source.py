"""Resolve a fixed LoRA checkpoint and materialize its weights as a new GRPO base."""
import gc
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Union


SOURCE_MANIFEST = "sword_lora_source.json"
_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
_ASSETS = ("adapter_config.json", SOURCE_MANIFEST, "chat_template.jinja", "tokenizer.json",
           "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
           "vocab.json", "merges.txt", "added_tokens.json")


def normalize_lora_checkpoint(checkpoint: Optional[Union[int, str]]) -> str:
    if checkpoint is None or checkpoint == "":
        return ""
    if isinstance(checkpoint, bool) or not isinstance(checkpoint, (int, str)):
        raise ValueError("train_from_lora_checkpoint must be a step, checkpoint folder, latest, or root")
    value = str(checkpoint).strip()
    if value.lower() in ("", "root"):
        return ""
    if value.lower() in ("latest", "last"):
        return "latest"
    if value.isdigit():
        value = "checkpoint-" + value
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or ":" in value or any(part in (".", "..") for part in value.split("/")):
        raise ValueError("train_from_lora_checkpoint must be a relative checkpoint folder")
    return value.rstrip("/")


def _select_checkpoint(files: List[str], checkpoint) -> str:
    folders = sorted({str(PurePosixPath(name).parent) for name in files
                      if PurePosixPath(name).name in _WEIGHTS})
    folders = ["" if name == "." else name for name in folders]
    want = normalize_lora_checkpoint(checkpoint)
    if want == "latest":
        stepped = [(int(match.group(1)), folder) for folder in folders
                   if (match := re.fullmatch(r"checkpoint-(\d+)", PurePosixPath(folder).name))]
        if stepped:
            step = max(item[0] for item in stepped)
            matches = [folder for number, folder in stepped if number == step]
        else:
            matches = [""] if "" in folders else []
    else:
        matches = [folder for folder in folders if folder == want or (want and folder.endswith("/" + want))]
    if len(matches) > 1:
        raise ValueError(f"Checkpoint {want!r} is ambiguous; use the full folder path: {matches}")
    if not matches:
        raise FileNotFoundError(f"LoRA checkpoint {want or 'root'!r} not found. Available adapter folders: {folders}")
    return matches[0]


def _json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _digest_files(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass
class LoraTrainingSource:
    base_model: str
    adapters: List[str]
    manifest: Dict[str, Any]
    tokenizer_dir: Optional[str] = None
    chat_template: Optional[str] = None


def _download_adapter(repo: str, checkpoint, *, token=None, revision=None,
                      cache_dir=None, source_cache_dir=None, local_files_only=False):
    local = Path(repo).expanduser()
    if local.is_dir():
        root = local.resolve()
        files = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
        pinned_revision = None
        repo = str(root)
    else:
        from huggingface_hub import HfApi, snapshot_download
        if local_files_only:
            root = Path(snapshot_download(repo, revision=revision, cache_dir=cache_dir,
                                          token=token, local_files_only=True))
            pinned_revision = root.name
            files = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
        else:
            api = HfApi(token=token)
            pinned_revision = api.model_info(repo, revision=revision).sha
            files = api.list_repo_files(repo, revision=pinned_revision)
        folder = _select_checkpoint(files, checkpoint)
        prefix = folder + "/" if folder else ""
        patterns = list(_ASSETS) + [prefix + name for name in (*_ASSETS, *_WEIGHTS)]
        root = Path(snapshot_download(repo, revision=pinned_revision, allow_patterns=patterns,
                                     cache_dir=cache_dir, token=token, local_files_only=local_files_only))

    folder = _select_checkpoint(files, checkpoint)
    adapter = root / folder
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        root_config = root / "adapter_config.json"
        if not root_config.is_file():
            raise FileNotFoundError(f"No adapter_config.json in {repo!r}/{folder or 'root'} or its repo root")
        # Borrow the root config without changing the user's repo or shared HF snapshots.
        key = hashlib.sha256(f"{repo}:{pinned_revision}:{folder}".encode()).hexdigest()[:24]
        staging = Path(source_cache_dir or "checkpoints/lora_sources") / key
        staging.mkdir(parents=True, exist_ok=True)
        for name in (*_ASSETS, *_WEIGHTS):
            source = adapter / name
            if not source.is_file() and name in _ASSETS:
                source = root / name
            if source.is_file():
                shutil.copy2(source, staging / name)
        adapter = staging.resolve()
        config_path = adapter / "adapter_config.json"
    weights = [adapter / name for name in _WEIGHTS if (adapter / name).is_file()]
    if not weights:
        raise FileNotFoundError(f"Selected checkpoint has no downloaded adapter weights: {adapter}")
    descriptor = {"repo": repo, "checkpoint": folder or "root", "revision": pinned_revision,
                  "sha256": _digest_files([config_path] + weights)}
    assets = [adapter, root]
    lineage_path = next((directory / SOURCE_MANIFEST for directory in assets
                         if (directory / SOURCE_MANIFEST).is_file()), None)
    template_path = next((directory / "chat_template.jinja" for directory in assets
                          if (directory / "chat_template.jinja").is_file()), None)
    tokenizer_dir = next((str(directory) for directory in assets
                          if (directory / "tokenizer_config.json").is_file() and
                          any((directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))), None)
    return str(adapter), descriptor, _json(config_path), lineage_path, template_path, tokenizer_dir


def resolve_lora_training_source(repo: str, checkpoint=None, *, base_model=None, token=None,
                                 revision=None, base_revision=None, cache_dir=None,
                                 source_cache_dir=None, local_files_only=False) -> LoraTrainingSource:
    """Download only the selected adapter, pin remote revisions, and recover any merged-base lineage."""
    token = token or os.environ.get("HF_TOKEN")
    options = dict(token=token, cache_dir=cache_dir, source_cache_dir=source_cache_dir,
                   local_files_only=local_files_only)
    adapter, descriptor, config, lineage_path, template_path, tokenizer_dir = _download_adapter(
        repo, checkpoint, revision=revision, **options,
    )
    lineage = _json(lineage_path) if lineage_path else {}
    if lineage and (lineage.get("format_version") != 1 or not isinstance(lineage.get("adapters"), list)):
        raise ValueError(f"Invalid merged-base source manifest in {lineage_path}")
    inferred_base = lineage.get("base_model") or config.get("base_model_name_or_path")
    if base_model and inferred_base and str(base_model) != str(inferred_base):
        raise ValueError(f"model={base_model!r} conflicts with the LoRA base model {inferred_base!r}")
    inferred_base = inferred_base or base_model
    if not inferred_base:
        raise ValueError("LoRA adapter has no base_model_name_or_path; provide model=<base model>")
    pinned_base_revision = lineage.get("base_revision", base_revision)
    if not Path(inferred_base).expanduser().is_dir() and not pinned_base_revision:
        from huggingface_hub import HfApi, snapshot_download
        if local_files_only:
            pinned_base_revision = Path(snapshot_download(
                inferred_base, token=token, cache_dir=cache_dir, local_files_only=True,
            )).name
        else:
            pinned_base_revision = HfApi(token=token).model_info(inferred_base).sha
    adapters, descriptors = [], []
    for parent in lineage.get("adapters", []):
        if not isinstance(parent, dict) or not parent.get("repo"):
            raise ValueError(f"Invalid ancestor adapter in {lineage_path}")
        path, resolved, _, _, parent_template, parent_tokenizer = _download_adapter(
            parent["repo"], parent.get("checkpoint"), revision=parent.get("revision"), **options,
        )
        if parent.get("sha256") != resolved["sha256"]:
            raise ValueError(f"Source LoRA weights changed for {parent['repo']!r}/{parent.get('checkpoint')}")
        adapters.append(path)
        descriptors.append(resolved)
        template_path = template_path or parent_template
        tokenizer_dir = tokenizer_dir or parent_tokenizer
    adapters.append(adapter)
    descriptors.append(descriptor)
    manifest = {"format_version": 1, "base_model": str(inferred_base),
                "base_revision": pinned_base_revision, "adapters": descriptors}
    if lineage.get("merge_dtype"):
        manifest["merge_dtype"] = lineage["merge_dtype"]
    template = template_path.read_text(encoding="utf-8") if template_path else None
    print(f"[LoRA Source] repo={descriptor['repo']} checkpoint={descriptor['checkpoint']} "
          f"revision={descriptor['revision'] or 'local'} base={inferred_base}", flush=True)
    return LoraTrainingSource(str(inferred_base), adapters, manifest, tokenizer_dir, template)


def prepare_merged_lora_base(source: LoraTrainingSource, output_dir: str, *, token=None,
                             cache_dir=None, local_files_only=False) -> str:
    """Merge without k-bit quantization, save a reusable full base, then release merge memory."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = source.manifest.get("merge_dtype") or ("bfloat16" if torch.cuda.is_available() else "float32")
    if dtype_name not in ("bfloat16", "float32", "float16"):
        raise ValueError(f"Unsupported source merge dtype: {dtype_name}")
    manifest = {**source.manifest, "merge_dtype": dtype_name}
    # Tokenizers and the exact training template are part of this cached base too.
    asset_hash = hashlib.sha256((source.chat_template or "").encode()).hexdigest()
    if source.tokenizer_dir:
        asset_hash += _digest_files([path for path in Path(source.tokenizer_dir).iterdir()
                                    if path.is_file() and path.name in _ASSETS and path.name != SOURCE_MANIFEST])
    key = hashlib.sha256((json.dumps(manifest, sort_keys=True) + asset_hash).encode()).hexdigest()[:24]
    cache_root = Path(output_dir).expanduser().resolve()
    target = cache_root / key
    if (target / SOURCE_MANIFEST).is_file() and (target / "config.json").is_file() and (
        any(target.glob("*.safetensors")) or any(target.glob("pytorch_model*.bin"))
    ):
        if _json(target / SOURCE_MANIFEST) == manifest:
            print(f"[LoRA Merge] Reusing merged base: {target}", flush=True)
            return str(target)
    if target.exists():
        raise RuntimeError(f"Incomplete merged-base cache at {target}; choose a new merged_base_cache_dir")
    cache_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="merging_", dir=cache_root))
    model = tokenizer = None
    try:
        options = dict(token=token or os.environ.get("HF_TOKEN"), cache_dir=cache_dir,
                       local_files_only=local_files_only, trust_remote_code=True)
        revision = manifest.get("base_revision")
        print(f"[LoRA Merge] Loading unquantized base {source.base_model} on CPU ({dtype_name})", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            source.base_model, revision=revision, torch_dtype=getattr(torch, dtype_name),
            device_map="cpu", low_cpu_mem_usage=True, **options,
        )
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False) or getattr(model, "hf_quantizer", None):
            raise ValueError("Source LoRA merge requires an unquantized base model")
        tokenizer = AutoTokenizer.from_pretrained(
            source.tokenizer_dir or source.base_model,
            revision=None if source.tokenizer_dir else revision, **options,
        )
        if source.chat_template:
            tokenizer.chat_template = source.chat_template
        for adapter_dir in source.adapters:
            print(f"[LoRA Merge] Merging adapter: {adapter_dir}", flush=True)
            model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
            model = model.merge_and_unload(safe_merge=True)
        if any("lora_" in name for name, _ in model.named_parameters()):
            raise RuntimeError("Source LoRA merge left adapter parameters attached")
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        if not (staging / "config.json").is_file() or not (
            any(staging.glob("*.safetensors")) or any(staging.glob("pytorch_model*.bin"))
        ):
            raise RuntimeError("Merged model serialization did not produce a full model")
        if source.chat_template:
            (staging / "chat_template.jinja").write_text(source.chat_template, encoding="utf-8")
        (staging / SOURCE_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(target)
        print(f"[LoRA Merge] Frozen GRPO base saved: {target}", flush=True)
        return str(target)
    except Exception as exc:
        raise RuntimeError("Could not merge the requested source LoRA; GRPO will not use the original base instead") from exc
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Only this invocation's private staging directory can be removed.
        if staging.exists() and staging.resolve().parent == cache_root:
            shutil.rmtree(staging)
