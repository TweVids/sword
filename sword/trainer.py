import os
import gc
import glob
import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    AutoTokenizer,
)

try:
    from datasets import load_dataset, Dataset
except ImportError:
    load_dataset = None
    Dataset = None

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    HfApi = None
    snapshot_download = None

from .patcher import patch_qwen


# =========================================================
# ⚙️  Blackwell (SM100) & High-Performance PyTorch Config
# =========================================================
def setup_blackwell_environment():
    """
    Optimizes PyTorch runtime for NVIDIA Blackwell (SM100) / Hopper / Ada GPUs:
    - Enables expandable segments to prevent CUDA memory fragmentation
    - Configures high precision BF16 / TF32 matrix multiplication
    - Enables flex attention / native SDPA kernels
    """
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

        if hasattr(torch.backends.cuda, "enable_flex_attention"):
            try:
                torch.backends.cuda.enable_flex_attention(True)
                print("[Sword] Flex Attention enabled for Blackwell SM100.")
            except Exception:
                pass
        print(f"[Sword] GPU Acceleration configured for: {torch.cuda.get_device_name(0)}")


# =========================================================
# 📥  GOOGLE DRIVE DATASET DOWNLOADER
# =========================================================
def _extract_drive_file_id(drive_url: str) -> str:
    """Extracts the unique file ID from any standard Google Drive sharing link."""
    match = re.search(r"(?:/d/|id=|open\?id=)([A-Za-z0-9_-]{20,})", drive_url)
    if not match:
        raise ValueError(f"Cannot parse Google Drive file ID from URL: {drive_url}")
    return match.group(1)


def _download_from_drive_requests_fallback(file_id: str, dest_path: str) -> str:
    """Fallback chunked streaming downloader using requests with confirmation token support."""
    import requests

    session = requests.Session()
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    print(f"📥 [requests fallback] Downloading Drive file ID {file_id} -> {dest_path}")
    response = session.get(download_url, stream=True)

    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break
    if confirm_token is None and b"confirm=" in response.content[:4096]:
        m = re.search(rb"confirm=([0-9A-Za-z_-]+)", response.content[:4096])
        if m:
            confirm_token = m.group(1).decode()

    if confirm_token:
        response = session.get(f"{download_url}&confirm={confirm_token}", stream=True)

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32 * 1024 * 1024):  # 32 MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:5.1f}%  ({downloaded/1e9:.2f} / {total/1e9:.2f} GB)", end="", flush=True)
    print()
    return dest_path


def download_from_drive(drive_url: str, dest_path: str, force: bool = False) -> str:
    """
    Downloads a dataset or file from Google Drive.
    Supports gdown with automatic requests-based fallback.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)

    if os.path.exists(dest_path) and not force:
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"✅ Dataset already present ({size_mb:.1f} MB) -> {dest_path} (skipping download)")
        return dest_path

    file_id = _extract_drive_file_id(drive_url)

    try:
        import gdown
        print(f"📥 Downloading Drive file {file_id} -> {dest_path} (via gdown)...")
        gdown.download(id=file_id, output=dest_path, quiet=False)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise RuntimeError("gdown reported success but output file is missing or empty")
    except ImportError:
        print("⚠️  gdown not installed (pip install gdown) — using streaming requests fallback.")
        _download_from_drive_requests_fallback(file_id, dest_path)
    except Exception as e:
        print(f"⚠️  gdown download encountered error ({e}) — retrying with requests fallback...")
        _download_from_drive_requests_fallback(file_id, dest_path)

    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        raise RuntimeError(f"Download failed — {dest_path} is missing or empty.")

    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"✅ Downloaded {size_mb:.1f} MB -> {dest_path}")
    return dest_path


# =========================================================
# 📥 CHECKPOINT DOWNLOADER & VALIDATOR
# =========================================================
def is_valid_checkpoint(ckpt_dir: str) -> bool:
    """Checks if a checkpoint directory has valid training state and weights."""
    if not os.path.isdir(ckpt_dir):
        return False
    if not os.path.exists(os.path.join(ckpt_dir, "trainer_state.json")):
        return False
    has_weights = (
        glob.glob(os.path.join(ckpt_dir, "adapter_model*"))
        or glob.glob(os.path.join(ckpt_dir, "*.safetensors"))
        or glob.glob(os.path.join(ckpt_dir, "pytorch_model*"))
        or os.path.exists(os.path.join(ckpt_dir, "adapter_config.json"))
    )
    return bool(has_weights)


def download_hf_checkpoint(
    hf_repo_id: str,
    checkpoint_name: str,
    local_output_dir: str,
    hf_token: Optional[str] = None,
) -> str:
    """
    Downloads a checkpoint directory from a HuggingFace repository and validates it.
    """
    if snapshot_download is None:
        raise ImportError("huggingface_hub is required for downloading checkpoints (pip install huggingface_hub).")

    ckpt_dir = os.path.join(local_output_dir, checkpoint_name)

    if is_valid_checkpoint(ckpt_dir):
        print(f"✅ Checkpoint already present and verified: {ckpt_dir}")
        return ckpt_dir

    print(f"📥 Downloading {checkpoint_name} from {hf_repo_id} -> {ckpt_dir} ...")
    os.makedirs(local_output_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=[f"{checkpoint_name}/*", f"{checkpoint_name}/**"],
            local_dir=local_output_dir,
            token=hf_token,
            endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        )
    except Exception as e:
        raise RuntimeError(f"HuggingFace checkpoint download failed: {e}")

    if not is_valid_checkpoint(ckpt_dir):
        possible_loose = glob.glob(os.path.join(local_output_dir, f"{checkpoint_name}*"))
        if possible_loose:
            print(f"   Found candidate files: {possible_loose}")
        raise RuntimeError(
            f"❌ Downloaded checkpoint at {ckpt_dir} is missing required files (trainer_state.json or adapter weights). "
            f"Check repo {hf_repo_id} structure."
        )

    print(f"✅ Checkpoint successfully verified and ready: {ckpt_dir}")
    return ckpt_dir


# =========================================================
# 📥 DATASET LOADER (OFFLINE PRE-FORMATTED JSONL)
# =========================================================
def load_offline_dataset(dataset_path: str, seed: int = 42, test_size: int = 700):
    """Loads a pre-formatted json or jsonl dataset and splits into train/eval sets."""
    if load_dataset is None:
        raise ImportError("datasets library is required to load datasets (pip install datasets).")

    print(f"📖 Loading pre-formatted offline dataset from: {dataset_path}")
    raw_ds = load_dataset("json", data_files=dataset_path, split="train")
    split_ds = raw_ds.train_test_split(test_size=test_size, seed=seed)

    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]

    print(f"✅ Loaded: {len(train_ds):,} Train samples | {len(eval_ds):,} Eval samples")
    return train_ds, eval_ds


# =========================================================
# 💾 CHECKPOINT CALLBACK WITH HF HUB SYNC & PRUNING
# =========================================================
class FullCheckpointCallback(TrainerCallback):
    """
    Custom TrainerCallback that:
    1. Saves model LoRA weights, tokenizer, trainer_state, optimizer, scheduler, and args
    2. Uploads checkpoint directory to HuggingFace Hub asynchronously / safely
    3. Prunes old local checkpoints to respect save_limit
    """

    def __init__(
        self,
        tokenizer: Any,
        output_dir: str,
        save_steps: int,
        save_limit: int = 2,
        hf_repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.save_limit = save_limit
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.trainer = None
        self.hf_api = None

        if HfApi is not None and self.hf_repo_id:
            try:
                self.hf_api = HfApi(
                    endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                    token=self.hf_token,
                )
            except Exception as e:
                print(f"[Sword] Warning: HfApi initialization failed ({e}). Checkpoints will only save locally.")

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if state.global_step == 0 or state.global_step % self.save_steps != 0:
            return control
        if not state.is_world_process_zero:
            return control

        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        step = state.global_step
        print(f"\n💾 [Sword] Saving full checkpoint at step {step} -> {ckpt_dir}")

        # Save model and tokenizer
        try:
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(ckpt_dir)
            if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(ckpt_dir)
            print("  ✅ LoRA weights & tokenizer saved.")
        except Exception as e:
            print(f"  ⚠️  LoRA/tokenizer save failed: {e}")

        # Save trainer state JSON
        try:
            state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
            print("  ✅ trainer_state.json saved.")
        except Exception as e:
            print(f"  ⚠️  trainer_state save failed: {e}")

        # Save optimizer state
        if self.trainer is not None and getattr(self.trainer, "optimizer", None) is not None:
            try:
                torch.save(self.trainer.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                print("  ✅ optimizer.pt saved.")
            except Exception as e:
                print(f"  ⚠️  Optimizer save failed: {e}")

        # Save scheduler state
        if self.trainer is not None and getattr(self.trainer, "lr_scheduler", None) is not None:
            try:
                torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                print("  ✅ scheduler.pt saved.")
            except Exception as e:
                print(f"  ⚠️  Scheduler save failed: {e}")

        # Save training arguments
        try:
            try:
                import trl.trainer.sft_config as _sft_mod
                _orig = getattr(_sft_mod, "SFTConfig", None)
                if _orig is not None:
                    _sft_mod.SFTConfig = args.__class__
                    torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))
                    _sft_mod.SFTConfig = _orig
                else:
                    torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))
            except Exception:
                torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))
            print("  ✅ training_args.bin saved.")
        except Exception as e:
            print(f"  ⚠️  training_args.bin save failed: {e}")

        print(f"✅ Local checkpoint complete: {ckpt_dir}")

        # Upload to HuggingFace Hub if configured
        if self.hf_api is not None and self.hf_repo_id:
            hf_path = f"checkpoint-{step}"
            print(f"☁️  Uploading {ckpt_dir} -> {self.hf_repo_id}/{hf_path} ...")
            try:
                self.hf_api.create_repo(repo_id=self.hf_repo_id, repo_type="model", exist_ok=True, private=True)
                self.hf_api.upload_folder(
                    folder_path=ckpt_dir,
                    repo_id=self.hf_repo_id,
                    path_in_repo=hf_path,
                    repo_type="model",
                    commit_message=f"Sword checkpoint step {step}",
                )
                print(f"  ✅ Uploaded to HF: {self.hf_repo_id}/{hf_path}")
            except Exception as e:
                print(f"  ⚠️  HF upload failed (checkpoint is safely preserved locally): {e}")

        # Prune older local checkpoints to respect save_limit
        checkpoints = sorted(
            glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1,
        )
        while len(checkpoints) > self.save_limit:
            old = checkpoints.pop(0)
            try:
                shutil.rmtree(old)
                print(f"🗑️  Pruned old local checkpoint: {old}")
            except Exception as e:
                print(f"  ⚠️  Could not remove old checkpoint {old}: {e}")

        return control
