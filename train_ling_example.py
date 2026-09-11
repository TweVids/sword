"""
Cold SFT Fine-Tuning Pipeline for Ling-3.0-tiny (Text-to-Text).

Optimized for Ling-3.0-tiny architecture (BailingMoeV3 hybrid MLA + MoE)
using Sword Flash MLA SDPA + Fast MoE speed engine and Unsloth / Hugging Face SFT.

Key Capabilities:
- Pure text-to-text cold SFT (audio/vision stripped)
- Automated dataset download from Google Drive (gdown + streaming requests fallback)
- Immediate dataset shuffling with fixed seed before batching/splitting
- Multi-tier model loader: Sword load_ling_model + FastModel / PeftModel
- Response-only training loss masking (<|im_start|>user / assistant or <think> tags)
- Safe Hugging Face checkpoint syncing & local rotation pruning (API tokens stripped from source)
"""

# ── 1. Unsloth Import Guard (Must be imported before transformers / peft) ────
try:
    import unsloth
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
except (ImportError, NotImplementedError, Exception):
    try:
        from unsloth import FastLanguageModel as FastModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
    except (ImportError, NotImplementedError, Exception):
        FastModel = None
        get_chat_template = None
        train_on_responses_only = None

import os
import gc
import glob
import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)

# ── 2. Sword Acceleration Import ──────────────────────────────────────
try:
    import sword
    from sword.ling import load_ling_model, patch_ling, setup_fla_compatibility
    from sword.trainer import setup_blackwell_environment, download_from_drive
    setup_fla_compatibility()
except ImportError:
    load_ling_model = None
    patch_ling = None
    setup_blackwell_environment = None
    download_from_drive = None
    setup_fla_compatibility = None

from datasets import load_dataset, Dataset
try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    SFTTrainer = None
    SFTConfig = None

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    HfApi = None
    snapshot_download = None

# Optimize runtime allocations
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "warning"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# =========================================================
# CONFIGURATION
# =========================================================
# HF credentials (read from environment variable, never hardcode in script)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "")  # e.g., "username/Ling-3.0-tiny-cold-sft"
RESUME_FROM_HF_CHECKPOINT = ""                       # e.g., "checkpoint-500" or "" for fresh start

# Dataset Source: If local file exists, it will be loaded directly.
# Otherwise, if DATASET_DRIVE_URL is provided, it downloads to DATASET_LOCAL_PATH.
DATASET_LOCAL_PATH = r"E:\lingtiny\merged_finetune_dataset_with_effort.jsonl"
DATASET_DRIVE_URL = os.environ.get("DATASET_DRIVE_URL", "")  # e.g. "https://drive.google.com/file/d/..."

CFG = dict(
    model_id="inclusionAI/Ling-3.0-tiny-base",  # Raw text completion base model (Cold SFT)
    output_dir="sft-ling-3.0-tiny-base",
    max_seq_len=4096,                          # Target sequence context
    num_epochs=1,
    per_device_bs=1,                           # bs=1 is optimal for long sequences (32k+ tokens)
    grad_accum=16,                             # Equivalent effective batch size (1 * 16 = 16)
    lr=2e-4,
    warmup_steps=50,
    seed=3407,
    save_steps=100,
    save_limit=2,
    lora_r=16,
    lora_alpha=32,
    qat_scheme=None,                       # Set to "int4" if TorchAO QAT is desired, else None for standard LoRA
    eval_steps=150,
    eval_sample_count=500,
)


# =========================================================
# 📥 GOOGLE DRIVE DOWNLOADER & FALLBACK
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


def download_dataset_from_drive(drive_url: str, dest_path: str, force: bool = False) -> str:
    """Downloads dataset from Google Drive if local destination is missing."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)

    if os.path.exists(dest_path) and not force:
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"✅ Dataset file already present locally ({size_mb:.1f} MB) -> {dest_path}")
        return dest_path

    file_id = _extract_drive_file_id(drive_url)

    try:
        import gdown
        print(f"📥 Downloading Drive file {file_id} -> {dest_path} (via gdown)...")
        gdown.download(id=file_id, output=dest_path, quiet=False)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise RuntimeError("gdown reported success but output file is missing or empty")
    except ImportError:
        print("⚠️ gdown not installed (pip install gdown) — using streaming requests fallback.")
        _download_from_drive_requests_fallback(file_id, dest_path)
    except Exception as e:
        print(f"⚠️ gdown download encountered error ({e}) — retrying with requests fallback...")
        _download_from_drive_requests_fallback(file_id, dest_path)

    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        raise RuntimeError(f"Download failed — {dest_path} is missing or empty.")

    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"✅ Download complete ({size_mb:.1f} MB) -> {dest_path}")
    return dest_path


# =========================================================
# 📊 VRAM REQUIREMENT CALCULATOR & PRE-FLIGHT CHECK
# =========================================================
def estimate_vram_requirement(
    model_params_b: float = 7.9,
    active_params_b: float = 1.3,
    precision_bytes: float = 2.0,  # 2 for BF16/FP16, 1 for FP8, 0.5 for INT4
    max_seq_len: int = 32768,
    per_device_bs: int = 1,
    lora_r: int = 16,
    num_layers: int = 24,
    hidden_dim: int = 2048,
    vocab_size: int = 157184,
    use_gradient_checkpointing: bool = True,
    use_chunked_loss: bool = True,
    optim: str = "adamw_8bit",
) -> Dict[str, Any]:
    """
    Computes precise breakdown of VRAM needed to fine-tune Ling-3.0-tiny MoE model:
    1. Base Model Weights
    2. LoRA Adapter Weights & Gradients
    3. Optimizer States (AdamW 8-bit vs 32-bit)
    4. Forward/Backward Activations (with Flash MLA SDPA & Gradient Checkpointing)
    5. Chunked Loss Logits vs Full Logit Materialization
    6. PyTorch CUDA Workspace & Context Buffers
    """
    # 1. Base Model Weights (frozen during LoRA)
    base_weights_gb = (model_params_b * 1e9 * precision_bytes) / (1024 ** 3)

    # 2. LoRA Parameters (Attention + MoE projections)
    # Target modules: q, k, v, o, kv_a, kv_b, gate, up, down across 24 layers
    # Approximate trainable params at rank 16: ~40M to 70M params
    lora_params_m = lora_r * 2 * 12 * num_layers * hidden_dim / 1e6
    lora_weights_gb = (lora_params_m * 1e6 * precision_bytes) / (1024 ** 3)
    lora_grads_gb = (lora_params_m * 1e6 * precision_bytes) / (1024 ** 3)

    # 3. Optimizer States
    # adamw_8bit: 2 bytes per param (1 byte momentum + 1 byte variance)
    # adamw_torch / adamw_fused: 8 bytes per param (4 bytes FP32 momentum + 4 bytes FP32 variance)
    optim_bytes_per_param = 2.0 if "8bit" in optim else 8.0
    optim_states_gb = (lora_params_m * 1e6 * optim_bytes_per_param) / (1024 ** 3)

    # 4. Activation Memory
    # With gradient checkpointing enabled, PyTorch only stores boundary activations per transformer block
    # Hidden state tensor per layer: [B, L, H] * precision_bytes
    boundary_act_per_layer = per_device_bs * max_seq_len * hidden_dim * precision_bytes
    if use_gradient_checkpointing:
        # Checkpointed: Stored boundary inputs across layers + recomputed peak block
        checkpointed_act_gb = (boundary_act_per_layer * num_layers) / (1024 ** 3)
        peak_block_act_gb = (boundary_act_per_layer * 3.5) / (1024 ** 3)  # Peak inner MLA/SwiGLU within 1 block
        total_act_gb = checkpointed_act_gb + peak_block_act_gb
    else:
        # Non-checkpointed: Quadratic/Linear activations retained across ALL layers
        total_act_gb = (boundary_act_per_layer * num_layers * 4.0) / (1024 ** 3)

    # 5. Logits & Loss Memory
    if use_chunked_loss:
        # Chunk size 512 avoids [B, L, Vocab] FP32 logits
        logits_gb = (per_device_bs * 512 * vocab_size * 4) / (1024 ** 3)
    else:
        # Full materialization [B, L, 157184] FP32
        logits_gb = (per_device_bs * max_seq_len * vocab_size * 4) / (1024 ** 3)

    # 6. CUDA Context & PyTorch Fragment Overhead (~1.0 - 1.5 GB)
    cuda_overhead_gb = 1.2

    total_vram_needed_gb = (
        base_weights_gb
        + lora_weights_gb
        + lora_grads_gb
        + optim_states_gb
        + total_act_gb
        + logits_gb
        + cuda_overhead_gb
    )

    return {
        "total_gb": round(total_vram_needed_gb, 2),
        "base_weights_gb": round(base_weights_gb, 2),
        "lora_weights_gb": round(lora_weights_gb, 3),
        "lora_grads_gb": round(lora_grads_gb, 3),
        "optim_states_gb": round(optim_states_gb, 3),
        "activations_gb": round(total_act_gb, 2),
        "logits_gb": round(logits_gb, 2),
        "cuda_overhead_gb": round(cuda_overhead_gb, 2),
        "max_seq_len": max_seq_len,
        "batch_size": per_device_bs,
    }


def print_vram_preflight(cfg: Dict[str, Any], precision_bytes: float = 2.0):
    """Prints a human-readable pre-flight VRAM report and warns if hardware is insufficient."""
    est = estimate_vram_requirement(
        model_params_b=7.9,
        precision_bytes=precision_bytes,
        max_seq_len=cfg.get("max_seq_len", 4096),
        per_device_bs=cfg.get("per_device_bs", 1),
        lora_r=cfg.get("lora_r", 16),
        optim=cfg.get("optim", "adamw_8bit"),
        use_gradient_checkpointing=True,
        use_chunked_loss=True,
    )

    avail_vram_gb = 0.0
    gpu_name = "None (CPU)"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        avail_vram_gb = round(p.total_memory / (1024 ** 3), 2)
        gpu_name = p.name

    print("=" * 68)
    print(f"🖥️  VRAM ESTIMATION PRE-FLIGHT CHECK ({cfg.get('model_id', 'Ling-3.0-tiny')})")
    print("=" * 68)
    print(f"Target Max Sequence Length : {est['max_seq_len']:,} tokens")
    print(f"Per-Device Batch Size      : {est['batch_size']}")
    print(f"Base Model Weights         : {est['base_weights_gb']} GB")
    print(f"LoRA Adapter Weights+Grads : {est['lora_weights_gb'] + est['lora_grads_gb']:.2f} GB")
    print(f"Optimizer States (8-bit)   : {est['optim_states_gb']} GB")
    print(f"Activations (with SDPA+GC) : {est['activations_gb']} GB")
    print(f"Chunked Loss Head (512 tk) : {est['logits_gb']} GB (saved {((est['batch_size'] * est['max_seq_len'] * 157184 * 4)/(1024**3) - est['logits_gb']):.1f} GB vs unchunked!)")
    print(f"PyTorch CUDA Overhead      : {est['cuda_overhead_gb']} GB")
    print("-" * 68)
    print(f"⚡ TOTAL ESTIMATED VRAM     : {est['total_gb']} GB")
    print(f"🎮 DETECTED GPU VRAM       : {avail_vram_gb} GB ({gpu_name})")
    print("=" * 68)

    if avail_vram_gb > 0 and avail_vram_gb < est["total_gb"]:
        deficit = est["total_gb"] - avail_vram_gb
        print(f"\n⚠️  CRITICAL VRAM WARNING: Current GPU has {avail_vram_gb} GB, but fine-tuning requires ~{est['total_gb']} GB (Deficit: {deficit:.1f} GB).")
        print(f"   Recommended Options:")
        print(f"   1. Use a cloud GPU with at least 24 GB (RTX 3090/4090, A10, L4, A100).")
        print(f"   2. If testing locally, load in 4-bit/FP8 or reduce sequence length.")
    else:
        print(f"✅ VRAM check passed: GPU memory is sufficient for training.")
    print("")


# =========================================================
# 🔀 DATASET LOADER & SHUFFLER
# =========================================================
def load_and_prepare_dataset(local_path: str, drive_url: str = "", seed: int = 3407) -> Dataset:
    """
    Resolves dataset source (local path or Google Drive link), loads JSONL,
    and immediately shuffles before any downstream train/test splitting.
    """
    # 1. Resolve path
    target_path = local_path
    if not os.path.exists(target_path):
        if drive_url:
            print(f"[*] Local dataset not found at {target_path}. Downloading from Google Drive...")
            target_path = download_dataset_from_drive(drive_url, dest_path=local_path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{local_path}' and no DATASET_DRIVE_URL provided.\n"
                f"Please place the dataset at '{local_path}' or set DATASET_DRIVE_URL."
            )

    file_size_gb = os.path.getsize(target_path) / 1e9
    print(f"\n📖 Loading dataset from {target_path} ({file_size_gb:.2f} GB)...")

    # 2. Fast memory-mapped loading via Hugging Face datasets
    dataset = load_dataset("json", data_files=target_path, split="train")
    print(f"✅ Loaded {len(dataset):,} total records from {os.path.basename(target_path)}")

    # Inspect first record structure
    first_sample = dataset[0]
    if "messages" not in first_sample:
        raise ValueError(f"Expected 'messages' column in dataset, but got keys: {list(first_sample.keys())}")
    print(f"[*] Sample message turns: {len(first_sample['messages'])} (Roles: {[m.get('role') for m in first_sample['messages']]})")

    # 3. Shuffle immediately before anything else
    print(f"🔀 Shuffling dataset with seed {seed}...")
    dataset = dataset.shuffle(seed=seed)
    print(f"✅ Dataset successfully shuffled.")

    return dataset


# =========================================================
# 📥 CHECKPOINT DOWNLOADER & VALIDATOR
# =========================================================
def _is_valid_checkpoint(ckpt_dir: str) -> bool:
    if not os.path.isdir(ckpt_dir):
        return False
    if not os.path.exists(os.path.join(ckpt_dir, "trainer_state.json")):
        return False
    has_weights = (
        glob.glob(os.path.join(ckpt_dir, "adapter_model*"))
        or glob.glob(os.path.join(ckpt_dir, "*.safetensors"))
        or os.path.exists(os.path.join(ckpt_dir, "adapter_config.json"))
        or glob.glob(os.path.join(ckpt_dir, "pytorch_model*"))
    )
    return bool(has_weights)


def download_hf_checkpoint(hf_repo_id: str, checkpoint_name: str, local_output_dir: str, hf_token: str = "") -> str:
    ckpt_dir = os.path.join(local_output_dir, checkpoint_name)
    if _is_valid_checkpoint(ckpt_dir):
        print(f"✅ Checkpoint already present and verified: {ckpt_dir}")
        return ckpt_dir
    print(f"📥 Downloading {checkpoint_name} from {hf_repo_id} -> {ckpt_dir} ...")
    os.makedirs(local_output_dir, exist_ok=True)
    snapshot_download(
        repo_id=hf_repo_id,
        allow_patterns=[f"{checkpoint_name}/*", f"{checkpoint_name}/**"],
        local_dir=local_output_dir,
        token=hf_token or None,
        endpoint="https://huggingface.co",
    )
    if not _is_valid_checkpoint(ckpt_dir):
        raise RuntimeError(f"❌ Downloaded checkpoint at {ckpt_dir} is incomplete.")
    print(f"✅ Checkpoint verified: {ckpt_dir}")
    return ckpt_dir


# =========================================================
# 💾 CHECKPOINT CALLBACK
# =========================================================
class FullCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer, output_dir: str, save_steps: int, save_limit: int = 2):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.save_limit = save_limit
        self.hf_api = HfApi(endpoint="https://huggingface.co", token=HF_TOKEN or None) if (HF_TOKEN and MODEL_REPO_ID) else None

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.save_steps != 0 or not state.is_world_process_zero:
            return control

        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\n💾 Saving checkpoint at step {state.global_step} → {ckpt_dir}")

        try:
            model.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)
            print("  ✅ LoRA weights + tokenizer saved.")
        except Exception as e:
            print(f"  ⚠️ Save model failed: {e}")

        try:
            state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
        except Exception as e:
            print(f"  ⚠️ Trainer_state save failed: {e}")

        if self.hf_api and MODEL_REPO_ID:
            try:
                self.hf_api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True, private=True)
                self.hf_api.upload_folder(
                    folder_path=ckpt_dir,
                    repo_id=MODEL_REPO_ID,
                    path_in_repo=f"checkpoint-{state.global_step}",
                    repo_type="model",
                    commit_message=f"checkpoint step {state.global_step}",
                )
                print(f"  ☁️ Uploaded to {MODEL_REPO_ID}/checkpoint-{state.global_step}")
            except Exception as e:
                print(f"  ⚠️ HF upload failed (preserved locally): {e}")

        # Rotate old local checkpoints
        checkpoints = sorted(
            glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0
        )
        while len(checkpoints) > self.save_limit:
            old = checkpoints.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            print(f"🗑️ Removed old local checkpoint: {old}")

        return control


# =========================================================
# 🚀 MAIN SFT PIPELINE FOR LING-3.0-TINY
# =========================================================
def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Pre-flight VRAM estimation check
    print_vram_preflight(CFG)

    # 1. Resolve & Prepare Dataset (with immediate shuffle)
    full_ds = load_and_prepare_dataset(
        local_path=DATASET_LOCAL_PATH,
        drive_url=DATASET_DRIVE_URL,
        seed=CFG["seed"],
    )

    test_split_size = min(CFG["eval_sample_count"], max(10, len(full_ds) // 50))
    split_ds = full_ds.train_test_split(test_size=test_split_size, seed=CFG["seed"])
    train_ds, eval_ds = split_ds["train"], split_ds["test"]
    print(f"✅ Train set: {len(train_ds):,} samples | Eval set: {len(eval_ds):,} samples")

    # 2. Load Ling-3.0-tiny Model & Tokenizer
    print(f"\n[Ling SFT] Loading model: {CFG['model_id']}...")
    model = None
    tokenizer = None

    # Target attention & MoE projections for Ling-3.0-tiny architecture
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "kv_a_proj_with_mqa", "kv_b_proj",
    ]

    # Attempt FastModel if available
    if FastModel is not None:
        try:
            print("Loading via FastModel (Unsloth)...")
            model, tokenizer = FastModel.from_pretrained(
                CFG["model_id"],
                max_seq_length=CFG["max_seq_len"],
                load_in_4bit=False,
                load_in_16bit=True,
                use_gradient_checkpointing="unsloth",
                token=HF_TOKEN or None,
                trust_remote_code=True,
            )
            peft_kwargs = dict(
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=CFG["lora_r"],
                lora_alpha=CFG["lora_alpha"],
                lora_dropout=0,
                bias="none",
                random_state=CFG["seed"],
                target_modules=target_modules,
            )
            if CFG.get("qat_scheme"):
                peft_kwargs["qat_scheme"] = CFG["qat_scheme"]
            model = FastModel.get_peft_model(model, **peft_kwargs)
        except Exception as e:
            print(f"⚠️ FastModel initialization failed ({e}). Falling back to Sword load_ling_model + PEFT.")
            model = None
            tokenizer = None

    # Fallback to Sword native load_ling_model + standard PEFT
    if model is None:
        if load_ling_model is not None:
            print("Loading via Sword load_ling_model (Flash MLA SDPA + Fast MoE)...")
            base_model, tokenizer = load_ling_model(
                model_name_or_path=CFG["model_id"],
                device_map=None,  # Do NOT use device_map="auto" during training!
                patch_sword=True,
                patch_moe=True,
            )
            base_model.train()
        else:
            print("Loading via Hugging Face AutoModelForCausalLM...")
            tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"], trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                CFG["model_id"],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=CFG["lora_r"],
            lora_alpha=CFG["lora_alpha"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()

    # Ensure tokenizer has padding token configured
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    tokenizer.padding_side = "right"

    # Ling-3.0-tiny Native Chat Template with <think> preservation:
    # Stock Ling chat_template.jinja strips literal <think>...</think> tags.
    # For cold-start SFT and subsequent GRPO, we preserve <think>...</think> tags so
    # the base model explicitly learns reasoning delimiters for RL rollout parsing.
    tokenizer.chat_template = (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "{{- '<role>SYSTEM</role>' + message.content + '<|role_end|>' }}"
        "{%- elif message.role == 'user' %}"
        "{{- '<role>HUMAN</role>' + message.content + '<|role_end|>' }}"
        "{%- elif message.role == 'assistant' %}"
        "{{- '<role>ASSISTANT</role>' + message.content + '<|role_end|>' }}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<role>ASSISTANT</role>' }}"
        "{%- endif %}"
    )
    print("[*] Configured Ling native <role> chat template (with <think> reasoning preservation).")

    # 3. Format dataset text prompts using chat template
    def formatting_prompts_func(examples):
        convs = examples["messages"]
        texts = []
        for conversation in convs:
            formatted = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(formatted)
        return {"text": texts}

    print("[*] Formatting train and eval datasets using chat template...")
    train_ds = train_ds.map(formatting_prompts_func, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(formatting_prompts_func, batched=True, remove_columns=eval_ds.column_names)

    # 4. Training Callbacks & Configuration
    checkpoint_cb = FullCheckpointCallback(
        tokenizer=tokenizer,
        output_dir=CFG["output_dir"],
        save_steps=CFG["save_steps"],
        save_limit=CFG["save_limit"],
    )

    # Enable gradient checkpointing to keep activation memory minimal for long sequences
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    training_args = SFTConfig(
        output_dir=CFG["output_dir"],
        num_train_epochs=CFG["num_epochs"],
        per_device_train_batch_size=CFG["per_device_bs"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["lr"],
        max_grad_norm=0.3,
        warmup_steps=CFG["warmup_steps"],
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",  # Handled by FullCheckpointCallback
        eval_strategy="steps",
        eval_steps=CFG["eval_steps"],
        report_to="none",
        seed=CFG["seed"],
        dataset_text_field="text",
        max_length=CFG["max_seq_len"],
        packing=False,
    )

    # Ensure model config has model_type populated so external tools don't crash
    if hasattr(model, "config") and (not getattr(model.config, "model_type", None) or model.config.model_type == ""):
        model.config.model_type = "bailing_hybrid"

    # Unsloth monkey-patches TRL's SFTTrainer.__init__ to inspect known model types (Llama, Qwen, etc.),
    # which raises TypeError for custom architectures like Ling's BailingMoeV3Config.
    # We unwrap SFTTrainer.__init__ back to the pure Hugging Face TRL implementation:
    import inspect
    import trl
    if hasattr(trl, "SFTTrainer") and hasattr(trl.SFTTrainer, "__init__"):
        try:
            trl.SFTTrainer.__init__ = inspect.unwrap(trl.SFTTrainer.__init__)
            print("[*] Unwrapped SFTTrainer to native TRL (bypassing Unsloth auto-packing check for Ling).")
        except Exception as e:
            print(f"⚠️ SFTTrainer unwrap notice: {e}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[checkpoint_cb],
    )

    # 5. Mask prompt turns — compute loss on assistant responses only
    if train_on_responses_only is not None:
        chat_tmpl = getattr(tokenizer, "chat_template", "") or ""
        if "<role>HUMAN</role>" in chat_tmpl or "<role>" in chat_tmpl:
            inst_part = "<role>HUMAN</role>"
            resp_part = "<role>ASSISTANT</role>"
        else:
            inst_part = "<|im_start|>user\n"
            resp_part = "<|im_start|>assistant\n"

        try:
            trainer = train_on_responses_only(
                trainer,
                instruction_part=inst_part,
                response_part=resp_part,
            )
            print(f"[*] Enabled response-only loss masking (response delimiter: {resp_part!r}).")
        except Exception as e:
            print(f"⚠️ train_on_responses_only skipped ({e}). Training on full sequence.")

    # 6. Resume or Start Training
    if RESUME_FROM_HF_CHECKPOINT:
        ckpt_dir = download_hf_checkpoint(MODEL_REPO_ID, RESUME_FROM_HF_CHECKPOINT, CFG["output_dir"], HF_TOKEN)
        print(f"▶️ Resuming from {ckpt_dir} ...")
        trainer.train(resume_from_checkpoint=ckpt_dir)
    else:
        print("▶️ Starting cold SFT training for Ling-3.0-tiny...")
        trainer.train()

    # 7. Final Model Export
    final_dir = os.path.join(CFG["output_dir"], "final_model")
    os.makedirs(final_dir, exist_ok=True)
    print(f"\n💾 Saving final trained model -> {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ Model & tokenizer saved locally at {final_dir}")

    # Optional QAT conversion if enabled
    if CFG.get("qat_scheme") == "int4":
        try:
            print("\n🎯 Converting QAT model to true int4 via TorchAO...")
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig
            from torchao.quantization import Int4WeightOnlyConfig

            quantize_(model, QATConfig(step="convert"))
            qat_dir = os.path.join(CFG["output_dir"], "final_model_qat_int4")
            os.makedirs(qat_dir, exist_ok=True)
            model.save_pretrained_torchao(
                qat_dir, tokenizer,
                torchao_config=Int4WeightOnlyConfig(),
            )
            print(f"✅ QAT int4 model saved locally: {qat_dir}")
        except Exception as e:
            print(f"⚠️ TorchAO conversion skipped: {e}")

    # Sync final model to HF Hub if configured
    if HF_TOKEN and MODEL_REPO_ID:
        try:
            api = HfApi(endpoint="https://huggingface.co", token=HF_TOKEN)
            api.upload_folder(
                folder_path=final_dir,
                repo_id=MODEL_REPO_ID,
                path_in_repo="final_model",
                repo_type="model",
                commit_message="final Ling-3.0-tiny cold SFT model",
            )
            print(f"✅ Uploaded to {MODEL_REPO_ID}/final_model")
        except Exception as e:
            print(f"⚠️ Final HF upload failed (model preserved locally): {e}")

    print("\n🎉 Cold SFT training pipeline completed successfully!")


if __name__ == "__main__":
    main()
