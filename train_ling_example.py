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
    from sword.ling import load_ling_model, patch_ling
    from sword.trainer import setup_blackwell_environment, download_from_drive
except ImportError:
    load_ling_model = None
    patch_ling = None
    setup_blackwell_environment = None
    download_from_drive = None

from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi, snapshot_download

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
    per_device_bs=2,
    grad_accum=8,
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
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
                patch_sword=True,
                patch_moe=True,
            )
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

    # Ensure tokenizer has padding token and chat template configured
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    tokenizer.padding_side = "right"

    # Base Model Cold-Start: Register ChatML and reasoning delimiters
    # Since Ling-3.0-tiny-base is a raw completion base model, ensure conversation tokens exist
    special_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    existing_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    missing_tokens = [t for t in special_tokens if t not in existing_vocab]
    if missing_tokens:
        print(f"[*] Base model cold-start: adding {len(missing_tokens)} special tokens: {missing_tokens}")
        tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))

    # Configure ChatML template for cold-start SFT
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
    )

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
        try:
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
            print("[*] Enabled response-only loss masking (<|im_start|>assistant).")
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
