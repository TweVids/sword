"""
Sword + Unsloth High-Throughput SFT Fine-Tuning Pipeline.

Accelerated by Sword Pure-PyTorch FlashAttention SDPA for NVIDIA Blackwell (SM100).
Supports:
- Auto dataset download from Google Drive
- Automated checkpoint resumption from HuggingFace Hub
- Pure-PyTorch FlashAttention attention patching (bypasses broken xformers/FA2 wheels)
- Safe HuggingFace sync and automated local checkpoint pruning
"""

import os
import gc
import glob
import shutil
import torch
from pathlib import Path

# ── 1. Unsloth MUST be imported before transformers / trl ────────
try:
    import unsloth
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
except ImportError:
    try:
        from unsloth import FastLanguageModel as FastModel
        from unsloth.chat_templates import train_on_responses_only
    except ImportError:
        FastModel = None
        train_on_responses_only = None

# ── 2. Initialize Sword Blackwell Optimizations ───────────────
import sword
from sword import (
    setup_blackwell_environment,
    download_from_drive,
    download_hf_checkpoint,
    is_valid_checkpoint,
    load_offline_dataset,
    FullCheckpointCallback,
    patch_qwen,
)

setup_blackwell_environment()

from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi


# Provide via environment variable OR fill directly in the strings below
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # e.g., "hf_..."
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")  # e.g., "your-username/your-model-repo"

# Google Drive link for dataset (leave blank if dataset is already local)
DATASET_DRIVE_URL = os.environ.get("DATASET_DRIVE_URL", "")

# Checkpoint resumption: set to folder name (e.g., "checkpoint-1100") to resume from HF Hub
# Leave as "" to start fresh
RESUME_FROM_HF_CHECKPOINT = ""

CFG = dict(
    model_id="model-name",  # Or your target Qwen model
    dataset_main="whenever is this as a path",
    output_dir="sft-qat-27b",
    max_seq_len=7500, # how much token you want to train w
    num_epochs=1,
    per_device_bs=4,
    grad_accum=8,
    lr=5e-5,
    warmup_steps=449,
    seed=42,
    save_steps=100,
    save_limit=2,
)


# =========================================================
# 🚀 MAIN TRAINING PIPELINE
# =========================================================
def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"[*] GPU : {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. Load Model & Tokenizer / Processor
    print(f"\n[*] Loading model and tokenizer in 4-bit QLoRA mode via FastModel...")
    processor = None
    if FastModel is not None:
        model, processor_or_tokenizer = FastModel.from_pretrained(
            model_name=CFG["model_id"],
            max_seq_length=CFG["max_seq_len"],
            dtype=torch.bfloat16,
            load_in_4bit=True,
            load_in_8bit=False,
            device_map={"": 0} if torch.cuda.is_available() else None,
            token=HF_TOKEN if HF_TOKEN else None,
        )
        # Vision-Language models (e.g. Qwen3VLProcessor) wrap the underlying tokenizer
        if hasattr(processor_or_tokenizer, "tokenizer") and processor_or_tokenizer.tokenizer is not None:
            processor = processor_or_tokenizer
            tokenizer = processor_or_tokenizer.tokenizer
            print(f"[*] Detected multimodal processor ({processor.__class__.__name__}); extracted text tokenizer.")
        else:
            tokenizer = processor_or_tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            CFG["model_id"],
            token=HF_TOKEN if HF_TOKEN else None,
            trust_remote_code=True,
        )
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            CFG["model_id"],
            quantization_config=quant_config,
            device_map={"": 0} if torch.cuda.is_available() else None,
            trust_remote_code=True,
            token=HF_TOKEN if HF_TOKEN else None,
        )

    # Ensure valid EOS and PAD tokens exist in tokenizer vocabulary
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if tokenizer.eos_token is None or (vocab and tokenizer.eos_token not in vocab):
        if "<|im_end|>" in vocab:
            tokenizer.eos_token = "<|im_end|>"
        elif "<|endoftext|>" in vocab:
            tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_len = len(tokenizer) if hasattr(tokenizer, "__len__") else len(vocab)
    print(f"[*] Tokenizer loaded. Vocab size: {vocab_len}, EOS token: {tokenizer.eos_token}")

    # Freeze vision/multimodal layers if present
    for name, param in model.named_parameters():
        if "visual" in name.lower():
            param.requires_grad_(False)

    # 2. Attach LoRA Adapters
    print("\n[*] Attaching LoRA adapters (r=64, alpha=128, rsLoRA)...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=64,
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        random_state=CFG["seed"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        target_modules="all-linear",
    )

    # 4. ⚡ INJECT SWORD PURE FLASHATTENTION SDPA SPEED ENGINE ⚡
    # Replaces attention layers with pure-PyTorch O(N) SDPA to eliminate
    # broken xformers / flash-attn C++ wheel dependencies on Blackwell SM100!
    print("\n[*] Injecting Sword Pure FlashAttention SDPA Speed Engine...")
    model = patch_qwen(model, mode="flash")

    # 5. Resolve & Download Dataset
    dataset_path = CFG["dataset_main"]
    if not os.path.exists(dataset_path):
        # Look in script directory as alternative
        script_dir_candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(dataset_path))
        if os.path.exists(script_dir_candidate):
            dataset_path = script_dir_candidate
            CFG["dataset_main"] = dataset_path

    if DATASET_DRIVE_URL and not os.path.exists(dataset_path):
        print(f"\n[*] Downloading dataset from Google Drive...")
        download_from_drive(DATASET_DRIVE_URL, dataset_path)
    elif not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please set DATASET_DRIVE_URL or place the dataset at {dataset_path}."
        )

    # 6. Load Dataset & Train/Test Split
    train_ds, eval_ds = load_offline_dataset(dataset_path, seed=CFG["seed"], test_size=700)

    # 7. Checkpoint Callback with automated HuggingFace upload & pruning
    cb_kwargs = dict(
        tokenizer=tokenizer,
        output_dir=CFG["output_dir"],
        save_steps=CFG["save_steps"],
        save_limit=CFG["save_limit"],
        hf_repo_id=HF_REPO_ID if HF_REPO_ID else None,
        hf_token=HF_TOKEN if HF_TOKEN else None,
    )
    try:
        checkpoint_cb = FullCheckpointCallback(processor=processor, **cb_kwargs)
    except TypeError:
        # Backward compatibility if an older version of sword is imported in the session
        checkpoint_cb = FullCheckpointCallback(**cb_kwargs)
        checkpoint_cb.processor = processor

    # 8. SFT Training Arguments
    sft_config_kwargs = dict(
        dataset_text_field="text",
        packing=True,
        max_length=CFG["max_seq_len"], #max_length instead
        output_dir=CFG["output_dir"],
        num_train_epochs=CFG["num_epochs"],
        per_device_train_batch_size=CFG["per_device_bs"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["lr"],
        max_grad_norm=0.5,
        warmup_steps=CFG["warmup_steps"],
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        logging_steps=10,
        save_strategy="no",  # Managed by FullCheckpointCallback
        eval_strategy="steps",
        eval_steps=150,
        report_to="none",
        seed=CFG["seed"],
    )

    # Explicitly set eos_token to tokenizer's actual vocabulary token
    # (Prevents TRL packing ValueError: '<EOS_TOKEN>' is not found in vocabulary)
    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
        sft_config_kwargs["eos_token"] = tokenizer.eos_token

    try:
        training_args = SFTConfig(**sft_config_kwargs)
    except TypeError:
        # For older TRL versions where eos_token is not a parameter of SFTConfig
        sft_config_kwargs.pop("eos_token", None)
        training_args = SFTConfig(**sft_config_kwargs)

    # Ensure training_args.eos_token matches a valid token in tokenizer vocabulary
    if hasattr(training_args, "eos_token"):
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        if not training_args.eos_token or (vocab and training_args.eos_token not in vocab):
            training_args.eos_token = tokenizer.eos_token

    # TRL ≥0.12 renamed `tokenizer` → `processing_class`; support both versions
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            callbacks=[checkpoint_cb],
        )
    except TypeError:
        # Fallback for older TRL that still uses `tokenizer`
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            callbacks=[checkpoint_cb],
        )
    checkpoint_cb.trainer = trainer

    # Apply instruction/response masking
    if train_on_responses_only is not None:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    # 9. Pre-Training Sanity Inspection
    print("\n" + "=" * 70)
    print("🔍 [Sword] PRE-TRAINING SANITY CHECK & TENSOR INSPECTION")
    print("=" * 70)
    sample_batch = next(iter(trainer.get_train_dataloader()))
    input_ids = sample_batch["input_ids"]
    labels = sample_batch["labels"]

    actual_batch_size = input_ids.shape[0]
    actual_seq_len = input_ids.shape[1]
    active_tokens = (labels != -100).sum().item()
    total_tokens = labels.numel()
    active_pct = (active_tokens / total_tokens) * 100

    print(f"[*] Configured max_seq_len        : {CFG['max_seq_len']:,} tokens")
    print(f"[*] Actual Dataloader Tensor Shape: {list(input_ids.shape)} (batch={actual_batch_size}, seq_len={actual_seq_len:,})")
    print(f"[*] Active Trained Tokens in Batch: {active_tokens:,} / {total_tokens:,} ({active_pct:.2f}% unmasked)")

    decoded_sample = tokenizer.decode(input_ids[0][:250], skip_special_tokens=False)
    print("\n📄 Raw Token Sample (first 250 tokens):")
    print("-" * 70)
    print(decoded_sample)
    print("-" * 70 + "\n")

    # 10. Execute Training (Resume or Fresh)
    if RESUME_FROM_HF_CHECKPOINT and HF_REPO_ID:
        ckpt_dir = download_hf_checkpoint(
            hf_repo_id=HF_REPO_ID,
            checkpoint_name=RESUME_FROM_HF_CHECKPOINT,
            local_output_dir=CFG["output_dir"],
            hf_token=HF_TOKEN if HF_TOKEN else None,
        )
        print(f"\n▶️  Resuming training from verified checkpoint: {ckpt_dir} ...")
        trainer.train(resume_from_checkpoint=ckpt_dir)
    else:
        print("\n▶️  Starting fresh training with Sword Pure FlashAttention...")
        trainer.train()

    # 11. Final Save & Export
    final_dir = os.path.join(CFG["output_dir"], "final_model")
    print(f"\n[*] Saving final model to {final_dir}...")
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(final_dir, tokenizer, save_method="merged_16bit")
    else:
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    if processor is not None and hasattr(processor, "save_pretrained"):
        try:
            processor.save_pretrained(final_dir)
        except Exception:
            pass
    print(f"✅ Local final model saved: {final_dir}")

    # Optional final upload to HF Hub
    if HF_REPO_ID and HF_TOKEN:
        print(f"☁️  Uploading final model to {HF_REPO_ID}/final_model ...")
        try:
            api = HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"), token=HF_TOKEN)
            api.upload_folder(
                folder_path=final_dir,
                repo_id=HF_REPO_ID,
                path_in_repo="final_model",
                repo_type="model",
                commit_message="Sword final merged model",
            )
            print(f"✅ Final model uploaded successfully to {HF_REPO_ID}/final_model")
        except Exception as e:
            print(f"⚠️  Final HF upload failed (model is safely saved locally): {e}")

    print("\n🎉 [Sword] Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
