"""
Custom Checkpoint Manager: Independent serialization, SHA-256 dataset tracking,
checkpoint resume discovery, and Hugging Face Hub synchronization.
"""

import os
import re
import json
import time
import glob
import shutil
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from huggingface_hub import HfApi, hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


class CheckpointManager:
    """
    Manages custom checkpoint saving, automatic repo creation, resume discovery,
    SHA-256 dataset deduplication logs, and Hugging Face uploads.
    """
    def __init__(
        self,
        output_dir: str = "checkpoints",
        num_latest_checkpoint_save: int = 2,
        upload_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        is_save_and_upload_checkpoint: bool = True,
    ):
        self.output_dir = output_dir
        self.num_latest_checkpoint_save = max(1, num_latest_checkpoint_save)
        self.upload_id = upload_id
        self.hf_token = hf_token
        self.is_save_and_upload = is_save_and_upload_checkpoint
        os.makedirs(self.output_dir, exist_ok=True)

        self.api = None
        if self.upload_id and self.hf_token and HAS_HF_HUB:
            try:
                self.api = HfApi(token=self.hf_token)
                self.api.create_repo(
                    repo_id=self.upload_id,
                    repo_type="model",
                    private=True,
                    exist_ok=True,
                )
            except Exception as e:
                print(f"[CheckpointManager] Note: Hugging Face repo setup notice: {e}")

    @staticmethod
    def compute_sha256(data_source: Union[str, bytes]) -> str:
        """Computes SHA-256 checksum for a file path or in-memory content."""
        hasher = hashlib.sha256()
        if isinstance(data_source, str) and os.path.isfile(data_source):
            with open(data_source, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        elif isinstance(data_source, str):
            hasher.update(data_source.encode("utf-8"))
        elif isinstance(data_source, bytes):
            hasher.update(data_source)
        return hasher.hexdigest()

    def get_completed_dataset_hashes(self) -> set:
        """Collects all SHA-256 hashes recorded in local data_complete.logs."""
        completed_hashes = set()
        log_patterns = [
            os.path.join(self.output_dir, "data_complete.logs"),
            os.path.join(self.output_dir, "checkpoint-*", "data_complete.logs"),
        ]
        for pat in log_patterns:
            for log_file in glob.glob(pat):
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            h = line.strip().split()[0] if line.strip() else ""
                            if h:
                                completed_hashes.add(h)
                except Exception:
                    pass
        return completed_hashes

    def is_dataset_completed(self, dataset_hash: str) -> bool:
        """Checks if dataset has already been trained on."""
        return dataset_hash in self.get_completed_dataset_hashes()

    def mark_dataset_completed(self, dataset_hash: str, dataset_name: str = ""):
        """Appends dataset hash to global data_complete.logs."""
        main_log = os.path.join(self.output_dir, "data_complete.logs")
        with open(main_log, "a", encoding="utf-8") as f:
            f.write(f"{dataset_hash}  {dataset_name}\n")

    def find_latest_checkpoint(self, resume_query: Optional[str] = None) -> Optional[str]:
        """
        Discovers the matching or latest checkpoint-xxx directory locally or on Hugging Face.
        """
        # 1. Exact path match
        if resume_query and os.path.isdir(resume_query):
            return resume_query

        # 2. Local search in output_dir
        local_checkpoints = sorted(
            glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
            key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)) if re.search(r"checkpoint-(\d+)", p) else -1
        )

        if resume_query:
            query_clean = os.path.basename(resume_query.rstrip("/\\"))
            for ckpt in local_checkpoints:
                if os.path.basename(ckpt) == query_clean:
                    return ckpt

        if local_checkpoints:
            return local_checkpoints[-1]

        # 3. Hugging Face search if enabled
        if self.api and self.upload_id:
            try:
                files = self.api.list_repo_files(repo_id=self.upload_id, repo_type="model")
                ckpt_names = set()
                for f in files:
                    m = re.match(r"(checkpoint-\d+)/", f)
                    if m:
                        ckpt_names.add(m.group(1))
                if ckpt_names:
                    sorted_hf = sorted(
                        list(ckpt_names),
                        key=lambda name: int(name.split("-")[1]) if "-" in name else -1
                    )
                    target_name = sorted_hf[-1] if not resume_query else resume_query
                    # Download checkpoint files locally
                    local_target = os.path.join(self.output_dir, target_name)
                    os.makedirs(local_target, exist_ok=True)
                    for f in files:
                        if f.startswith(f"{target_name}/"):
                            hf_hub_download(
                                repo_id=self.upload_id,
                                filename=f,
                                local_dir=self.output_dir,
                                token=self.hf_token,
                            )
                    return local_target
            except Exception as e:
                print(f"[CheckpointManager] Remote checkpoint lookup notice: {e}")

        return None

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        base_model_name: str = "Qwen/Qwen3.5-0.8B",
        lora_r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        training_state: Optional[Dict[str, Any]] = None,
        reward_logs: Optional[List[Dict[str, Any]]] = None,
        completed_dataset_hash: Optional[str] = None,
    ) -> str:
        """
        Performs custom checkpoint serialization:
        adapter_model.safetensors, adapter_config.json, training_state.json,
        reward_details.jsonl, data_complete.logs, and uploads to Hugging Face.
        """
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. Save LoRA adapter weights (PEFT standard format)
        lora_sd = {}
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_sd[name] = param.detach().cpu()

        if hasattr(model, "save_pretrained") and hasattr(model, "peft_config"):
            try:
                model.save_pretrained(ckpt_dir)
            except Exception:
                pass

        if not os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
            peft_cfg = {
                "base_model_name_or_path": base_model_name,
                "bias": "none",
                "lora_alpha": lora_alpha,
                "lora_dropout": 0.0,
                "r": lora_r,
                "target_modules": target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
                "task_type": "CAUSAL_LM",
                "peft_type": "LORA",
            }
            with open(os.path.join(ckpt_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_cfg, f, indent=2)

        try:
            from safetensors.torch import save_file
            save_file(lora_sd, os.path.join(ckpt_dir, "adapter_model.safetensors"))
        except Exception:
            torch.save(lora_sd, os.path.join(ckpt_dir, "adapter_model.bin"))

        # 2. Save training state
        state = training_state or {}
        state["step"] = step
        state["timestamp"] = time.time() if "time" in globals() else 0.0
        with open(os.path.join(ckpt_dir, "training_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # 3. Save reward details log inside checkpoint folder
        if reward_logs:
            with open(os.path.join(ckpt_dir, "reward_details.jsonl"), "w", encoding="utf-8") as f:
                for entry in reward_logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 4. Copy / update data_complete.logs into checkpoint
        main_log = os.path.join(self.output_dir, "data_complete.logs")
        ckpt_log = os.path.join(ckpt_dir, "data_complete.logs")
        if os.path.exists(main_log):
            shutil.copy2(main_log, ckpt_log)
        elif completed_dataset_hash:
            with open(ckpt_log, "w", encoding="utf-8") as f:
                f.write(f"{completed_dataset_hash}\n")

        # 5. Save optimizer state if provided
        if optimizer is not None:
            try:
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
            except Exception:
                pass

        # 6. Prune older local checkpoints to respect num_latest_checkpoint_save
        self.prune_local_checkpoints()

        # 7. Upload checkpoint to Hugging Face if enabled
        if self.is_save_and_upload and self.api and self.upload_id:
            try:
                self.api.upload_folder(
                    folder_path=ckpt_dir,
                    path_in_repo=f"checkpoint-{step}",
                    repo_id=self.upload_id,
                    repo_type="model",
                    commit_message=f"save checkpoint-{step}",
                )
                print(f"[CheckpointManager] Checkpoint-{step} uploaded to https://huggingface.co/{self.upload_id}")
            except Exception as e:
                print(f"[CheckpointManager] Upload notice for checkpoint-{step}: {e}")

        return ckpt_dir

    def prune_local_checkpoints(self):
        """Keeps only num_latest_checkpoint_save latest checkpoints."""
        local_checkpoints = sorted(
            glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
            key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)) if re.search(r"checkpoint-(\d+)", p) else -1
        )
        if len(local_checkpoints) > self.num_latest_checkpoint_save:
            to_remove = local_checkpoints[:-self.num_latest_checkpoint_save]
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)
