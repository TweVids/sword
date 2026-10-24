"""
Custom Checkpoint Manager: Independent serialization, SHA-256 dataset tracking,
checkpoint resume discovery, read-back verification, and Hugging Face Hub synchronization.
"""

import os
import re
import json
import time
import glob
import shutil
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn

try:
    from huggingface_hub import HfApi, hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


class CheckpointManager:
    """
    Manages custom checkpoint saving, read-back verification, automatic repo creation,
    resume discovery, SHA-256 dataset deduplication logs, and Hugging Face uploads.
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

    def get_completed_dataset_hashes(self) -> Set[str]:
        """Collects all SHA-256 hashes recorded in local data_complete.logs."""
        completed_hashes: Set[str] = set()
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
        Compares step numbers across local and remote repositories to select the newest state.
        """
        # 1. Exact path match
        if resume_query and os.path.isdir(resume_query):
            return resume_query

        # 2. Local search in output_dir
        local_ckpts: Dict[int, str] = {}
        for p in glob.glob(os.path.join(self.output_dir, "checkpoint-*")):
            m = re.search(r"checkpoint-(\d+)$", p)
            if m:
                local_ckpts[int(m.group(1))] = p

        query_step: Optional[int] = None
        if resume_query:
            clean_q = os.path.basename(resume_query.rstrip("/\\"))
            m_q = re.search(r"(?:checkpoint-)?(\d+)$", clean_q)
            if m_q:
                query_step = int(m_q.group(1))

        # If a specific step query was given and found locally
        if query_step is not None and query_step in local_ckpts:
            return local_ckpts[query_step]

        # 3. Remote Hugging Face search if enabled
        remote_ckpts: Dict[int, str] = {}
        if self.api and self.upload_id:
            try:
                files = self.api.list_repo_files(repo_id=self.upload_id, repo_type="model")
                for f in files:
                    m = re.match(r"(checkpoint-(\d+))/", f)
                    if m:
                        step_num = int(m.group(2))
                        remote_ckpts[step_num] = m.group(1)

                # If query requested a specific step available remotely
                target_step = query_step if query_step is not None else (max(remote_ckpts.keys()) if remote_ckpts else None)

                # If remote has a newer step than local, or matches query
                local_max_step = max(local_ckpts.keys()) if local_ckpts else -1
                if target_step is not None and (query_step is not None or target_step > local_max_step):
                    target_name = remote_ckpts[target_step]
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

        # Return latest local checkpoint if available
        if local_ckpts:
            latest_local_step = max(local_ckpts.keys())
            return local_ckpts[latest_local_step]

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
        Performs custom checkpoint serialization with read-back verification:
        adapter_model.safetensors, adapter_config.json, training_state.json,
        reward_details.jsonl, data_complete.logs, optimizer.pt, and uploads to Hugging Face.
        """
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. Save LoRA adapter weights
        peft_saved = False
        if hasattr(model, "save_pretrained") and hasattr(model, "peft_config"):
            try:
                model.save_pretrained(ckpt_dir)
                peft_saved = True
            except Exception as e_peft:
                print(f"[CheckpointManager] model.save_pretrained notice ({e_peft}). Proceeding to direct tensor serialization.")

        # Ensure adapter_config.json exists with base_model_name_or_path populated
        cfg_path = os.path.join(ckpt_dir, "adapter_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    peft_cfg = json.load(f)
                if not peft_cfg.get("base_model_name_or_path"):
                    peft_cfg["base_model_name_or_path"] = base_model_name
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(peft_cfg, f, indent=2)
            except Exception:
                pass
        else:
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
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(peft_cfg, f, indent=2)

        # If save_pretrained didn't write weights, serialize trainable/lora parameters directly
        weights_file = os.path.join(ckpt_dir, "adapter_model.safetensors")
        weights_bin = os.path.join(ckpt_dir, "adapter_model.bin")

        if not (os.path.exists(weights_file) or os.path.exists(weights_bin)):
            lora_sd = {}
            for name, param in model.named_parameters():
                if ("lora_" in name or param.requires_grad):
                    lora_sd[name] = param.detach().cpu()

            try:
                from safetensors.torch import save_file
                save_file(lora_sd, weights_file)
            except Exception:
                torch.save(lora_sd, weights_bin)

        # 2. Read-Back Verification: Confirm weights file exists and is readable
        self.verify_saved_weights(ckpt_dir)

        # 3. Save training state
        state = training_state or {}
        state["step"] = step
        state["timestamp"] = time.time()
        with open(os.path.join(ckpt_dir, "training_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # 4. Save reward details log inside checkpoint folder
        if reward_logs:
            with open(os.path.join(ckpt_dir, "reward_details.jsonl"), "w", encoding="utf-8") as f:
                for entry in reward_logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 5. Copy / update data_complete.logs into checkpoint
        main_log = os.path.join(self.output_dir, "data_complete.logs")
        ckpt_log = os.path.join(ckpt_dir, "data_complete.logs")
        if os.path.exists(main_log):
            shutil.copy2(main_log, ckpt_log)
        elif completed_dataset_hash:
            with open(ckpt_log, "w", encoding="utf-8") as f:
                f.write(f"{completed_dataset_hash}\n")

        # 6. Save optimizer state if provided
        if optimizer is not None:
            try:
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
            except Exception as e_opt:
                print(f"[CheckpointManager] Optimizer serialization notice: {e_opt}")

        # 7. Prune older local checkpoints to respect num_latest_checkpoint_save
        self.prune_local_checkpoints()

        # 8. Upload checkpoint to Hugging Face if enabled
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

    def verify_saved_weights(self, ckpt_dir: str):
        """Validates that saved weights file can be loaded back into memory."""
        weights_file = os.path.join(ckpt_dir, "adapter_model.safetensors")
        if not os.path.exists(weights_file):
            weights_file = os.path.join(ckpt_dir, "adapter_model.bin")

        if not os.path.exists(weights_file):
            raise RuntimeError(f"Checkpoint verification failed: No adapter weights found in {ckpt_dir}")

        try:
            if weights_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                loaded = load_file(weights_file)
            else:
                loaded = torch.load(weights_file, map_location="cpu")

            if not loaded or len(loaded) == 0:
                raise RuntimeError(f"Checkpoint verification failed: Saved weights file in {ckpt_dir} is empty.")
        except Exception as e:
            raise RuntimeError(f"Checkpoint verification read-back failed for {ckpt_dir}: {e}")

    def load_checkpoint(
        self,
        ckpt_dir: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Restores model adapter weights and optimizer state from checkpoint.
        Verifies key matching and reports restoration metrics.
        """
        weights_file = os.path.join(ckpt_dir, "adapter_model.safetensors")
        if not os.path.exists(weights_file):
            weights_file = os.path.join(ckpt_dir, "adapter_model.bin")

        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"No weights file found in {ckpt_dir}")

        if weights_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(weights_file)
        else:
            sd = torch.load(weights_file, map_location="cpu")

        # Key prefix and adapter naming adaptation
        matched_keys: List[str] = []

        # 1. If PEFT model, use canonical set_peft_model_state_dict
        if hasattr(model, "peft_config"):
            try:
                from peft import set_peft_model_state_dict
                res = set_peft_model_state_dict(model, sd)
                unexpected = getattr(res, "unexpected_keys", [])
                if len(unexpected) < len(sd):
                    matched_keys = [k for k in sd.keys() if k not in unexpected]
            except Exception as e_peft:
                print(f"[CheckpointManager] set_peft_model_state_dict note ({e_peft}). Falling back to manual key mapping.")

        # 2. If not a PEFT model or set_peft did not match, perform manual key adaptation
        if not matched_keys:
            model_keys = set(model.state_dict().keys())
            candidate_sd = {}

            for k, v in sd.items():
                if k in model_keys:
                    candidate_sd[k] = v
                    continue

                # Adaptation A: inject .default. adapter suffix
                alt_k = k
                if ".lora_A.weight" in alt_k:
                    alt_k = alt_k.replace(".lora_A.weight", ".lora_A.default.weight")
                elif ".lora_B.weight" in alt_k:
                    alt_k = alt_k.replace(".lora_B.weight", ".lora_B.default.weight")

                if alt_k in model_keys:
                    candidate_sd[alt_k] = v
                    continue

                # Adaptation B: strip or add base_model.model. prefix
                if alt_k.startswith("base_model.model."):
                    alt_k2 = alt_k.replace("base_model.model.", "")
                else:
                    alt_k2 = f"base_model.model.{alt_k}"

                if alt_k2 in model_keys:
                    candidate_sd[alt_k2] = v

            if candidate_sd:
                model.load_state_dict(candidate_sd, strict=False)
                matched_keys = list(candidate_sd.keys())

        print(f"[CheckpointManager] Restored {len(matched_keys)} / {len(sd)} adapter weight tensors.")

        # Restore optimizer if present
        opt_restored = False
        opt_file = os.path.join(ckpt_dir, "optimizer.pt")
        if optimizer is not None and os.path.exists(opt_file):
            try:
                opt_sd = torch.load(opt_file, map_location="cpu")
                optimizer.load_state_dict(opt_sd)
                opt_restored = True
                print("[CheckpointManager] Successfully restored optimizer state.")
            except Exception as e_opt:
                print(f"[CheckpointManager] Optimizer state restoration notice: {e_opt}")

        # Restore step from training_state.json
        step = 0
        state_file = os.path.join(ckpt_dir, "training_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    st = json.load(f)
                    step = int(st.get("step", 0))
            except Exception:
                pass

        return {
            "step": step,
            "matched_keys": len(matched_keys),
            "total_keys": len(sd),
            "optimizer_restored": opt_restored,
            "ckpt_dir": ckpt_dir,
        }

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
