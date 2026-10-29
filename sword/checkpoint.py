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
import random
import tempfile
import threading
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
        self._status_lock = threading.Lock()
        self._pending_uploads = set()
        self._upload_status = {"last_saved_step": None, "last_uploaded_step": None, "upload_error": None}
        pending_file = os.path.join(self.output_dir, "checkpoint_uploads.json")
        if os.path.isfile(pending_file):
            with open(pending_file, encoding="utf-8") as stream:
                pending = json.load(stream)
            if pending.get("repo_id") == self.upload_id:
                self._pending_uploads = {name for name in pending.get("pending", [])
                                        if re.fullmatch(r"checkpoint-\d+", name)}
                self._upload_status["last_uploaded_step"] = pending.get("last_uploaded_step")
                self._upload_status["last_saved_step"] = pending.get("last_saved_step")
                self._upload_status["upload_error"] = pending.get("upload_error")

        self.api = None
        if self.upload_id and HAS_HF_HUB:
            try:
                self.api = HfApi(token=self.hf_token)
                if self.is_save_and_upload:
                    self.api.create_repo(repo_id=self.upload_id, repo_type="model", private=True, exist_ok=True)
            except Exception as e:
                self._upload_status["upload_error"] = str(e)
                print(f"[CheckpointManager] Note: Hugging Face repo setup notice: {e}")
        elif self.upload_id and self.is_save_and_upload:
            self._upload_status["upload_error"] = "huggingface_hub is not installed"

    def upload_info(self):
        with self._status_lock:
            return {"enabled": bool(self.is_save_and_upload and self.upload_id), "repo_id": self.upload_id,
                    **self._upload_status, "pending_uploads": sorted(self._pending_uploads)}

    def _save_upload_status(self):
        with self._status_lock:
            state = {"repo_id": self.upload_id, "pending": sorted(self._pending_uploads),
                     **self._upload_status}
        self._write_json_atomic(os.path.join(self.output_dir, "checkpoint_uploads.json"), state)

    @staticmethod
    def _write_json_atomic(path, value):
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _remove_checkpoint_directory(self, path):
        root, target = os.path.realpath(self.output_dir), os.path.realpath(path)
        if target == root or os.path.commonpath([root, target]) != root:
            raise ValueError("Checkpoint cleanup path escaped output_dir")
        shutil.rmtree(path, ignore_errors=True)

    def _publish_checkpoint(self, temporary, target):
        root = os.path.realpath(self.output_dir)
        for path in (temporary, target):
            resolved = os.path.realpath(path)
            if resolved == root or os.path.commonpath([root, resolved]) != root:
                raise ValueError("Checkpoint publication path escaped output_dir")
        backup = target + ".previous-" + os.urandom(16).hex()
        if os.path.exists(target):
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except BaseException:
            if os.path.exists(backup):
                os.replace(backup, target)
            raise
        if os.path.exists(backup):
            self._remove_checkpoint_directory(backup)

    @staticmethod
    def read_training_state(ckpt_dir):
        path = os.path.join(ckpt_dir, "training_state.json")
        if not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8") as stream:
            state = json.load(stream)
        if not isinstance(state, dict):
            raise ValueError("Invalid checkpoint training state")
        return state

    @staticmethod
    def read_dataset_snapshot(ckpt_dir):
        path = os.path.join(ckpt_dir, "dataset_snapshot.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)

    def restore_random_state(self, ckpt_dir):
        rng_file = os.path.join(ckpt_dir, "rng_state.pt")
        if not os.path.isfile(rng_file):
            return False
        rng = torch.load(rng_file, map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        if "cuda" in rng and torch.cuda.is_available():
            for device, state in enumerate(rng["cuda"][:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state, device=device)
        if "numpy" in rng:
            import numpy as np
            name, keys, position, gaussian, cached = rng["numpy"]
            np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
        return True

    @staticmethod
    def _checkpoint_ready(path):
        return (os.path.isfile(os.path.join(path, "training_state.json")) and
                any(os.path.isfile(os.path.join(path, name))
                    for name in ("adapter_model.safetensors", "adapter_model.bin")))

    def _validate_checkpoint(self, path):
        if not self._checkpoint_ready(path):
            raise RuntimeError(f"Incomplete checkpoint: {path}")
        manifest_path = os.path.join(path, "checkpoint_manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
            root = os.path.realpath(path)
            for name, expected in manifest["files"].items():
                candidate = os.path.realpath(os.path.join(path, name))
                if os.path.commonpath([root, candidate]) != root or not os.path.isfile(candidate):
                    raise RuntimeError(f"Missing checkpoint file: {name}")
                if self.compute_sha256(candidate) != expected:
                    raise RuntimeError(f"Checkpoint checksum mismatch: {name}")
        self.read_training_state(path)

    def restore_completion_log(self, ckpt_dir):
        source = os.path.join(ckpt_dir, "data_complete.logs")
        if not os.path.isfile(source):
            return
        target = os.path.join(self.output_dir, "data_complete.logs")
        entries = self._completion_entries(target)
        state = self.read_training_state(ckpt_dir)
        task_state = self._task_completion_state(state)
        if task_state:
            for key in {*task_state["tasks"], state.get("dataset_sha256")}:
                entries.pop(key, None)
        entries.update(self._completion_entries(source))
        temporary = target + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            stream.writelines(line + "\n" for line in entries.values())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    @staticmethod
    def _completion_entries(path):
        entries = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    value = line.strip()
                    if value and not value.startswith("#"):
                        entries[value.split()[0]] = value
        return entries

    @staticmethod
    def _task_completion_state(state):
        task_state = state.get("task_queue")
        if (isinstance(task_state, dict) and task_state.get("version") == 1
                and isinstance(task_state.get("tasks"), dict)
                and isinstance(task_state.get("config"), dict) and "enabled" in task_state["config"]):
            return task_state
        return {}

    def retry_pending_uploads(self):
        if not (self.is_save_and_upload and self.api and self.upload_id):
            return
        for name in sorted(self._pending_uploads, key=lambda value: int(value.split("-")[-1]), reverse=True):
            directory = os.path.join(self.output_dir, name)
            if not self._checkpoint_ready(directory):
                continue
            uploaded = False
            for attempt in range(3):
                try:
                    self.api.upload_folder(folder_path=directory, path_in_repo=name, repo_id=self.upload_id,
                                           repo_type="model", commit_message=f"save {name}")
                    step = int(name.split("-")[-1])
                    with self._status_lock:
                        self._pending_uploads.discard(name)
                        self._upload_status["last_uploaded_step"] = max(step, self._upload_status["last_uploaded_step"] or 0)
                        self._upload_status["upload_error"] = None
                    print(f"[CheckpointManager] {name} uploaded to https://huggingface.co/{self.upload_id}")
                    uploaded = True
                    break
                except Exception as exc:
                    with self._status_lock:
                        self._upload_status["upload_error"] = str(exc)
                    if attempt < 2:
                        time.sleep(attempt + 1)
                    else:
                        print(f"[CheckpointManager] Upload failed for {name}; retained for retry: {exc}")
            self._save_upload_status()
            if not uploaded:
                # Retry the newest snapshot first; an outage must not multiply
                # training delays by every previously failed upload.
                break

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
        """
        Collects all completed SHA-256 hashes recorded in the authoritative data_complete.logs.
        Only reads the validated master log file in output_dir to prevent aborted runs from corrupting history.
        """
        completed_hashes: Set[str] = set()
        main_log = os.path.join(self.output_dir, "data_complete.logs")
        if os.path.isfile(main_log):
            try:
                with open(main_log, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str and not line_str.startswith("#"):
                            h = line_str.split()[0]
                            if h:
                                completed_hashes.add(h)
            except Exception:
                pass
        return completed_hashes

    def is_dataset_completed(self, dataset_hash: str) -> bool:
        """Checks if dataset has already been trained on."""
        return dataset_hash in self.get_completed_dataset_hashes()

    def mark_dataset_completed(self, dataset_hash: str, dataset_name: str = ""):
        """Appends validated dataset hash to global data_complete.logs."""
        main_log = os.path.join(self.output_dir, "data_complete.logs")
        with open(main_log, "a", encoding="utf-8") as f:
            f.write(f"{dataset_hash}  {dataset_name}\n")
            f.flush()

    def find_latest_checkpoint(self, resume_query: Optional[str] = None) -> Optional[str]:
        """
        Discovers the matching or latest checkpoint-xxx directory locally or on Hugging Face.
        Only resumes if resume_query is explicitly requested (not None or empty).
        """
        if not resume_query or not str(resume_query).strip():
            return None

        clean_query = str(resume_query).strip()
        if clean_query.lower() in ("none", "false", "0"):
            return None

        # 1. Exact path match
        if os.path.isdir(clean_query) and self._checkpoint_ready(clean_query):
            return clean_query

        # 2. Local search in output_dir
        local_ckpts: Dict[int, str] = {}
        for p in glob.glob(os.path.join(self.output_dir, "checkpoint-*")):
            m = re.search(r"checkpoint-(\d+)$", p)
            if m and self._checkpoint_ready(p):
                local_ckpts[int(m.group(1))] = p

        query_step: Optional[int] = None
        clean_name = os.path.basename(clean_query.rstrip("/\\"))
        m_q = re.search(r"(?:checkpoint-)?(\d+)$", clean_name)
        if m_q:
            query_step = int(m_q.group(1))

        # If a specific step was requested and exists locally
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
                        name = m.group(1)
                        if (f"{name}/training_state.json" in files and
                                any(f"{name}/{weight}" in files for weight in ("adapter_model.safetensors", "adapter_model.bin"))):
                            remote_ckpts[step_num] = name

                target_step = query_step if query_step is not None else (max(remote_ckpts.keys()) if remote_ckpts else None)
                local_max_step = max(local_ckpts.keys()) if local_ckpts else -1

                if target_step in remote_ckpts and (query_step is not None or target_step > local_max_step):
                    target_name = remote_ckpts[target_step]
                    local_target = os.path.join(self.output_dir, target_name)
                    download_dir = tempfile.mkdtemp(prefix=".resume-", dir=self.output_dir)
                    try:
                        for f in files:
                            if f.startswith(f"{target_name}/"):
                                hf_hub_download(repo_id=self.upload_id, filename=f, local_dir=download_dir, token=self.hf_token)
                        staged = os.path.join(download_dir, target_name)
                        self._validate_checkpoint(staged)
                        self._publish_checkpoint(staged, local_target)
                    finally:
                        self._remove_checkpoint_directory(download_dir)
                    with self._status_lock:
                        self._upload_status["last_uploaded_step"] = target_step
                        self._upload_status["last_saved_step"] = target_step
                    return local_target
            except Exception as e:
                print(f"[CheckpointManager] Remote checkpoint lookup notice: {e}")

        # If query was 'latest' or 'auto', return newest local checkpoint
        if clean_query.lower() in ("latest", "auto", "true", "1") and local_ckpts:
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
        source_lora_manifest: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
        dataset_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Publish only complete snapshots, then upload and retain failed uploads for retry."""
        target = os.path.join(self.output_dir, f"checkpoint-{step}")
        temporary = tempfile.mkdtemp(prefix=f".checkpoint-{step}-", dir=self.output_dir)
        try:
            self._write_checkpoint(temporary, step, model, optimizer, base_model_name, lora_r, lora_alpha,
                                   target_modules, training_state, reward_logs, completed_dataset_hash,
                                   source_lora_manifest, tokenizer, dataset_snapshot)
            self._publish_checkpoint(temporary, target)
        except BaseException:
            self._remove_checkpoint_directory(temporary)
            raise
        # Completion history advances only after the model/optimizer snapshot
        # has been published. A failed save cannot mark a dataset completed.
        self.restore_completion_log(target)
        with self._status_lock:
            self._upload_status["last_saved_step"] = step
            if self.is_save_and_upload and self.upload_id:
                self._pending_uploads.add(f"checkpoint-{step}")
        if self.is_save_and_upload and self.upload_id:
            self._save_upload_status()
            self.retry_pending_uploads()
        self.prune_local_checkpoints()
        return target

    def _write_checkpoint(self, ckpt_dir, step, model, optimizer, base_model_name, lora_r, lora_alpha,
                          target_modules, training_state, reward_logs, completed_dataset_hash,
                          source_lora_manifest, tokenizer, dataset_snapshot):
        """
        Performs custom checkpoint serialization with read-back verification:
        adapter_model.safetensors, adapter_config.json, training_state.json,
        reward_details.jsonl, data_complete.logs, optimizer.pt, and uploads to Hugging Face.
        """
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
        state = dict(training_state or {})
        state["step"] = step
        state["timestamp"] = time.time()
        with open(os.path.join(ckpt_dir, "training_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        if dataset_snapshot is not None:
            with open(os.path.join(ckpt_dir, "dataset_snapshot.json"), "w", encoding="utf-8") as stream:
                json.dump(dataset_snapshot, stream, ensure_ascii=False)
        if source_lora_manifest is not None:
            from .lora_source import SOURCE_MANIFEST
            with open(os.path.join(ckpt_dir, SOURCE_MANIFEST), "w", encoding="utf-8") as f:
                json.dump(source_lora_manifest, f, indent=2)
        if tokenizer is not None:
            tokenizer.save_pretrained(ckpt_dir)
            if getattr(tokenizer, "chat_template", None) and isinstance(tokenizer.chat_template, str):
                with open(os.path.join(ckpt_dir, "chat_template.jinja"), "w", encoding="utf-8") as f:
                    f.write(tokenizer.chat_template)

        # 4. Save reward details log inside checkpoint folder
        if reward_logs:
            with open(os.path.join(ckpt_dir, "reward_details.jsonl"), "w", encoding="utf-8") as f:
                for entry in reward_logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 5. Assemble completion history inside the unpublished snapshot first.
        main_log = os.path.join(self.output_dir, "data_complete.logs")
        ckpt_log = os.path.join(ckpt_dir, "data_complete.logs")
        entries = self._completion_entries(main_log)
        task_state = self._task_completion_state(state)
        if task_state:
            tasks = task_state["tasks"]
            for key in {*tasks, state.get("dataset_sha256")}:
                entries.pop(key, None)
            visited = set(state.get("visited_row_hashes", []))
            enabled = task_state["config"]["enabled"]
            for key, task in tasks.items():
                if task["solved"] or (not enabled and key in visited):
                    entries[key] = f"{key}  row_{task['problem_id']}"
        if completed_dataset_hash:
            entries[completed_dataset_hash] = completed_dataset_hash
        if entries or task_state or os.path.isfile(main_log):
            with open(ckpt_log, "w", encoding="utf-8") as stream:
                stream.writelines(line + "\n" for line in entries.values())

        # 6. Save optimizer state if provided
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

        rng = {"torch": torch.get_rng_state(), "python": random.getstate()}
        if torch.cuda.is_initialized():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        try:
            import numpy as np
            name, keys, position, gaussian, cached = np.random.get_state()
            rng["numpy"] = (name, keys.tolist(), position, gaussian, cached)
        except ImportError:
            pass
        torch.save(rng, os.path.join(ckpt_dir, "rng_state.pt"))
        files = {}
        for directory, _, names in os.walk(ckpt_dir):
            for name in names:
                path = os.path.join(directory, name)
                files[os.path.relpath(path, ckpt_dir).replace(os.sep, "/")] = self.compute_sha256(path)
        self._write_json_atomic(os.path.join(ckpt_dir, "checkpoint_manifest.json"), {"version": 1, "files": files})

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
        self._validate_checkpoint(ckpt_dir)
        training_state = self.read_training_state(ckpt_dir)
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
        if not matched_keys:
            raise RuntimeError("Resume checkpoint matched no model parameters")

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
                raise RuntimeError("Failed to restore checkpoint optimizer state") from e_opt

        # Restore step from training_state.json
        step = int(training_state.get("step", 0))
        rng_restored = self.restore_random_state(ckpt_dir)
        self.restore_completion_log(ckpt_dir)

        return {
            "step": step,
            "matched_keys": len(matched_keys),
            "total_keys": len(sd),
            "optimizer_restored": opt_restored,
            "rng_restored": rng_restored,
            "training_state": training_state,
            "ckpt_dir": ckpt_dir,
        }

    def prune_local_checkpoints(self):
        """Keeps only num_latest_checkpoint_save latest checkpoints."""
        local_checkpoints = sorted(
            [path for path in glob.glob(os.path.join(self.output_dir, "checkpoint-*"))
             if re.fullmatch(r"checkpoint-\d+", os.path.basename(path)) and self._checkpoint_ready(path)],
            key=lambda path: int(os.path.basename(path).split("-")[-1])
        )
        if len(local_checkpoints) > self.num_latest_checkpoint_save:
            to_remove = local_checkpoints[:-self.num_latest_checkpoint_save]
            for p in to_remove:
                if os.path.basename(p) not in self._pending_uploads:
                    self._remove_checkpoint_directory(p)
