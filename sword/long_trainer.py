"""
Infinite GRPO Trainer Engine:
Coordinates continuous dataset loading, SHA-256 deduplication, Model B Ollama evaluation,
agentic harness tool dispatch, custom checkpointing, OOM auto-recovery, and Hugging Face uploads.
"""

import os
import gc
import re
import json
import time
import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn

try:
    import unsloth
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except Exception:
    HAS_UNSLOTH = False

from transformers import AutoTokenizer, AutoModelForCausalLM

from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.harness import get_harness, HarnessResult
from sword.reward import MultiMetricRewardScorer, RewardBreakdown
from sword.checkpoint import CheckpointManager
from sword.grpo_bridge import GRPOBridge, GRPOBridgeConfig, RolloutBatch, RolloutItem


@dataclass
class DatasetProblemRow:
    """Normalized representation of a dataset problem entry."""
    problem: str
    solution: str
    reasoning_rule: str = ""
    answer_rule: str = ""
    can_be_verify: bool = True
    difficulty: str = "medium"
    category: str = "general"
    harness: Optional[str] = None
    problem_id: str = ""


def parse_dataset_row(raw: Dict[str, Any], index: int = 0) -> DatasetProblemRow:
    """Defensively parses any dataset dictionary into a standardized DatasetProblemRow."""
    problem = str(raw.get("problem") or raw.get("question") or raw.get("user") or raw.get("prompt") or "").strip()
    solution = str(raw.get("solution") or raw.get("answer") or raw.get("assistant") or raw.get("model") or "").strip()
    reasoning_rule = str(raw.get("reasoning_rule") or raw.get("reasoning") or "").strip()
    answer_rule = str(raw.get("answer_rule") or "").strip()

    # Handle spelling variations in dataset schemas: can_be__verify or can_be_verify
    cbv = raw.get("can_be__verify") if "can_be__verify" in raw else raw.get("can_be_verify")
    if cbv is None:
        # Default True if solution is numeric or short, else False
        can_be_verify = bool(re.search(r"^\s*[-+]?(?:\d*\.\d+|\d+)\s*$", solution))
    elif isinstance(cbv, str):
        can_be_verify = cbv.strip().lower() in ("true", "1", "yes")
    else:
        can_be_verify = bool(cbv)

    difficulty = str(raw.get("difficult") or raw.get("difficulty") or "medium").strip().lower()
    category = str(raw.get("cagetory") or raw.get("category") or "general").strip().lower()
    harness = str(raw.get("harness") or "").strip().lower() or None

    return DatasetProblemRow(
        problem=problem,
        solution=solution,
        reasoning_rule=reasoning_rule,
        answer_rule=answer_rule,
        can_be_verify=can_be_verify,
        difficulty=difficulty,
        category=category,
        harness=harness,
        problem_id=str(raw.get("id") or f"problem_{index}"),
    )


def load_dataset_records(dataset_spec: Union[str, List[Any], Dict[str, Any]], hf_token: Optional[str] = None) -> Tuple[List[DatasetProblemRow], str]:
    """
    Loads dataset from Google Drive link, Hugging Face dataset ID, local file, or in-memory list.
    Returns (records, sha256_hash).
    """
    raw_items = []
    content_bytes = b""

    # 1. In-memory list or dictionary
    if isinstance(dataset_spec, list):
        raw_items = dataset_spec
        content_bytes = json.dumps(dataset_spec, sort_keys=True).encode("utf-8")
    elif isinstance(dataset_spec, dict) and "data" in dataset_spec:
        raw_items = dataset_spec["data"]
        content_bytes = json.dumps(dataset_spec, sort_keys=True).encode("utf-8")

    # 2. String path, URL, or Hugging Face dataset ID
    elif isinstance(dataset_spec, str):
        spec = dataset_spec.strip()

        # Google Drive URL
        if "drive.google.com" in spec:
            from sword.trainer import download_from_drive
            dest = os.path.join("datasets", "drive_dataset.jsonl")
            os.makedirs("datasets", exist_ok=True)
            download_from_drive(spec, dest)
            spec = dest

        # Local file (json, jsonl, csv)
        if os.path.isfile(spec):
            with open(spec, "rb") as f:
                content_bytes = f.read()

            if spec.endswith(".jsonl"):
                for line in content_bytes.decode("utf-8", errors="replace").splitlines():
                    if line.strip():
                        try:
                            raw_items.append(json.loads(line))
                        except Exception:
                            pass
            elif spec.endswith(".json"):
                data = json.loads(content_bytes.decode("utf-8", errors="replace"))
                raw_items = data if isinstance(data, list) else [data]
            else:
                # Delimited / CSV fallback
                for line in content_bytes.decode("utf-8", errors="replace").splitlines():
                    if line.strip():
                        parts = line.split("\t") if "\t" in line else line.split(",")
                        if len(parts) >= 2:
                            raw_items.append({"problem": parts[0], "solution": parts[1]})

        # Hugging Face Hub dataset ID
        else:
            content_bytes = spec.encode("utf-8")
            try:
                from datasets import load_dataset
                ds = load_dataset(spec, split="train", token=hf_token)
                for item in ds:
                    raw_items.append(dict(item))
            except Exception as e:
                print(f"[DatasetLoader] Hub load note ({e}). Creating synthetic problem records.")
                raw_items = [
                    {
                        "problem": f"Sample problem {i}: Compute 2^{i} + {i * 3}",
                        "solution": str(2**i + i * 3),
                        "reasoning_rule": "Derive step by step.",
                        "answer_rule": "Provide exact integer answer.",
                        "can_be__verify": True,
                        "difficult": "easy" if i < 3 else "medium",
                        "cagetory": "math",
                        "harness": "none",
                    }
                    for i in range(1, 9)
                ]

    # Compute SHA-256
    sha256 = hashlib.sha256(content_bytes if content_bytes else b"empty").hexdigest()

    records = [parse_dataset_row(item, idx) for idx, item in enumerate(raw_items)]
    return records, sha256


def run_long_train(
    dataset: Union[str, List[Any], Dict[str, Any]] = "math-ai/TemplateGSM",
    model: str = "Qwen/Qwen3.5-0.8B",
    B_model_port: int = 11343,
    custom_checkpoint_save: bool = True,
    is_save_and_upload_checkpoint: bool = True,
    num_latest_checkpoint_save: int = 2,
    upload_id: Optional[str] = None,
    HF_TOKEN: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    lora_cfg: Optional[Dict[str, Any]] = None,
    Argument: Optional[Dict[str, Any]] = None,
    show_detail_logs: bool = True,
    score_logs: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Main long-running infinite GRPO training entry point.
    Coordinates continuous dataset ingestion, SHA-256 tracking, Model B scoring,
    agentic harnesses, custom checkpointing, and OOM auto-recovery.
    """
    args = Argument or {}
    lora_conf = lora_cfg or {}

    steps = int(args.get("steps") or args.get("max_steps") or 200)
    lr = float(args.get("lr") or args.get("learning_rate") or 5e-6)
    concurrency_g = int(args.get("concurrency_g") or args.get("num_rollouts") or 4)
    save_steps = int(args.get("save_steps") or 25)
    micro_batch_size = int(args.get("micro_batch_size") or 2)
    max_seq_len = int(args.get("max_seq_len") or 2048)
    max_new_tokens = int(args.get("max_new_tokens") or 512)
    output_dir = str(args.get("output_dir") or "checkpoints")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(" SWORD INFINITE GRPO TRAINING ENGINE")
    print("=" * 72)
    print(f" Target Model:            {model}")
    print(f" Model B Judge Port:      {B_model_port}")
    print(f" Concurrency (G):         {concurrency_g} rollouts/problem")
    print(f" Steps Planned:           {steps} (save every {save_steps} steps)")
    print(f" Device:                  {device}")
    print(f" Upload Repo ID:          {upload_id or 'Local only'}")
    print(f" Checkpoint Saver:        Custom={custom_checkpoint_save} (Retain latest {num_latest_checkpoint_save})")
    print("=" * 72)

    # 1. Initialize Checkpoint Manager
    ckpt_mgr = CheckpointManager(
        output_dir=output_dir,
        num_latest_checkpoint_save=num_latest_checkpoint_save,
        upload_id=upload_id,
        hf_token=HF_TOKEN,
        is_save_and_upload_checkpoint=is_save_and_upload_checkpoint,
    )

    # 2. Ingest Dataset & Check SHA-256 Deduplication
    records, dataset_sha256 = load_dataset_records(dataset, hf_token=HF_TOKEN)
    print(f"[*] Dataset SHA-256: {dataset_sha256} ({len(records)} problem rows)")

    if ckpt_mgr.is_dataset_completed(dataset_sha256):
        print(f"\n[Sword] The data is already train on, suggest prepare new dataset to continues.")
        return {
            "status": "already_trained",
            "dataset_sha256": dataset_sha256,
            "steps_completed": 0,
        }

    # 3. Model B Judge & Scorer Initialization
    b_judge_config = ModelBJudgeConfig(
        port=B_model_port,
        host=kwargs.get("B_model_host", "127.0.0.1"),
        model_name=kwargs.get("B_model_name", "gpt-oss:120b"),
    )
    b_judge = ModelBJudge(config=b_judge_config)
    reward_scorer = MultiMetricRewardScorer(b_model_judge=b_judge)

    # 4. GRPO Bridge Setup
    bridge_config = GRPOBridgeConfig(
        model_name=model,
        concurrency_g=concurrency_g,
        max_model_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        micro_batch_size=micro_batch_size,
        lora_r=int(lora_conf.get("r", 16)),
        lora_alpha=int(lora_conf.get("lora_alpha", 32)),
        lora_dropout=float(lora_conf.get("lora_dropout", 0.0)),
        target_modules=lora_conf.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        device=device,
    )
    bridge = GRPOBridge(config=bridge_config)
    bridge.initialize_trainer(
        model=kwargs.get("trainer_model"),
        tokenizer=kwargs.get("tokenizer"),
    )

    # 5. Checkpoint Resume Discovery
    start_step = 0
    resolved_ckpt = ckpt_mgr.find_latest_checkpoint(resume_from_checkpoint)
    if resolved_ckpt:
        print(f"[*] Resuming from checkpoint: {resolved_ckpt}")
        try:
            from safetensors.torch import load_file as load_safetensors
            weights_file = os.path.join(resolved_ckpt, "adapter_model.safetensors")
            if not os.path.exists(weights_file):
                weights_file = os.path.join(resolved_ckpt, "adapter_model.bin")

            if os.path.exists(weights_file):
                sd = load_safetensors(weights_file) if weights_file.endswith(".safetensors") else torch.load(weights_file, map_location="cpu")
                bridge.trainer_model.load_state_dict(sd, strict=False)
                print(f"[*] Restored {len(sd)} LoRA adapter weights.")

            state_file = os.path.join(resolved_ckpt, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    st = json.load(f)
                    start_step = int(st.get("step", 0))
                    print(f"[*] Resumed at step: {start_step}")
        except Exception as e_res:
            print(f"[Sword] Resume note: {e_res}. Continuing with current parameters.")

    # 6. Initialize vLLM / Fallback Engine
    bridge.initialize_vllm()

    trainable_params = [p for p in bridge.trainer_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    reward_history_logs: List[Dict[str, Any]] = []
    current_step = start_step

    print(f"\n[*] Starting training loop from step {start_step + 1} to {start_step + steps}...\n")

    try:
        while current_step < start_step + steps:
            step_idx = current_step + 1
            row_idx = (step_idx - 1) % len(records)
            current_row = records[row_idx]

            # Determine whether this task uses an agentic CLI harness
            harness = get_harness(current_row.harness)

            # Generate rollouts
            prompts = [current_row.problem]
            batch = bridge.generate_rollouts(prompts)

            # Score each rollout with multi-metric scorer
            step_scores = []
            for item in batch.items:
                tools_called = 0
                tools_succeeded = 0

                # If harness is requested, run harness validation
                if harness is not None:
                    h_res = harness.run(
                        prompt=item.prompt_text,
                        model_generate_fn=lambda p: item.response_text,
                        task_id=current_row.problem_id,
                    )
                    tools_called = h_res.tools_called
                    tools_succeeded = h_res.tools_succeeded

                score_breakdown = reward_scorer.score(
                    problem=current_row.problem,
                    solution=current_row.solution,
                    candidate_response=item.response_text,
                    reasoning_rule=current_row.reasoning_rule,
                    answer_rule=current_row.answer_rule,
                    can_be_verify=current_row.can_be_verify,
                    difficulty=current_row.difficulty,
                    category=current_row.category,
                    harness_name=current_row.harness,
                    tools_called=tools_called,
                    tools_succeeded=tools_succeeded,
                )
                item.reward = score_breakdown.total_score
                step_scores.append(score_breakdown)

                if score_logs:
                    reward_history_logs.append({
                        "step": step_idx,
                        "problem_id": current_row.problem_id,
                        "scores": score_breakdown.to_dict(),
                    })

            # Calculate group-centered advantages
            groups: Dict[int, List[RolloutItem]] = {}
            for it in batch.items:
                groups.setdefault(it.group_id, []).append(it)

            for g_id, g_items in groups.items():
                r_vals = [it.reward for it in g_items]
                mean_r = sum(r_vals) / len(r_vals)
                var_r = sum((r - mean_r) ** 2 for r in r_vals) / max(len(r_vals), 1)
                std_r = math.sqrt(var_r)
                for it in g_items:
                    it.advantage = (it.reward - mean_r) / (std_r + 1e-6) if std_r > 1e-6 else 0.0

            # Execute training step with OOM auto-recovery
            try:
                metrics = bridge.train_step(batch, optimizer)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e_oom:
                if "out of memory" in str(e_oom).lower():
                    print(f"\n[Sword Alert] CUDA Out-of-Memory intercepted at step {step_idx}!")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Save emergency checkpoint of the latest completed state
                    emergency_ckpt = ckpt_mgr.save_checkpoint(
                        step=current_step,
                        model=bridge.trainer_model,
                        optimizer=optimizer,
                        base_model_name=model,
                        reward_logs=reward_history_logs[-50:],
                        completed_dataset_hash=dataset_sha256,
                    )
                    print(f"[Sword] Emergency checkpoint successfully secured at: {emergency_ckpt}")
                    return {
                        "status": "oom_aborted",
                        "last_completed_step": current_step,
                        "emergency_checkpoint": emergency_ckpt,
                    }
                raise e_oom

            current_step = step_idx

            # Logging
            if show_detail_logs:
                avg_reward = sum(s.total_score for s in step_scores) / max(1, len(step_scores))
                avg_truth = sum(s.truth for s in step_scores) / max(1, len(step_scores))
                print(
                    f"Step {step_idx:04d}/{start_step + steps:04d} | "
                    f"Loss: {metrics.get('grpo_loss', 0.0):.4f} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"Truth: {avg_truth:.2f} | "
                    f"Harness: {current_row.harness or 'none'} | "
                    f"Diff: {current_row.difficulty}"
                )

            # Periodic checkpoint saving
            if current_step % save_steps == 0:
                saved_path = ckpt_mgr.save_checkpoint(
                    step=current_step,
                    model=bridge.trainer_model,
                    optimizer=optimizer,
                    base_model_name=model,
                    lora_r=bridge_config.lora_r,
                    lora_alpha=bridge_config.lora_alpha,
                    target_modules=bridge_config.target_modules,
                    reward_logs=reward_history_logs[-100:],
                )
                print(f"[*] Checkpoint saved at: {saved_path}")

    except KeyboardInterrupt:
        print("\n[Sword] Training interrupted by user. Saving current checkpoint...")
    finally:
        # Mark dataset completed and save final state
        ckpt_mgr.mark_dataset_completed(dataset_sha256, str(dataset)[:50])
        final_ckpt = ckpt_mgr.save_checkpoint(
            step=current_step,
            model=bridge.trainer_model,
            optimizer=optimizer,
            base_model_name=model,
            lora_r=bridge_config.lora_r,
            lora_alpha=bridge_config.lora_alpha,
            target_modules=bridge_config.target_modules,
            reward_logs=reward_history_logs,
            completed_dataset_hash=dataset_sha256,
        )
        print(f"\n[Sword] Training finished. Final checkpoint: {final_ckpt}")

    return {
        "status": "completed",
        "final_step": current_step,
        "dataset_sha256": dataset_sha256,
        "checkpoint_dir": final_ckpt,
    }
