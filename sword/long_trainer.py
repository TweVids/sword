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
from sword.harness import get_harness, HarnessResult, WorkspaceSandbox
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
    row_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "solution": self.solution,
            "reasoning_rule": self.reasoning_rule,
            "answer_rule": self.answer_rule,
            "can_be_verify": self.can_be_verify,
            "difficulty": self.difficulty,
            "category": self.category,
            "harness": self.harness,
            "problem_id": self.problem_id,
            "row_hash": self.row_hash,
        }


def parse_dataset_row(raw: Dict[str, Any], index: int = 0) -> DatasetProblemRow:
    """Defensively parses any dataset dictionary into a standardized DatasetProblemRow."""
    problem = str(raw.get("problem") or raw.get("question") or raw.get("user") or raw.get("prompt") or "").strip()
    solution = str(raw.get("solution") or raw.get("answer") or raw.get("assistant") or raw.get("model") or "").strip()
    reasoning_rule = str(raw.get("reasoning_rule") or raw.get("reasoning") or "").strip()
    answer_rule = str(raw.get("answer_rule") or "").strip()

    # Handle spelling variations in dataset schemas: can_be__verify or can_be_verify
    cbv = raw.get("can_be__verify") if "can_be__verify" in raw else raw.get("can_be_verify")
    if cbv is None:
        can_be_verify = bool(re.search(r"^\s*[-+]?(?:\d*\.\d+|\d+)\s*$", solution))
    elif isinstance(cbv, str):
        can_be_verify = cbv.strip().lower() in ("true", "1", "yes")
    else:
        can_be_verify = bool(cbv)

    difficulty = str(raw.get("difficult") or raw.get("difficulty") or "medium").strip().lower()
    category = str(raw.get("cagetory") or raw.get("category") or "general").strip().lower()
    harness = str(raw.get("harness") or "").strip().lower() or None

    # Deterministic row hash for expandable deduplication
    row_dict = {
        "problem": problem,
        "solution": solution,
        "reasoning_rule": reasoning_rule,
        "answer_rule": answer_rule,
        "can_be_verify": can_be_verify,
        "difficulty": difficulty,
        "category": category,
        "harness": harness,
    }
    row_hash = hashlib.sha256(json.dumps(row_dict, sort_keys=True).encode("utf-8")).hexdigest()[:16]

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
        row_hash=row_hash,
    )


def load_dataset_records(dataset_spec: Union[str, List[Any], Dict[str, Any]], hf_token: Optional[str] = None) -> Tuple[List[DatasetProblemRow], str]:
    """
    Loads dataset from Google Drive link, Hugging Face dataset ID, local file, or in-memory list.
    Computes exact SHA-256 over canonical serialized records content.
    Returns (records, sha256_hash).
    """
    raw_items = []

    # 1. In-memory list or dictionary
    if isinstance(dataset_spec, list):
        raw_items = dataset_spec
    elif isinstance(dataset_spec, dict) and "data" in dataset_spec:
        raw_items = dataset_spec["data"]
    elif isinstance(dataset_spec, dict):
        raw_items = [dataset_spec]

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
                for line in content_bytes.decode("utf-8", errors="replace").splitlines():
                    if line.strip():
                        parts = line.split("\t") if "\t" in line else line.split(",")
                        if len(parts) >= 2:
                            raw_items.append({"problem": parts[0], "solution": parts[1]})

        # Hugging Face Hub dataset ID
        else:
            try:
                from datasets import load_dataset
                ds = load_dataset(spec, split="train", token=hf_token)
                for item in ds:
                    raw_items.append(dict(item))
            except Exception as e:
                # If explicit test or mock dataset name
                if any(k in spec.lower() for k in ("test", "dummy", "synthetic", "mock", "templategsm")):
                    print(f"[DatasetLoader] Hub load notice ({e}). Using synthetic problem records for testing.")
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
                else:
                    raise ValueError(f"Failed to load dataset '{spec}' from Hugging Face Hub: {e}")

    records = [parse_dataset_row(item, idx) for idx, item in enumerate(raw_items)]

    # Compute exact SHA-256 over parsed records content
    canonical_str = json.dumps([r.to_dict() for r in records], sort_keys=True)
    sha256 = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

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
    auto_unload_B_model: bool = True,
    B_model_keep_alive: str = "5m",
    **kwargs,
) -> Dict[str, Any]:
    """
    Main long-running infinite GRPO training entry point.
    Coordinates continuous dataset ingestion, expandable SHA-256 tracking, Model B scoring,
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

    # 2. Ingest Dataset & Check Expandable SHA-256 Deduplication
    records, dataset_sha256 = load_dataset_records(dataset, hf_token=HF_TOKEN)
    print(f"[*] Dataset SHA-256: {dataset_sha256} ({len(records)} problem rows)")

    completed_hashes = ckpt_mgr.get_completed_dataset_hashes()

    # If entire dataset SHA-256 matches completed log
    if dataset_sha256 in completed_hashes:
        print(f"\n[Sword] The data is already train on, suggest prepare new dataset to continues.")
        return {
            "status": "already_trained",
            "dataset_sha256": dataset_sha256,
            "steps_completed": 0,
        }

    # Expandable dataset: filter out already-completed rows
    uncompleted_records = [r for r in records if r.row_hash not in completed_hashes]
    if not uncompleted_records:
        ckpt_mgr.mark_dataset_completed(dataset_sha256, str(dataset)[:50])
        print(f"\n[Sword] The data is already train on, suggest prepare new dataset to continues.")
        return {
            "status": "already_trained",
            "dataset_sha256": dataset_sha256,
            "steps_completed": 0,
        }

    if len(uncompleted_records) < len(records):
        skipped = len(records) - len(uncompleted_records)
        print(f"[*] Expandable dataset detected: skipping {skipped} already-trained rows. Continuing with {len(uncompleted_records)} new rows.")
        records = uncompleted_records

    # 3. Model B Judge & Scorer Initialization
    b_judge_config = ModelBJudgeConfig(
        port=B_model_port,
        host=kwargs.get("B_model_host", "127.0.0.1"),
        model_name=kwargs.get("B_model_name", "gpt-oss:120b"),
        num_predict=int(kwargs.get("B_model_num_predict", 1024)),
        auto_unload=auto_unload_B_model,
        keep_alive_during_eval=B_model_keep_alive,
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

    trainable_params = [p for p in bridge.trainer_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # 5. Checkpoint Resume Discovery (only when explicitly requested)
    start_step = 0
    resolved_ckpt = ckpt_mgr.find_latest_checkpoint(resume_from_checkpoint)
    if resolved_ckpt:
        print(f"[*] Resuming from checkpoint: {resolved_ckpt}")
        try:
            res_info = ckpt_mgr.load_checkpoint(
                ckpt_dir=resolved_ckpt,
                model=bridge.trainer_model,
                optimizer=optimizer,
            )
            start_step = res_info["step"]
            print(f"[*] Resumed at step {start_step} ({res_info['matched_keys']} adapter keys matched).")
        except Exception as e_res:
            print(f"[Sword] Resume notice: {e_res}. Continuing with fresh state.")

    # 6. Initialize vLLM / Fallback Engine
    bridge.initialize_vllm()

    reward_history_logs: List[Dict[str, Any]] = []
    visited_row_hashes = set()
    current_step = start_step
    completed_successfully = False

    print(f"\n[*] Starting training loop from step {start_step + 1} to {start_step + steps}...\n")

    try:
        while current_step < start_step + steps:
            step_idx = current_step + 1
            row_idx = (step_idx - 1) % len(records)
            current_row = records[row_idx]
            harness = get_harness(current_row.harness)

            step_success = False
            log_start_len = len(reward_history_logs)

            for attempt in range(2):
                # Clean up reward logs from previous failed attempt to avoid duplicates
                if attempt > 0 and len(reward_history_logs) > log_start_len:
                    del reward_history_logs[log_start_len:]

                try:
                    # Rollout Phase with authentic harness prompt injection and token-budget guard
                    prompt_token_budget = max(128, max_seq_len - max_new_tokens)
                    rollout_prompt = (
                        harness.format_prompt(
                            current_row.problem,
                            max_tokens=prompt_token_budget,
                            tokenizer=getattr(bridge, "tokenizer", None),
                        )
                        if harness is not None
                        else current_row.problem
                    )
                    prompts = [rollout_prompt]
                    batch = bridge.generate_rollouts(prompts)

                    # Scoring Phase with Model B memory preparation
                    b_judge.prepare_for_eval()
                    step_scores = []

                    for item in batch.items:
                        tools_called = 0
                        tools_succeeded = 0

                        # Multi-turn harness interaction
                        if harness is not None:
                            def multi_turn_generate(turn_p: str) -> str:
                                if turn_p == item.prompt_text:
                                    return item.response_text
                                return bridge.generate_single(turn_p)

                            h_res = harness.run(
                                prompt=item.prompt_text,
                                model_generate_fn=multi_turn_generate,
                                task_id=current_row.problem_id,
                            )
                            tools_called = h_res.tools_called
                            tools_succeeded = h_res.tools_succeeded

                            # Synchronize tokens and logprobs with updated multi-turn response text
                            if h_res.final_text and h_res.final_text != item.response_text:
                                item.response_text = h_res.final_text
                                if hasattr(bridge, "tokenizer") and bridge.tokenizer is not None:
                                    try:
                                        tok_out = bridge.tokenizer(item.response_text, add_special_tokens=False)
                                        r_ids = tok_out.get("input_ids", []) if isinstance(tok_out, dict) else getattr(tok_out, "input_ids", [])
                                        if isinstance(r_ids, torch.Tensor):
                                            r_ids = r_ids.squeeze().tolist()
                                        if isinstance(r_ids, int):
                                            r_ids = [r_ids]
                                        item.response_token_ids = list(r_ids)
                                        if len(item.rollout_logprobs) < len(item.response_token_ids):
                                            item.rollout_logprobs.extend([0.0] * (len(item.response_token_ids) - len(item.rollout_logprobs)))
                                        elif len(item.rollout_logprobs) > len(item.response_token_ids):
                                            item.rollout_logprobs = item.rollout_logprobs[:len(item.response_token_ids)]
                                    except Exception:
                                        pass

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

                    # Release Model B VRAM before training forward / backward pass
                    b_judge.release_after_eval()

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

                    # Execute training forward / backward step
                    metrics = bridge.train_step(batch, optimizer)
                    step_success = True
                    break

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e_oom:
                    is_oom = "out of memory" in str(e_oom).lower() or isinstance(e_oom, torch.cuda.OutOfMemoryError)
                    if is_oom and attempt == 0:
                        print(f"\n[Sword Alert] CUDA Out-of-Memory intercepted at step {step_idx} (attempt 1). Clearing cache and retrying...")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        b_judge.release_after_eval()
                        continue
                    elif is_oom:
                        print(f"\n[Sword Alert] CUDA Out-of-Memory persisted at step {step_idx}. Securing emergency checkpoint...")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        b_judge.release_after_eval()

                        # Emergency save: NEVER mark incomplete dataset as completed
                        emergency_ckpt = ckpt_mgr.save_checkpoint(
                            step=current_step,
                            model=bridge.trainer_model,
                            optimizer=optimizer,
                            base_model_name=model,
                            reward_logs=reward_history_logs[-50:],
                            completed_dataset_hash=None,
                        )
                        print(f"[Sword] Emergency checkpoint successfully secured at: {emergency_ckpt}")
                        return {
                            "status": "oom_aborted",
                            "last_completed_step": current_step,
                            "emergency_checkpoint": emergency_ckpt,
                            "dataset_sha256": dataset_sha256,
                        }
                    else:
                        raise e_oom

            if not step_success:
                break

            visited_row_hashes.add(current_row.row_hash)
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

        completed_successfully = (current_step >= start_step + steps)

    except KeyboardInterrupt:
        print("\n[Sword] Training interrupted by user. Saving current checkpoint...")
    finally:
        # Mark dataset completed ONLY if training truly completed all planned steps and all dataset rows were visited
        all_rows_visited = len(records) > 0 and all(r.row_hash in visited_row_hashes for r in records)
        if completed_successfully and all_rows_visited:
            ckpt_mgr.mark_dataset_completed(dataset_sha256, str(dataset)[:50])
            print(f"[Sword] Dataset {dataset_sha256} marked completed in data_complete.logs.")

        # Always record completed row hashes for rows that were visited and trained
        for r in records:
            if r.row_hash in visited_row_hashes:
                ckpt_mgr.mark_dataset_completed(r.row_hash, f"row_{r.problem_id}")

        final_ckpt = ckpt_mgr.save_checkpoint(
            step=current_step,
            model=bridge.trainer_model,
            optimizer=optimizer,
            base_model_name=model,
            lora_r=bridge_config.lora_r,
            lora_alpha=bridge_config.lora_alpha,
            target_modules=bridge_config.target_modules,
            reward_logs=reward_history_logs,
            completed_dataset_hash=dataset_sha256 if (completed_successfully and all_rows_visited) else None,
        )
        print(f"\n[Sword] Training process recorded. Final checkpoint: {final_ckpt}")

    if completed_successfully and all_rows_visited:
        run_status = "completed"
    elif completed_successfully:
        run_status = "partial"
    else:
        run_status = "interrupted"

    return {
        "status": run_status,
        "final_step": current_step,
        "visited_rows": len(visited_row_hashes),
        "total_rows": len(records),
        "dataset_sha256": dataset_sha256,
        "checkpoint_dir": final_ckpt,
    }
