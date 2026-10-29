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
import random
import hashlib
import csv
import io
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

from sword.b_model import ModelBJudge, model_b_config
from sword.defaults import (
    DEFAULT_GRPO_MODEL, DEFAULT_GRPO_CONTEXT, DEFAULT_GRPO_MAX_NEW_TOKENS, DEFAULT_MOCK_MAX_NEW_TOKENS,
)
from sword.harness.prompts import harness_repo_policy
from sword.harness import get_harness, HarnessResult, WorkspaceSandbox
from sword.reward import MultiMetricRewardScorer, RewardBreakdown
from sword.checkpoint import CheckpointManager
from sword.task_queue import TaskRetryQueue
from sword.grpo_bridge import (
    GRPOBridge, GRPOBridgeConfig, RolloutBatch, RolloutItem, rollout_prompt_budget, normalize_effort,
)
from sword.logging_utils import quiet_quantization_cast_notices
from sword.lora_source import resolve_lora_training_source, SOURCE_MANIFEST
from sword.execution import normalize_tests, normalize_code_tests, coding_row, has_candidate_code, execute_coding_answer
from sword.execution_evidence import build_execution_evidence


@dataclass
class DatasetProblemRow:
    """Normalized representation of a dataset problem entry."""
    problem: str
    solution: str
    truth: str = ""
    reasoning_rule: str = ""
    answer_rule: str = ""
    difficulty: str = "medium"
    category: str = "general"
    harness: Optional[str] = None
    problem_id: str = ""
    row_hash: str = ""
    effort: str = "dynamic"
    tests: List[Dict[str, Any]] = field(default_factory=list)
    initial_files: Dict[str, str] = field(default_factory=dict)
    language: Optional[str] = None
    code_tests: List[Dict[str, Any]] = field(default_factory=list)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.effort = normalize_effort(self.effort)
        self.code_tests = normalize_code_tests(self.code_tests)

    def to_dict(self) -> Dict[str, Any]:
        result = {key: value for key, value in self.extra_fields.items()
                  if key not in ("can_be_verify", "can_be__verify")}
        result.update({
            "problem": self.problem,
            "solution": self.solution,
            "reasoning_rule": self.reasoning_rule,
            "answer_rule": self.answer_rule,
            "effort": self.effort,
            "difficulty": self.difficulty,
            "category": self.category,
            "harness": self.harness,
            "problem_id": self.problem_id,
            "row_hash": self.row_hash,
        })
        if self.truth or self.tests or self.initial_files or self.language:
            result.update(truth=self.truth, tests=self.tests, initial_files=self.initial_files,
                          language=self.language)
        if self.code_tests:
            result["code_tests"] = self.code_tests
        return result


def policy_task_prompt(row: DatasetProblemRow) -> str:
    """Give the policy its dataset rules without exposing the reference answer."""
    requirements = []
    if row.reasoning_rule:
        requirements.append("Reasoning Rule:\n" + row.reasoning_rule)
    if row.answer_rule:
        requirements.append("Answer Rule:\n" + row.answer_rule)
    if not requirements:
        return row.problem
    return (row.problem + "\n\nDataset response requirements:\n" + "\n\n".join(requirements)
            + "\n\nFollow these requirements throughout your response and tool work. "
              "Follow the reasoning rule when deciding whether and how to reason; "
              "make the final answer satisfy the answer rule.")


def parse_dataset_row(raw: Dict[str, Any], index: int = 0) -> DatasetProblemRow:
    """Defensively parses any dataset dictionary into a standardized DatasetProblemRow."""
    problem = str(raw.get("problem") or raw.get("question") or raw.get("user") or raw.get("prompt") or "").strip()
    solution = next((str(raw[key]).strip() for key in ("solution", "answer", "assistant", "model")
                     if key in raw and raw[key] is not None and str(raw[key]).strip()), "")
    reasoning_rule = str(raw.get("reasoning_rule") or raw.get("reasoning") or "").strip()
    answer_rule = str(raw.get("answer_rule") or raw.get("answer_field") or "").strip()
    truth = str(raw.get("truth") or "").strip()
    effort = normalize_effort(raw.get("effort"))

    difficulty = str(raw.get("difficult") or raw.get("difficulty") or "medium").strip().lower()
    category = str(raw.get("cagetory") or raw.get("category") or "general").strip().lower()
    harness = str(raw.get("harness") or "").strip().lower() or None
    test_spec = raw.get("tests")
    if test_spec is None and raw.get("test_commands"):
        commands = raw["test_commands"]
        if isinstance(commands, str):
            try:
                commands = json.loads(commands)
            except ValueError:
                commands = [commands]
        if isinstance(commands, str):
            commands = [commands]
        if not isinstance(commands, list):
            raise ValueError("test_commands must be a string or a list of command strings")
        test_spec = [{"command": command} for command in commands]
    tests = normalize_tests(test_spec)
    code_tests = normalize_code_tests(raw.get("code_tests"))
    initial_files = raw.get("initial_files", raw.get("files")) or {}
    if isinstance(initial_files, str):
        initial_files = json.loads(initial_files)
    if not isinstance(initial_files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in initial_files.items()):
        raise ValueError("initial_files must map relative filenames to text contents")
    language = str(raw.get("language") or "").strip().lower() or None

    normalized_keys = {"problem", "solution", "truth", "reasoning_rule", "answer_rule", "effort", "difficulty", "category",
                       "harness", "problem_id", "row_hash", "tests", "initial_files", "language", "code_tests",
                       "can_be_verify", "can_be__verify"}
    # Keep other columns (including original aliases) available to the judge
    # and checkpoint snapshots instead of silently discarding row metadata.
    extra_fields = {key: value for key, value in raw.items() if key not in normalized_keys}
    problem_id = raw.get("problem_id", raw.get("id"))
    row = DatasetProblemRow(
        problem=problem,
        solution=solution,
        truth=truth,
        reasoning_rule=reasoning_rule,
        answer_rule=answer_rule,
        effort=effort,
        difficulty=difficulty,
        category=category,
        harness=harness,
        problem_id=str(problem_id) if problem_id is not None else f"problem_{index}",
        tests=tests, initial_files=initial_files, language=language,
        code_tests=code_tests, extra_fields=extra_fields,
    )
    row_dict = {key: value for key, value in row.to_dict().items() if key not in ("problem_id", "row_hash", "id")}
    row.row_hash = hashlib.sha256(json.dumps(row_dict, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return row


def restore_dataset_row(raw: Dict[str, Any], index: int = 0) -> DatasetProblemRow:
    """Migrate stored schemas while retaining queue/completion identity."""
    row = parse_dataset_row(raw, index)
    if raw.get("row_hash"):
        row.row_hash = str(raw["row_hash"])
    return row


def _dataset_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate dataset key %r; each code_tests input/output pair must be a separate object" % key)
        result[key] = value
    return result


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
                            raw_items.append(json.loads(line, object_pairs_hook=_dataset_json_object))
                        except json.JSONDecodeError:
                            pass
            elif spec.endswith(".json"):
                data = json.loads(content_bytes.decode("utf-8", errors="replace"), object_pairs_hook=_dataset_json_object)
                raw_items = data if isinstance(data, list) else [data]
            else:
                text = content_bytes.decode("utf-8-sig", errors="replace")
                delimiter = "\t" if "\t" in text.partition("\n")[0] else ","
                rows = [parts for parts in csv.reader(io.StringIO(text), delimiter=delimiter)
                        if any(part.strip() for part in parts)]
                if rows:
                    columns = [column.strip().lower() for column in rows[0]]
                    if any(column in columns for column in ("problem", "question", "user", "prompt")):
                        raw_items.extend(dict(zip(columns, parts)) for parts in rows[1:])
                    else:
                        raw_items.extend({"problem": parts[0], "solution": parts[1]}
                                         for parts in rows if len(parts) >= 2)

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


def task_rollout_solved(row, candidate, score, harness_result=None, *, threshold=1.0, judge_source=""):
    """Use correctness and completed final answers, independently of aggregate reward."""
    from sword.harness.protocol import visible_response
    answer = visible_response(candidate).strip()
    if not answer or score.cheating or not getattr(harness_result, "is_completed", True):
        return False
    tests = getattr(harness_result, "test_results", [])
    if len(tests) < len(row.tests) + len(row.code_tests):
        return False
    code_cases = [test for test in tests if test.get("kind") == "code_io"]
    if len(code_cases) < len(row.code_tests):
        return False
    if tests and any(not test.get("success", False) for test in tests):
        return False
    return score.truth >= threshold


_DATASET_SHUFFLE_SEED = "sword-grpo-row-shuffle-v1"


def run_long_train(
    dataset: Optional[Union[str, List[Any], Dict[str, Any]]] = "math-ai/TemplateGSM",
    model: str = DEFAULT_GRPO_MODEL,
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
    b_model: bool = True,
    model_lora_repo: str = "",
    train_from_lora_checkpoint: Optional[Union[int, str]] = None,
    gradio_logger: bool = False,
    chat_template_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Main long-running infinite GRPO training entry point.
    Coordinates continuous dataset ingestion, expandable SHA-256 tracking, Model B scoring,
    agentic harnesses, custom checkpointing, and OOM auto-recovery.
    """
    args = Argument or {}
    lora_conf = lora_cfg or {}
    effective_mock = bool(kwargs.get("mock_test", False) or model in ("dummy", "test_model", "mock"))
    from sword.chat_templates import read_chat_template
    template_override = kwargs.get("_chat_template_override")
    if template_override is None:
        template_override = read_chat_template(chat_template_path)
    if not model_lora_repo and train_from_lora_checkpoint not in (None, ""):
        raise ValueError("train_from_lora_checkpoint requires model_lora_repo")
    source_lora = kwargs.get("_lora_training_source")
    if model_lora_repo:
        if effective_mock or kwargs.get("trainer_model") is not None or kwargs.get("tokenizer") is not None:
            raise ValueError("model_lora_repo requires loading the selected checkpoint; mock_test and injected models are incompatible")
        source_lora = source_lora or resolve_lora_training_source(
            model_lora_repo, train_from_lora_checkpoint,
            base_model=None if model == DEFAULT_GRPO_MODEL else model, token=HF_TOKEN,
            revision=kwargs.get("model_lora_revision"), base_revision=kwargs.get("model_revision"),
            cache_dir=kwargs.get("model_cache_dir"), source_cache_dir=kwargs.get("lora_cache_dir"),
            local_files_only=bool(kwargs.get("local_files_only", False)),
        )
        model = source_lora.base_model

    steps = int(args.get("steps") or args.get("max_steps") or 200)
    lr = float(args.get("lr") or args.get("learning_rate") or 5e-6)
    concurrency_g = int(args.get("concurrency_g") or args.get("num_rollouts") or 4)
    save_steps = int(25 if args.get("save_steps") is None else args["save_steps"])
    if save_steps < 1:
        raise ValueError("save_steps must be positive")
    solved_threshold = float(kwargs.get("task_solved_threshold", args.get("task_solved_threshold", 1.0)))
    if not 0 < solved_threshold <= 1:
        raise ValueError("task_solved_threshold must be in (0, 1]")
    micro_batch_size = int(args.get("micro_batch_size") or 1)
    max_seq_len = int(args.get("max_seq_len") or DEFAULT_GRPO_CONTEXT)
    max_new_tokens = int(args.get("max_new_tokens") or
                         (DEFAULT_MOCK_MAX_NEW_TOKENS if effective_mock else DEFAULT_GRPO_MAX_NEW_TOKENS))
    output_dir = str(args.get("output_dir") or "checkpoints")
    harness_max_turns = int(kwargs.get("harness_max_turns", args.get("harness_max_turns", 8)))
    keep_harness_workspaces = bool(kwargs.get("keep_harness_workspaces", args.get("keep_harness_workspaces", False)))
    if harness_max_turns < 1:
        raise ValueError("harness_max_turns must be positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(" SWORD INFINITE GRPO TRAINING ENGINE")
    print("=" * 72)
    print(f" Target Model:            {model}")
    print(f" Model B Judge Port:      {B_model_port}")
    print(f" Concurrency (G):         {concurrency_g} rollouts/problem")
    print(f" Policy context:          {max_seq_len} tokens ({max_new_tokens} reserved per response)")
    if b_model:
        print(f" Judge context:           {model_b_config(**kwargs).num_ctx} tokens")
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
    resolved_ckpt = ckpt_mgr.find_latest_checkpoint(resume_from_checkpoint)
    if (resume_from_checkpoint and str(resume_from_checkpoint).strip().lower() not in ("none", "false", "0")
            and not resolved_ckpt):
        raise FileNotFoundError(f"Requested resume checkpoint was not found: {resume_from_checkpoint}")
    if resolved_ckpt:
        ckpt_mgr._validate_checkpoint(resolved_ckpt)
    restored_state = ckpt_mgr.read_training_state(resolved_ckpt) if resolved_ckpt else {}
    saved_dataset = ckpt_mgr.read_dataset_snapshot(resolved_ckpt) if resolved_ckpt else None
    if resolved_ckpt:
        ckpt_mgr.restore_completion_log(resolved_ckpt)

    # 2. Ingest Dataset & Check Expandable SHA-256 Deduplication
    snapshot_requested = dataset is None or (
        saved_dataset is not None and isinstance(dataset, str) and not os.path.exists(dataset)
        and (os.path.isabs(dataset) or dataset.lower().endswith((".json", ".jsonl", ".csv", ".tsv"))))
    if snapshot_requested:
        if not saved_dataset:
            raise ValueError("dataset=None or a missing local dataset requires a checkpoint dataset snapshot")
        records = [restore_dataset_row(row, index) for index, row in enumerate(saved_dataset["records"])]
        dataset_sha256 = saved_dataset["dataset_sha256"]
        print("[*] Restored the dataset snapshot from the checkpoint.")
    else:
        records, dataset_sha256 = load_dataset_records(dataset, hf_token=HF_TOKEN)
    if not records:
        raise ValueError("The training dataset contains no problems")
    if source_lora is not None and not snapshot_requested:
        source_key = json.dumps(source_lora.manifest, sort_keys=True)
        dataset_sha256 = hashlib.sha256((dataset_sha256 + source_key).encode()).hexdigest()
        for row in records:
            row.row_hash = hashlib.sha256((row.row_hash + source_key).encode()).hexdigest()[:16]
    if template_override is not None and not snapshot_requested:
        template_key = hashlib.sha256(template_override.encode("utf-8")).hexdigest()
        dataset_sha256 = hashlib.sha256((dataset_sha256 + template_key).encode()).hexdigest()
        for row in records:
            row.row_hash = hashlib.sha256((row.row_hash + template_key).encode()).hexdigest()[:16]
    print(f"[*] Dataset SHA-256: {dataset_sha256} ({len(records)} problem rows)")
    all_records = list(records)
    dataset_snapshot = {"schema_version": 2, "dataset_sha256": dataset_sha256,
                        "records": [row.to_dict() for row in all_records]}

    allow_retest = bool(kwargs.get("allow_retest", args.get("allow_retest", False)))
    completed_hashes = ckpt_mgr.get_completed_dataset_hashes()

    # If entire dataset SHA-256 matches completed log
    if dataset_sha256 in completed_hashes and not allow_retest and not resolved_ckpt:
        print(f"\n[Sword] The data is already train on, suggest prepare new dataset to continues.")
        return {
            "status": "already_trained",
            "dataset_sha256": dataset_sha256,
            "steps_completed": 0,
        }

    # Expandable dataset: filter out already-completed rows
    if not allow_retest and not resolved_ckpt:
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

    # Deterministic dataset shuffle with a hardcoded seed: the round-robin
    # training walk otherwise follows the file/append order of the Drive file
    # (easy rows first, new rows last). random.Random(<str>) ignores Python's
    # hash randomization, so every run - including a GRPO resume from a LoRA
    # repo - shuffles the identical training order for the same dataset.
    # Pass shuffle_seed="" to keep original file order instead.
    shuffle_seed = str(kwargs.get("shuffle_seed", args.get("shuffle_seed", _DATASET_SHUFFLE_SEED)))
    if shuffle_seed and len(records) > 1:
        grouped = bool(kwargs.get("shuffle_by_category", args.get("shuffle_by_category", True)))
        categories = {}  # BUG (shuffle.category_scatter) fix: plain shuffling scattered
        for record in records:  # rows of one cagetory across the whole walk, so the
            categories.setdefault(record.category, []).append(record)  # model bounced between topics mid-pass instead of drilling each one.
        if grouped and len(categories) > 1:
            # Category-cohesive shuffle: shuffle rows WITHIN each category
            # deterministically, then concatenate the categories in their own
            # deterministic shuffled order. Same seed -> same order every run,
            # row-hash dedup and resume all still see the identical sequence.
            rng = random.Random(shuffle_seed)
            for rows in categories.values():
                rng.shuffle(rows)
            order = list(categories)
            rng.shuffle(order)
            records = [row for cat in order for row in categories[cat]]
            print(f"[*] Dataset shuffled deterministically by category "
                  f"(seed={shuffle_seed!r}, {len(records)} rows, {len(categories)} categories)")
        else:
            random.Random(shuffle_seed).shuffle(records)
            print(f"[*] Dataset shuffled deterministically (seed={shuffle_seed!r}, {len(records)} rows)")

    # 3. Model B Judge & Scorer Initialization
    quiet_quantization_cast_notices()
    judge_options = {"log_model_outputs": show_detail_logs, **kwargs, "mock_test": effective_mock}
    b_judge_config = model_b_config(
        B_model_port, auto_unload=auto_unload_B_model, keep_alive=B_model_keep_alive, **judge_options,
    )
    b_judge = ModelBJudge(config=b_judge_config) if b_model else None
    if b_judge is not None and effective_mock:
        b_judge._memory_ready = False
        b_judge.config.auto_unload = False
    if b_judge is not None and b_judge_config.auto_setup:
        # Download before allocating trainer VRAM; inference loads only in the score phase.
        b_judge.ensure_ready()
    # Style metrics cannot subsidize factually wrong answers: "pretty but
    # wrong" caps at 0.25 of the flat average. Pass wrong_answer_cap=1.0 to
    # restore the legacy behavior.
    reward_scorer = MultiMetricRewardScorer(
        b_model_judge=b_judge,
        wrong_answer_cap=float(kwargs.get("wrong_answer_cap", 0.25)),
    )

    # 4. GRPO Bridge Setup
    bridge_config = GRPOBridgeConfig(
        model_name=model,
        concurrency_g=concurrency_g,
        max_model_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        micro_batch_size=micro_batch_size,
        # BUG (rollout.max_num_seqs_default) fix: the default of 2 ignored
        # concurrency_g - rollouts are always submitted as one group in a single
        # vLLM generate() call, so the engine needs at least one decode lane per
        # group member. max_num_seqs=2 serialized an 8-rollout batch 4x and
        # crushed per-sequence tok/s with it.
        vllm_max_num_seqs=int(kwargs.get("vllm_max_num_seqs", max(2, concurrency_g))),
        vllm_max_num_batched_tokens=int(kwargs.get("vllm_max_num_batched_tokens", 2048)),
        use_cross_step_baseline=bool(kwargs.get("use_cross_step_baseline", args.get("use_cross_step_baseline", True))),
        temperature=float(args.get("temperature", 0.7)),
        top_p=float(args.get("top_p", 0.9)),
        chunk_size=int(args.get("chunk_size", 512)),
        enable_vllm_sleep=bool(kwargs.get("enable_vllm_sleep", True)),
        offload_trainer_during_rollout=bool(kwargs.get("offload_trainer_during_rollout", True)),
        vllm_gpu_utilization=float(kwargs.get("vllm_gpu_utilization", 0.40)),
        adapter_sync_dir=os.path.join(output_dir, "rollout_adapters"),
        lora_r=int(lora_conf.get("r", 16)),
        lora_alpha=int(lora_conf.get("lora_alpha", 32)),
        lora_dropout=float(lora_conf.get("lora_dropout", 0.0)),
        target_modules=lora_conf.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        vllm_quantization=kwargs.get("vllm_quantization", kwargs.get("quantization")),
        kv_cache_dtype=str(kwargs.get("kv_cache_dtype", "auto")),
        speculative_ngram=bool(kwargs.get("speculative_ngram", kwargs.get("ngram", False))),
        num_speculative_tokens=int(kwargs.get("num_speculative_tokens", 4)),
        ngram_prompt_lookup_max=int(kwargs.get("ngram_prompt_lookup_max", 4)),
        hf_overrides=kwargs.get("hf_overrides"),
        load_in_4bit=bool(kwargs.get("load_in_4bit", False)),
        load_in_8bit=bool(kwargs.get("load_in_8bit", False)),
        hf_token=HF_TOKEN, model_cache_dir=kwargs.get("model_cache_dir"),
        model_revision=kwargs.get("model_revision"),
        local_files_only=bool(kwargs.get("local_files_only", False)),
        require_vllm=bool(kwargs.get("require_vllm", False)),
        device=device,
        log_model_outputs=bool(kwargs.get("log_model_outputs", show_detail_logs)),
        model_output_chars=int(kwargs.get("model_output_chars", 1200)),
        enable_thinking=bool(kwargs.get("A_model_thinking", True)),
        lora_training_source=source_lora,
        chat_template_path=chat_template_path, chat_template_override=template_override,
        merged_base_cache_dir=str(kwargs.get("merged_base_cache_dir") or os.path.join(output_dir, "merged_bases")),
    )
    bridge = GRPOBridge(config=bridge_config)
    try:
        bridge.initialize_trainer(
            model=kwargs.get("trainer_model"),
            tokenizer=kwargs.get("tokenizer"),
            mock_test=effective_mock,
        )
    except Exception:
        bridge.weight_sync.cleanup()
        if b_judge is not None:
            b_judge.release_after_eval()
        raise

    trainable_params = [p for p in bridge.trainer_model.parameters() if p.requires_grad]
    checkpoint_source_options = {"source_lora_manifest": bridge.source_lora_manifest,
                                 "tokenizer": bridge.tokenizer} if bridge.source_lora_manifest else {}
    if template_override is not None:
        checkpoint_source_options["tokenizer"] = bridge.tokenizer
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # 5. Checkpoint Resume Discovery (only when explicitly requested)
    start_step = 0
    try:
        if resolved_ckpt:
            if bridge.source_lora_manifest is not None:
                source_file = os.path.join(resolved_ckpt, SOURCE_MANIFEST)
                if not os.path.isfile(source_file):
                    raise ValueError("Resume checkpoint is missing its merged-base source manifest")
                with open(source_file, encoding="utf-8") as stream:
                    if json.load(stream) != bridge.source_lora_manifest:
                        raise ValueError("Resume checkpoint was trained on a different source LoRA base")
            print(f"[*] Resuming from checkpoint: {resolved_ckpt}")
            if restored_state.get("dataset_sha256", dataset_sha256) != dataset_sha256:
                raise ValueError("Resume checkpoint was saved for a different dataset")
            res_info = ckpt_mgr.load_checkpoint(ckpt_dir=resolved_ckpt, model=bridge.trainer_model, optimizer=optimizer)
            start_step = res_info["step"]
            bridge.global_step = start_step
            baseline = restored_state.get("reward_baseline", {})
            bridge._reward_ema = baseline.get("mean")
            bridge._reward_ema_var = baseline.get("variance", 1.0)
            if "task_solved_threshold" not in kwargs and "task_solved_threshold" not in args:
                solved_threshold = float(restored_state.get("task_solved_threshold", solved_threshold))
            print(f"[*] Resumed at step {start_step} ({res_info['matched_keys']} adapter keys matched).")

        task_queue = TaskRetryQueue(
            all_records,
            retry_limit=kwargs.get("task_retry_limit", args.get("task_retry_limit", 3)),
            retry_delay=kwargs.get("task_retry_delay", args.get("task_retry_delay", 100)),
            failure_action=kwargs.get("task_failure_action", args.get("task_failure_action", "waitlist")),
            enabled=kwargs.get("task_retry_enabled", args.get("task_retry_enabled", True)),
            completed_hashes=completed_hashes if not allow_retest and not resolved_ckpt else (),
            state=restored_state.get("task_queue"),
            decisions_path=os.path.join(output_dir, "task_decisions.jsonl"),
        )
        # 6. Initialize vLLM / Fallback Engine, then undo initialization's RNG use.
        bridge.initialize_vllm()
        if resolved_ckpt:
            ckpt_mgr.restore_random_state(resolved_ckpt)
    except BaseException:
        bridge.weight_sync.cleanup()
        if b_judge is not None:
            b_judge.release_after_eval()
        raise
    record_lookup = {row.row_hash: row for row in all_records}

    reward_history_logs: List[Dict[str, Any]] = []
    step_metrics: List[Dict[str, Any]] = []
    visited_row_hashes = set(restored_state.get("visited_row_hashes", []))
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    rollout_details_path = os.path.join(output_dir, "rollout_details.jsonl")
    trace_rollouts = bool(kwargs.get("trace_rollouts", False))
    current_step = start_step
    completed_successfully = False
    exit_reason = None
    score_sum = float(restored_state.get("score_sum", 0.0))
    score_count = int(restored_state.get("score_count", 0))
    last_checkpoint_revision = int(restored_state.get("task_queue", {}).get("decision_revision", task_queue.decision_revision))
    wait_checkpoint_saved = False
    dashboard = None
    if gradio_logger:
        from sword.training_logger import create_training_logger
        dashboard = create_training_logger(max_rollouts=kwargs.get("gradio_max_rollouts", 8),
                                           refresh_interval=kwargs.get("gradio_refresh_interval", 1.0),
                                           server_port=kwargs.get("gradio_port"))
        dashboard.attach_progress(task_queue, ckpt_mgr, save_steps=save_steps, score_sum=score_sum, score_count=score_count)

    def save_progress_checkpoint(logs=None, completed_hash=None):
        nonlocal last_checkpoint_revision
        queue_state = task_queue.export_state()
        path = ckpt_mgr.save_checkpoint(
            step=current_step, model=bridge.trainer_model, optimizer=optimizer,
            base_model_name=bridge.config.model_name, lora_r=bridge_config.lora_r,
            lora_alpha=bridge_config.lora_alpha, target_modules=bridge_config.target_modules,
            reward_logs=reward_history_logs if logs is None else logs,
            completed_dataset_hash=completed_hash, dataset_snapshot=dataset_snapshot,
            training_state={"dataset_sha256": dataset_sha256, "task_queue": queue_state,
                            "visited_row_hashes": sorted(visited_row_hashes), "score_sum": score_sum,
                            "score_count": score_count, "task_solved_threshold": solved_threshold,
                            "reward_baseline": {"mean": bridge._reward_ema, "variance": bridge._reward_ema_var}},
            **checkpoint_source_options,
        )
        last_checkpoint_revision = queue_state["decision_revision"]
        return path

    print(f"\n[*] Starting training loop from step {start_step + 1} to {start_step + steps}...\n")

    try:
        while current_step < start_step + steps:
            if task_queue.decision_revision != last_checkpoint_revision:
                save_progress_checkpoint(reward_history_logs[-100:])
            current_key = task_queue.next_key()
            if current_key is None:
                waiting = task_queue.monitor_snapshot(limit=0)["summary"]["waiting_approval"]
                exit_reason = "waiting_approval" if waiting else "exhausted"
                if not waiting:
                    break
                if not wait_checkpoint_saved:
                    save_progress_checkpoint(reward_history_logs[-100:])
                    wait_checkpoint_saved = True
                if dashboard is not None:
                    dashboard.publish("status", status="Waiting for task approval")
                if (dashboard is None or not kwargs.get("gradio_wait_for_approval", args.get("gradio_wait_for_approval", True))
                        or dashboard.error or (dashboard._launch_complete.is_set() and dashboard.app is None)):
                    break
                task_queue.decision_event.wait(0.5)
                task_queue.decision_event.clear()
                continue
            exit_reason = None
            wait_checkpoint_saved = False
            step_idx = current_step + 1
            current_row = record_lookup[current_key]
            # Keep the same dataset effort for the initial response and all tool follow-ups.
            bridge.config.effort = current_row.effort
            policy_task = policy_task_prompt(current_row)
            harness = get_harness(current_row.harness)
            if harness is not None:
                harness.workspace_parent_dir = os.path.join(output_dir, "harness_workspaces")
                harness.keep_workspace = keep_harness_workspaces
                harness.dataset_tests = current_row.tests
                harness.code_tests = current_row.code_tests
                harness.initial_files = current_row.initial_files
                harness.language = current_row.language

            step_success = False
            log_start_len = len(reward_history_logs)

            for attempt in range(2):
                reward_baseline_state = (bridge._reward_ema, bridge._reward_ema_var)
                # Clean up reward logs from previous failed attempt to avoid duplicates
                if attempt > 0 and len(reward_history_logs) > log_start_len:
                    del reward_history_logs[log_start_len:]

                try:
                    if dashboard is not None:
                        dashboard.publish("task", step=step_idx, problem_id=current_row.problem_id,
                                          problem=current_row.problem, rollout_count=concurrency_g,
                                          row_hash=current_row.row_hash, category=current_row.category)
                        bridge.generation_callback = dashboard.generation
                    # Rollout Phase with authentic harness prompt injection and token-budget guard
                    prompt_token_budget = rollout_prompt_budget(
                        bridge.tokenizer, max_seq_len, max_new_tokens, enable_thinking=bridge.config.enable_thinking,
                        effort=current_row.effort,
                    )
                    with harness_repo_policy(auto_clone=not effective_mock):
                        rollout_prompt = (
                            harness.format_prompt(
                                policy_task,
                                max_tokens=prompt_token_budget,
                                tokenizer=getattr(bridge, "tokenizer", None),
                            )
                            if harness is not None else policy_task
                        )
                    prompts = [rollout_prompt]
                    batch = bridge.generate_rollouts(prompts)

                    # Complete all harness generation before loading the judge.
                    step_scores = []
                    scoring_inputs = []
                    step_details = []

                    for item in batch.items:
                        if dashboard is not None:
                            dashboard.publish("status", status=f"Executing rollout {len(scoring_inputs) + 1}")
                        tools_called = 0
                        tools_succeeded = 0
                        h_res = None

                        # Multi-turn harness interaction
                        if harness is not None:
                            harness.log_tool_outputs = bool(kwargs.get("log_model_outputs", show_detail_logs))
                            harness.model_output_chars = int(kwargs.get("model_output_chars", 1200))
                            turn_number = 0
                            def multi_turn_generate(turn_p: str) -> str:
                                nonlocal turn_number
                                turn_number += 1
                                if turn_p == item.prompt_text:
                                    if show_detail_logs:
                                        print(f"[Agent Turn] rollout={len(scoring_inputs)} turn={turn_number} "
                                              f"phase=initial tokens={len(item.response_token_ids)}", flush=True)
                                    return item.response_text
                                # Preserve the task and recent observations without
                                # overflowing the model window as tool turns grow.
                                history = turn_p[len(item.prompt_text):] if turn_p.startswith(item.prompt_text) else turn_p
                                followup = harness.format_followup_prompt(
                                    policy_task, history, prompt_token_budget, tokenizer=bridge.tokenizer,
                                )
                                saved_callback = bridge.generation_callback
                                if dashboard is not None:
                                    bridge.generation_callback = lambda _, text, finished=False: dashboard.generation(
                                        len(scoring_inputs), text, finished=finished, turn=turn_number - 1)
                                try:
                                    turn = bridge.generate_single(followup, return_item=True)
                                finally:
                                    bridge.generation_callback = saved_callback
                                if not isinstance(turn, RolloutItem):
                                    raise RuntimeError("Harness follow-up generation must retain sampled tokens and logprobs")
                                item.followup_turns.append(turn)
                                if show_detail_logs:
                                    print(f"[Agent Turn] rollout={len(scoring_inputs)} turn={turn_number} "
                                          f"phase=followup tokens={len(turn.response_token_ids)}", flush=True)
                                return turn.response_text

                            h_res = harness.run(
                                prompt=item.prompt_text,
                                model_generate_fn=multi_turn_generate,
                                task_id=current_row.problem_id,
                                max_turns=harness_max_turns,
                            )
                            tools_called = h_res.tools_called
                            tools_succeeded = h_res.tools_succeeded

                            # Assign the outcome reward to the complete sampled trajectory.
                            candidate_response = h_res.final_text if getattr(h_res, "is_completed", True) else ""
                            if show_detail_logs:
                                print(f"[Harness {len(scoring_inputs)}] turns={getattr(h_res, 'turns', 0)} "
                                      f"stop={getattr(h_res, 'stop_reason', 'unknown')} tools={tools_succeeded}/{tools_called} "
                                      f"workspace={getattr(h_res, 'workspace_dir', '')} "
                                      f"({'removed' if getattr(h_res, 'workspace_removed', False) else 'kept'})")
                        else:
                            candidate_response = item.response_text
                            if kwargs.get("auto_code_executor", True) and (coding_row(current_row) or has_candidate_code(candidate_response)):
                                h_res = execute_coding_answer(
                                    current_row, candidate_response,
                                    parent_dir=os.path.join(output_dir, "harness_workspaces"),
                                    keep_workspace=keep_harness_workspaces)
                                tools_called, tools_succeeded = h_res.tools_called, h_res.tools_succeeded
                        scoring_inputs.append((item, candidate_response, tools_called, tools_succeeded, h_res))

                    bridge.vram_mgr.enter_score_phase(bridge.vllm_engine)
                    if dashboard is not None:
                        dashboard.publish("status", status="Scoring")
                    if b_judge is not None and not effective_mock:
                        b_judge.prepare_for_eval()
                    for rollout_idx, (item, candidate_response, tools_called, tools_succeeded, h_res) in enumerate(scoring_inputs):
                        execution_evidence = build_execution_evidence(
                            getattr(h_res, "tool_history", []), tools_called=tools_called, tools_succeeded=tools_succeeded,
                            test_results=getattr(h_res, "test_results", []), execution_results=getattr(h_res, "execution_results", []),
                            max_chars=int(kwargs.get("judge_execution_chars", 4000)),
                            max_success=int(kwargs.get("judge_execution_success_samples", 2)),
                            max_failed=int(kwargs.get("judge_execution_failed_samples", 4))) if h_res is not None else None
                        assistant_turns = [item] + item.followup_turns
                        # An unfinished last turn is still evidence for the reasoning rule.
                        earlier_turns = assistant_turns if h_res is not None and not getattr(
                            h_res, "is_completed", True,
                        ) else assistant_turns[:-1]
                        score_breakdown = reward_scorer.score(
                            problem=current_row.problem,
                            solution=current_row.solution,
                            candidate_response=candidate_response,
                            reasoning_rule=current_row.reasoning_rule,
                            answer_rule=current_row.answer_rule,
                            effort=current_row.effort,
                            difficulty=current_row.difficulty,
                            category=current_row.category,
                            harness_name=current_row.harness,
                            tools_called=tools_called,
                            tools_succeeded=tools_succeeded,
                            assistant_history=[turn.response_text for turn in earlier_turns],
                            dataset_row=current_row.to_dict(),
                            code_execution=getattr(h_res, "code_execution", None),
                            **({"truth_guide": current_row.truth} if current_row.truth else {}),
                            **({"execution_evidence": execution_evidence} if execution_evidence is not None else {}),
                        )
                        item.reward = score_breakdown.total_score
                        step_scores.append(score_breakdown)

                        detail = {
                            "step": step_idx, "rollout": rollout_idx, "group_id": item.group_id,
                            "problem_id": current_row.problem_id, "problem": current_row.problem,
                            "row_hash": current_row.row_hash, "category": current_row.category,
                            "effort": current_row.effort,
                            "scores": score_breakdown.to_dict(), "judge_scores": score_breakdown.judge_scores,
                            "judge_source": str(score_breakdown.judge_source or (b_judge.last_evaluation_source if b_judge else "disabled")),
                            "judge_response": score_breakdown.judge_response,
                            "candidate_preview": candidate_response[:512], "candidate_tail": candidate_response[-512:],
                            "candidate_chars": len(candidate_response), "response_tokens": len(item.response_token_ids),
                            "is_truncated": item.is_truncated,
                            "training_turn_count": 1 + len(item.followup_turns),
                            "training_response_tokens": sum(len(turn.response_token_ids) for turn in [item] + item.followup_turns),
                            "tools_called": tools_called, "tools_succeeded": tools_succeeded,
                            "execution_evidence": execution_evidence,
                            "code_execution": getattr(h_res, "code_execution", {}),
                        }
                        detail["solved"] = task_rollout_solved(
                            current_row, candidate_response, score_breakdown, h_res, threshold=solved_threshold,
                            judge_source=detail["judge_source"])
                        if h_res is not None:
                            detail["harness"] = {
                                "turns": getattr(h_res, "turns", 0), "stop_reason": getattr(h_res, "stop_reason", "unknown"),
                                "is_completed": getattr(h_res, "is_completed", True),
                                "workspace_dir": getattr(h_res, "workspace_dir", ""),
                                "workspace_removed": getattr(h_res, "workspace_removed", False),
                            }
                            detail["harness_last_response_preview"] = h_res.final_text[-512:]
                        if trace_rollouts:
                            detail.update(
                                candidate_response=candidate_response, first_response=item.response_text,
                                prompt_text=item.prompt_text, prompt_token_ids=item.prompt_token_ids,
                                response_token_ids=item.response_token_ids, rollout_logprobs=item.rollout_logprobs,
                                tool_history=getattr(h_res, "tool_history", []) if h_res is not None else [],
                                test_results=getattr(h_res, "test_results", []) if h_res is not None else [],
                                execution_results=getattr(h_res, "execution_results", []) if h_res is not None else [],
                                assistant_turns=getattr(h_res, "assistant_turns", []) if h_res is not None else [],
                                training_turns=[{
                                    "prompt_text": turn.prompt_text, "prompt_token_ids": turn.prompt_token_ids,
                                    "response_text": turn.response_text, "response_token_ids": turn.response_token_ids,
                                    "rollout_logprobs": turn.rollout_logprobs, "policy_version": turn.policy_version,
                                    "is_truncated": turn.is_truncated,
                                } for turn in [item] + item.followup_turns],
                                judge_attempts=[dict(attempt) for attempt in getattr(b_judge, "last_evaluation_attempts", [])]
                                               if b_judge is not None else [],
                            )
                        step_details.append(detail)
                        if dashboard is not None:
                            dashboard.publish("score", rollout=rollout_idx, scores=score_breakdown.to_dict(),
                                              judge_scores=score_breakdown.judge_scores,
                                              judge_response=score_breakdown.judge_response[:16000],
                                              judge_source=score_breakdown.judge_source,
                                              execution_evidence=execution_evidence,
                                              code_execution=detail["code_execution"])
                        if score_logs:
                            reward_history_logs.append(detail)

                    # Release Model B VRAM before training forward / backward pass
                    if b_judge is not None:
                        b_judge.release_after_eval()

                    # Share advantage calculation and degeneracy diagnostics with
                    # the bridge; harness outcome rewards have already been assigned.
                    bridge.compute_advantages(batch)
                    for detail, (item, _, _, _, _) in zip(step_details, scoring_inputs):
                        detail["advantage"] = item.advantage
                    if not batch.items:
                        raise RuntimeError("No complete rollout groups available for training")

                    # Execute training forward / backward step
                    if dashboard is not None:
                        dashboard.publish("status", status="Training")
                    metrics = bridge.train_step(batch, optimizer)
                    metrics["judge_evaluation_source"] = b_judge.last_evaluation_source if b_judge else "disabled"
                    metrics.update(
                        unique_responses=len({candidate for _, candidate, _, _, _ in scoring_inputs}),
                        empty_responses=sum(not candidate.strip() for _, candidate, _, _, _ in scoring_inputs),
                        truncated_rollouts=sum(item.is_truncated for item in batch.items),
                        tools_called=sum(detail["tools_called"] for detail in step_details),
                        incomplete_harness_rollouts=sum(h_res is not None and not getattr(h_res, "is_completed", True)
                                                        for _, _, _, _, h_res in scoring_inputs),
                    )
                    step_success = True
                    break

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e_oom:
                    # A failed attempt must not become its retry's reward history.
                    bridge._reward_ema, bridge._reward_ema_var = reward_baseline_state
                    is_oom = "out of memory" in str(e_oom).lower() or isinstance(e_oom, torch.cuda.OutOfMemoryError)
                    if is_oom and attempt == 0:
                        print(f"\n[Sword Alert] CUDA Out-of-Memory intercepted at step {step_idx} (attempt 1). Clearing cache and retrying...")
                        # Release failed autograd graphs and gradients before retry.
                        optimizer.zero_grad(set_to_none=True)
                        e_oom.__traceback__ = None
                        bridge.config.micro_batch_size = max(1, bridge.config.micro_batch_size // 2)
                        bridge.loss_fn.chunk_size = max(1, bridge.loss_fn.chunk_size // 2)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if b_judge is not None:
                            b_judge.release_after_eval()
                        continue
                    elif is_oom:
                        print(f"\n[Sword Alert] CUDA Out-of-Memory persisted at step {step_idx}. Securing emergency checkpoint...")
                        optimizer.zero_grad(set_to_none=True)
                        e_oom.__traceback__ = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if b_judge is not None:
                            b_judge.release_after_eval()

                        # Emergency save: NEVER mark incomplete dataset as completed
                        del reward_history_logs[log_start_len:]
                        emergency_ckpt = save_progress_checkpoint(reward_history_logs[-50:])
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
            best_truth = max(s.truth for s in step_scores)
            task_outcome = task_queue.record_result(
                current_key, solved_rollouts=sum(detail["solved"] for detail in step_details),
                rollout_count=len(step_details), best_truth=best_truth, step=current_step,
                reason=f"No correct completed rollout; highest truth score {best_truth:.4g}")
            score_sum += sum(s.total_score for s in step_scores)
            score_count += len(step_scores)
            metrics.update(task_status=task_outcome["status"], task_failures=task_outcome["failures"],
                           task_attempts=task_outcome["attempts"])
            for detail in step_details:
                detail.update(task_status=task_outcome["status"], task_attempt=task_outcome["attempts"],
                              task_failures=task_outcome["failures"])
            if dashboard is not None:
                dashboard.publish("step", step=step_idx, scores=[s.total_score for s in step_scores])
            if metrics.get("optimizer_step_skipped", False):
                skipped_optimizer_steps += 1
            else:
                optimizer_steps += 1
            if kwargs.get("return_step_metrics", False):
                step_metrics.append(dict(metrics))
            # Trace IO runs after committing the completed step so an IO error
            # cannot leave an updated policy checkpointed under the prior step.
            if score_logs:
                with open(rollout_details_path, "a", encoding="utf-8") as detail_file:
                    for detail in step_details:
                        detail_file.write(json.dumps(detail, ensure_ascii=False) + "\n")

            # Logging
            if show_detail_logs:
                avg_reward = sum(s.total_score for s in step_scores) / max(1, len(step_scores))
                avg_truth = sum(s.truth for s in step_scores) / max(1, len(step_scores))
                grad_norm_disp = metrics.get('grad_norm', 0.0)
                print(
                    f"Step {step_idx:04d}/{start_step + steps:04d} | "
                    f"Loss: {metrics.get('grpo_loss', 0.0):.6g} | "
                    f"GradNorm: {grad_norm_disp:.6g} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"RewardStd: {metrics.get('reward_std', 0.0):.6g} | "
                    f"AdvMax: {metrics.get('advantage_abs_max', 0.0):.6g} | "
                    f"TrainTurns: {metrics.get('training_turns', 0)} | "
                    f"ThoughtTurns: {metrics.get('reasoning_turns', 0)} | "
                    f"Truth: {avg_truth:.2f} | "
                    f"Judge: {metrics['judge_evaluation_source']} | "
                    f"Update: {'skipped' if metrics.get('optimizer_step_skipped') else 'applied'} | "
                    f"Harness: {current_row.harness or 'none'} | "
                    f"Diff: {current_row.difficulty}"
                )
                if metrics.get("reward_std", 1.0) <= 1e-6 or metrics.get("optimizer_step_skipped", False):
                    reason = "Uniform rewards" if metrics.get("reward_std", 1.0) <= 1e-6 else "Zero-gradient step"
                    print(f"[GRPO] {reason}: unique responses {metrics['unique_responses']}/{len(batch.items)}, "
                          f"empty {metrics['empty_responses']}, truncated {metrics['truncated_rollouts']}, "
                          f"tools called {metrics['tools_called']}; policy loss {metrics.get('policy_loss', 0):.6g}, "
                          f"entropy penalty {metrics.get('entropy_penalty', 0):.6g}, "
                          f"gradient buffers {metrics.get('tensors_with_grad', 0)}/{metrics.get('trainable_tensors', 0)}")
                    print(f"[GRPO] Gradient signal: max advantage {metrics.get('advantage_abs_max', 0):.6g}, "
                          f"unique first turns {metrics.get('unique_first_turns', 0)}/{len(batch.items)}, "
                          f"unique trajectories {metrics.get('unique_training_trajectories', 0)}/{len(batch.items)}, "
                          f"training turns {metrics.get('training_turns', 0)}, "
                          f"response tokens {metrics.get('response_token_count', 0)}, "
                          f"nonzero gradient buffers {metrics.get('tensors_with_nonzero_grad', 0)}")
                    print(f"[GRPO] Policy statistics: logprob min/max "
                          f"{metrics.get('policy_logprob_min', 0):.6g}/{metrics.get('policy_logprob_max', 0):.6g}, "
                          f"policy/rollout mean logprob {metrics.get('policy_logprob_mean', 0):.6g}/"
                          f"{metrics.get('rollout_logprob_mean', 0):.6g}, "
                          f"entropy {metrics.get('mean_entropy', 0):.6g}, "
                          f"zero importance fraction {metrics.get('importance_zero_fraction', 0):.6g}; "
                          f"reason={metrics.get('zero_gradient_reason', '')}")
                    for detail in step_details[:2]:
                        print(f"[GRPO Rollout {detail['rollout']}] Scores: {json.dumps(detail['scores'])}")
                        print(f"[GRPO Response {detail['rollout']}] {detail['candidate_preview']!r}")
                        if detail["judge_response"]:
                            print(f"[ModelB JSON {detail['rollout']}] {detail['judge_response'][:2048]}")
                    if score_logs:
                        print(f"[GRPO] Rollout details: {rollout_details_path}")

            # Periodic checkpoint saving
            if current_step % save_steps == 0:
                saved_path = save_progress_checkpoint(reward_history_logs[-100:])
                print(f"[*] Checkpoint saved at: {saved_path}")

        completed_successfully = (current_step >= start_step + steps)

    except KeyboardInterrupt:
        print("\n[Sword] Training interrupted by user. Saving current checkpoint...")
    finally:
        bridge.generation_callback = None
        if b_judge is not None:
            b_judge.release_after_eval()
        bridge.weight_sync.cleanup()
        # Unsolved rows must remain eligible after a reset or a new invocation.
        all_rows_visited = len(records) > 0 and all(r.row_hash in visited_row_hashes for r in records)
        solved_keys = task_queue.solved_keys()
        complete_keys = solved_keys if task_queue.config["enabled"] else visited_row_hashes | completed_hashes
        all_rows_solved = all(row.row_hash in complete_keys for row in all_records)
        dataset_completed = completed_successfully and all_rows_visited and all_rows_solved
        final_ckpt = save_progress_checkpoint(completed_hash=dataset_sha256 if dataset_completed else None)
        if dataset_completed:
            print(f"[Sword] Dataset {dataset_sha256} marked completed in data_complete.logs.")
        print(f"\n[Sword] Training process recorded. Final checkpoint: {final_ckpt}")
        if dashboard is not None:
            dashboard.finish("Waiting for task approval" if exit_reason == "waiting_approval" else
                             "Task queue exhausted" if exit_reason == "exhausted" else
                             "Completed" if completed_successfully else "Interrupted")

    if exit_reason:
        run_status = exit_reason
    elif completed_successfully and all_rows_visited:
        run_status = "completed"
    elif completed_successfully:
        run_status = "partial"
    else:
        run_status = "interrupted"

    return {
        "status": run_status,
        "final_step": current_step,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "rollout_details_path": rollout_details_path if score_logs else None,
        "visited_rows": len(visited_row_hashes),
        "total_rows": len(records),
        "dataset_sha256": dataset_sha256,
        "checkpoint_dir": final_ckpt,
        "rollout_backend": "vllm" if bridge.vllm_engine is not None else "pytorch",
        "vllm_initialization_error": bridge.vllm_initialization_error,
        "training_base_model": bridge.config.model_name,
        "source_lora": bridge.source_lora_manifest,
        "step_metrics": step_metrics,
        "gradio_logger": dashboard.info() if dashboard is not None else {"enabled": False},
        "task_progress": task_queue.monitor_snapshot(),
        "checkpoint_upload": ckpt_mgr.upload_info(),
        "b_model": {
            "enabled": True,
            "model_name": b_judge.config.model_name,
            "active_model_name": b_judge.active_model_name,
            "evaluation_source": b_judge.last_evaluation_source,
            "evaluation_counts": dict(b_judge.evaluation_counts),
            "last_error": b_judge.last_error,
        } if b_judge is not None else {"enabled": False, "evaluation_counts": {}},
    }
