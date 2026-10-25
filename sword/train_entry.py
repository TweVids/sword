import os
import sys
import time
import json
import subprocess
from typing import Optional, Any, Dict, List, Union

import torch


DEFAULT_BENCHMARK_PROMPTS = [
    "Explain how pure-PyTorch SDPA achieves FlashAttention performance without external wheels.",
    "Write a Python function to demonstrate Grouped-Query Attention (GQA) head expansion.",
    "How does static KV caching eliminate tensor allocation bottlenecks in high-concurrency decode?",
    "Describe the architectural innovations of Qwen 3.5 hybrid Gated DeltaNet and Attention.",
    "What are the computational complexity differences between causal linear attention and full attention?",
    "How does zero-sync dispatch eliminate host-device synchronization latency in serving?",
    "Write an efficient PyTorch implementation of RMSNorm with fused scale multiplication.",
    "Explain the mathematical formulation of speculative prompt lookup decoding.",
    "Detail how CUDA graphs eliminate kernel launch latency during auto-regressive decoding.",
    "Compare the throughput and memory footprint of 4-bit NF4 versus 8-bit FP8 weight quantization.",
    "How does prefix KV cache broadcasting accelerate multi-trajectory rollout generation in GRPO?",
    "Describe the memory access pattern differences between row-major and column-major attention weights.",
]


def resolve_prompts(
    prompt: Optional[Union[int, str, List[str]]],
    concurrent: int,
) -> List[str]:
    """
    Resolves prompt argument into a list of strings matching the concurrency count.
    Supports:
    - prompt=None or int: generates `concurrent` distinct benchmark prompts.
    - prompt="interger..." or text containing 'auto'/'generate': auto-generates prompts.
    - prompt="literal prompt": replicates across all concurrent rollout streams.
    - prompt=[list]: uses/cycles list to match concurrency.
    """
    if prompt is None:
        return [
            DEFAULT_BENCHMARK_PROMPTS[i % len(DEFAULT_BENCHMARK_PROMPTS)]
            for i in range(concurrent)
        ]

    if isinstance(prompt, int):
        return [
            DEFAULT_BENCHMARK_PROMPTS[i % len(DEFAULT_BENCHMARK_PROMPTS)]
            for i in range(concurrent)
        ]

    if isinstance(prompt, list):
        if not prompt:
            return resolve_prompts(None, concurrent)
        return [prompt[i % len(prompt)] for i in range(concurrent)]

    if isinstance(prompt, str):
        prompt_strip = prompt.strip()
        lower = prompt_strip.lower()
        if (
            prompt_strip.isdigit()
            or "auto" in lower
            or "interger" in lower
            or "integer" in lower
            or "generate" in lower
            or "match the number of concurrent" in lower
        ):
            return [
                DEFAULT_BENCHMARK_PROMPTS[i % len(DEFAULT_BENCHMARK_PROMPTS)]
                for i in range(concurrent)
            ]
        return [prompt_strip] * concurrent

    return [
        DEFAULT_BENCHMARK_PROMPTS[i % len(DEFAULT_BENCHMARK_PROMPTS)]
        for i in range(concurrent)
    ]


def train(
    test: bool = True,
    long_train: bool = False,
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
    concurrent: int = 8,
    prompt: Optional[Union[int, str, List[str]]] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 50,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    compile: bool = False,
    speculative: Optional[bool] = None,
    speculative_k: int = 3,
    mock_test: bool = False,
    use_vllm: bool = False,
    prep_and_logs: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified entry point for Sword training and concurrent serving benchmark.
    Usage for quick inference testing:
        sword.train(test=True, model="Qwen/Qwen3.5-0.8B", concurrent=8)

    Usage for preparation, environment validation, and 5-step test training:
        sword.train(prep_and_logs=True, B_model_port=11343, test=True)

    Usage for long-running infinite GRPO training:
        sword.train(
            long_train=True,
            test=False,
            dataset="huggingface_dataset_or_drive_link",
            model="model_id",
            B_model_port=11343,
            custom_checkpoint_save=True,
            is_save_and_upload_checkpoint=True,
            num_latest_checkpoint_save=2,
            upload_id="user/hfmodel",
            HF_TOKEN="...",
            resume_from_checkpoint="",
            lora_cfg=CONF,
            Argument=ARG,
            show_detail_logs=True,
            score_logs=True,
        )
    """
    # 0. Preparation and comprehensive system diagnostic logs mode
    if prep_and_logs:
        return _run_prep_and_logs(
            test=test,
            long_train=long_train,
            dataset=dataset,
            model=model,
            B_model_port=B_model_port,
            concurrent=concurrent,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            lora_cfg=lora_cfg,
            Argument=Argument,
            mock_test=mock_test,
            **kwargs,
        )

    if long_train or not test:
        from sword.long_trainer import run_long_train
        return run_long_train(
            dataset=dataset,
            model=model,
            B_model_port=B_model_port,
            custom_checkpoint_save=custom_checkpoint_save,
            is_save_and_upload_checkpoint=is_save_and_upload_checkpoint,
            num_latest_checkpoint_save=num_latest_checkpoint_save,
            upload_id=upload_id,
            HF_TOKEN=HF_TOKEN,
            resume_from_checkpoint=resume_from_checkpoint,
            lora_cfg=lora_cfg,
            Argument=Argument,
            show_detail_logs=show_detail_logs,
            score_logs=score_logs,
            auto_unload_B_model=auto_unload_B_model,
            B_model_keep_alive=B_model_keep_alive,
            **kwargs,
        )

    prompts = resolve_prompts(prompt, concurrent)

    if test:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        engine_name = "vLLM" if (use_vllm or kwargs.get("backend") == "vllm") else "Sword Speed Engine"

        print("=" * 72)
        print(f"{engine_name} - {concurrent}-Concurrency Serving Test")
        print("=" * 72)
        print(f"Model ID:            {model}")
        print(f"Concurrency:         {concurrent} concurrent streams")
        print(f"Max New Tokens:      {max_new_tokens}")
        print(f"Device:              {device_str} ({device_name})")
        print(f"Engine Backend:      {engine_name}")
        print(f"Quantization 4-bit:  {load_in_4bit}")
        print(f"Compilation:         {'Enabled' if compile else 'Disabled'}")
        print("=" * 72)

        if (use_vllm or kwargs.get("backend") == "vllm") and not mock_test:
            try:
                from vllm import LLM, SamplingParams
                print(f"[Sword] Running with vLLM engine (vllm==0.23.0 patched)...")

                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # In notebook environments, ensure spawn method and disable broken JIT flashinfer sampler
                os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
                os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
                os.environ.setdefault("VLLM_USE_FLASHINFER", "0")
                if kwargs.get("use_v1") is False:
                    os.environ["VLLM_USE_V1"] = "0"
                
                # Configure MoE backend if specified
                moe_backend = kwargs.get("moe_backend", None)
                if moe_backend:
                    print(f"[Sword] Configured vLLM MoE backend: {moe_backend}")

                # Auto-heal heterogeneous architectures (Gemma 4, etc.)
                try:
                    from transformers.configuration_utils import PretrainedConfig
                    PretrainedConfig.allow_global_per_layer_attribute_access = True
                except Exception:
                    pass

                hf_overrides = kwargs.get("hf_overrides", {}) or {}
                hf_overrides.setdefault("allow_global_per_layer_attribute_access", True)
                if isinstance(hf_overrides.get("text_config"), dict):
                    hf_overrides["text_config"].setdefault("allow_global_per_layer_attribute_access", True)
                else:
                    hf_overrides["text_config"] = {"allow_global_per_layer_attribute_access": True}

                llm_kwargs = {
                    "model": model,
                    "trust_remote_code": True,
                    "max_model_len": kwargs.get("max_model_len", 2048),
                    "max_num_seqs": kwargs.get("max_num_seqs", max(concurrent, 16)),
                    "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.8),
                    "dtype": kwargs.get("dtype", "auto"),
                    "hf_overrides": hf_overrides,
                }
                # Quantization configuration
                quant = kwargs.get("quantization", None)
                if quant:
                    llm_kwargs["quantization"] = quant
                elif load_in_8bit:
                    llm_kwargs["quantization"] = "fp8"
                elif load_in_4bit:
                    llm_kwargs["quantization"] = "bitsandbytes"
                    llm_kwargs["load_format"] = "bitsandbytes"

                if "kv_cache_dtype" in kwargs:
                    llm_kwargs["kv_cache_dtype"] = kwargs["kv_cache_dtype"]
                if kwargs.get("enforce_eager", False):
                    llm_kwargs["enforce_eager"] = True
                if moe_backend:
                    llm_kwargs["moe_backend"] = moe_backend

                try:
                    llm = LLM(**llm_kwargs)
                except TypeError:
                    # In case installed vLLM version does not accept some kwargs in constructor
                    llm_kwargs.pop("moe_backend", None)
                    try:
                        llm = LLM(**llm_kwargs)
                    except TypeError:
                        llm_kwargs.pop("hf_overrides", None)
                        llm = LLM(**llm_kwargs)

                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else -1,
                    max_tokens=max_new_tokens,
                )
                t0 = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params)
                elapsed = time.perf_counter() - t0
                responses = [out.outputs[0].text for out in outputs]
                total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
                tps = total_tokens / elapsed if elapsed > 0 else 0.0

                print("-" * 72)
                print("Generated Stream Outputs (vLLM):")
                print("-" * 72)
                for idx, (p, r) in enumerate(zip(prompts, responses)):
                    p_snip = (p[:55] + "...") if len(p) > 58 else p
                    r_snip = r.replace("\n", " ")
                    r_snip = (r_snip[:65] + "...") if len(r_snip) > 68 else r_snip
                    print(f"[Stream {idx + 1}] Prompt: {p_snip}")
                    print(f"           Output: {r_snip}")

                print("=" * 72)
                print(f"Concurrent Streams:      {concurrent}")
                print(f"Engine:                  vLLM (patched)")
                print(f"Total Generated Tokens:  {total_tokens}")
                print(f"Elapsed Time:            {elapsed:.2f} s")
                print(f"Aggregate Throughput:    {tps:.2f} tok/s")
                print("=" * 72)

                return {
                    "ok": True,
                    "mode": "test",
                    "engine": "vllm",
                    "model": model,
                    "concurrent": concurrent,
                    "prompts": prompts,
                    "responses": responses,
                    "total_tokens": total_tokens,
                    "elapsed_seconds": elapsed,
                    "aggregate_tps": tps,
                    "raw": outputs,
                }
            except ImportError:
                print("[Sword] vLLM is not installed or importable. Falling back to Sword Speed Engine...")

        if not mock_test and (torch.cuda.is_available() or kwargs.get("force_load", False)):
            from sword.loader import load_qwen_model
            from sword.server import FastQwenServer

            print(f"[Sword] Loading {model} with Sword Pure FlashAttention...")
            model_obj, tokenizer = load_qwen_model(
                model_name_or_path=model,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            server = FastQwenServer(
                model=model_obj,
                tokenizer=tokenizer,
                max_concurrency=concurrent,
                compile_decode=compile,
            )
        else:
            print("[Sword] Running with synthetic lightweight test harness...")
            from transformers import AutoConfig, AutoModelForCausalLM
            from sword.patcher import patch_qwen
            from sword.server import FastQwenServer

            class MockTokenizer:
                vocab_size = 1000
                pad_token_id = 0
                eos_token_id = 1
                def __call__(self, texts, padding=True, return_tensors="pt", **_kw):
                    max_len = 16
                    input_ids = torch.randint(2, 990, (len(texts), max_len))
                    attention_mask = torch.ones_like(input_ids)
                    return {"input_ids": input_ids, "attention_mask": attention_mask}
                def batch_decode(self, token_ids, skip_special_tokens=True):
                    return [f"Synthetic output for stream {i+1} ({token_ids.shape[1]} tokens generated)" for i in range(token_ids.shape[0])]

            tokenizer = MockTokenizer()
            cfg = AutoConfig.for_model(
                "qwen2",
                vocab_size=tokenizer.vocab_size + 10,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=1024,
            )
            model_obj = AutoModelForCausalLM.from_config(cfg)
            model_obj = patch_qwen(model_obj)
            server = FastQwenServer(model=model_obj, tokenizer=tokenizer, max_concurrency=concurrent)

        print(f"[Sword] Serving {len(prompts)} concurrent streams...")
        t0 = time.perf_counter()
        results = server.serve(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_speculative=speculative,
            speculative_k=speculative_k,
        )
        elapsed = time.perf_counter() - t0

        responses = results.get("responses", [])
        total_tokens = results.get("total_tokens", len(prompts) * max_new_tokens)
        tps = results.get("total_tps", total_tokens / elapsed if elapsed > 0 else 0.0)

        print("-" * 72)
        print("Generated Stream Outputs:")
        print("-" * 72)
        for idx, (p, r) in enumerate(zip(prompts, responses)):
            p_snip = (p[:55] + "...") if len(p) > 58 else p
            r_snip = r.replace("\n", " ")
            r_snip = (r_snip[:65] + "...") if len(r_snip) > 68 else r_snip
            print(f"[Stream {idx + 1}] Prompt: {p_snip}")
            print(f"           Output: {r_snip}")

        print("=" * 72)
        print(f"Concurrent Streams:      {concurrent}")
        print(f"Total Prompt Tokens:     {len(prompts) * 16}")
        print(f"Total Generated Tokens:  {total_tokens}")
        print(f"Elapsed Time:            {elapsed:.2f} s")
        print(f"Aggregate Throughput:    {tps:.2f} tok/s")
        print("=" * 72)

        return {
            "ok": True,
            "mode": "test",
            "model": model,
            "concurrent": concurrent,
            "prompts": prompts,
            "responses": responses,
            "total_tokens": total_tokens,
            "elapsed_seconds": elapsed,
            "aggregate_tps": tps,
            "raw": results,
        }

    # Training mode
    print(f"[Sword] Starting training pipeline for {model}...")
    from sword.train_math_grpo import run_standalone_math_grpo
    return run_standalone_math_grpo(**kwargs)


def _run_prep_and_logs(
    test: bool = True,
    long_train: bool = False,
    dataset: Union[str, List[Any], Dict[str, Any]] = "math-ai/TemplateGSM",
    model: str = "google/gemma-4-31B-it",
    B_model_port: int = 11343,
    concurrent: int = 8,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    lora_cfg: Optional[Dict[str, Any]] = None,
    Argument: Optional[Dict[str, Any]] = None,
    mock_test: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Diagnostic preparation, validation, and 5-step test training runner:
    1. Sets required environment variables (vLLM spawn, disable broken flashinfer sampler).
    2. Runs vllm_patcher bootstrap and prints step-by-step patch logs.
    3. Validates and tests all 4 authentic harnesses (OpenCode, Codex, Gemini CLI, Pi).
    4. Connects to Ollama on B_model_port, tests model tags and thinking activation.
    5. Prints system environment, GPU info, and library versions.
    6. Executes a 5-step simple test training run (2 code, 3 general/math) verifying
       vLLM generation with ngram speedup, loss, and gradient norm tracking.
    """
    # 1. Environment variables setup
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_USE_FLASHINFER"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Configure heterogeneous layer access
    try:
        from transformers.configuration_utils import PretrainedConfig
        PretrainedConfig.allow_global_per_layer_attribute_access = True
    except Exception:
        pass

    hf_overrides = kwargs.get("hf_overrides") or {
        "allow_global_per_layer_attribute_access": True,
        "text_config": {"allow_global_per_layer_attribute_access": True},
    }

    print("=" * 80)
    print(" SWORD PREP & DIAGNOSTIC SYSTEM LOGS")
    print("=" * 80)
    print(f"[*] Environment Variables Set:")
    print(f"    VLLM_USE_FLASHINFER_SAMPLER = {os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')}")
    print(f"    VLLM_USE_FLASHINFER         = {os.environ.get('VLLM_USE_FLASHINFER')}")
    print(f"    VLLM_WORKER_MULTIPROC_METHOD = {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')}")
    print(f"[*] Hugging Face Config Overrides: {hf_overrides}")

    # System and device telemetry
    print("\n[*] Hardware & Software Telemetry:")
    print(f"    Python:       {sys.version.split()[0]} ({sys.executable})")
    print(f"    PyTorch:      {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"    CUDA Avail:   {cuda_avail}")
    if cuda_avail:
        print(f"    GPU Device:   {torch.cuda.get_device_name(0)}")
        print(f"    GPU Count:    {torch.cuda.device_count()}")
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"    VRAM Info:    {free_b / (1024**3):.2f} GB free / {total_b / (1024**3):.2f} GB total")
    else:
        print(f"    Device:       CPU fallback")

    # 2. Prepare vLLM Patcher
    print("\n" + "-" * 80)
    print(" STEP 1: PREPARE vLLM PATCHER")
    print("-" * 80)
    patch_result = {}
    try:
        from sword.vllm_patch import vllm_patcher
        patch_result = vllm_patcher(verbose=True)
    except Exception as e_patch:
        print(f"[Sword Alert] vLLM patcher note: {e_patch}")
        patch_result = {"ok": False, "error": str(e_patch)}

    # 3. Harness Validation Checking (All 4 harnesses)
    print("\n" + "-" * 80)
    print(" STEP 2: HARNESS VALIDATION CHECK (ALL 4 HARNESSES)")
    print("-" * 80)
    from sword.harness import (
        ensure_harness_repo,
        get_harness,
        get_harness_system_prompt,
        WorkspaceSandbox,
    )
    harness_status: Dict[str, Any] = {}
    test_harnesses = ["opencode", "codex", "gemini_cli", "pi"]
    for h_name in test_harnesses:
        try:
            repo_ok = ensure_harness_repo(h_name)
            p_text = get_harness_system_prompt(h_name)
            h_inst = get_harness(h_name)
            harness_status[h_name] = {
                "repo_ready": repo_ok,
                "prompt_chars": len(p_text),
                "initialized": h_inst is not None,
                "class": h_inst.__class__.__name__ if h_inst else None,
            }
            print(f"    [{h_name.upper()}] Repo: {'OK' if repo_ok else 'Missing'} | "
                  f"Prompt: {len(p_text)} chars | Harness: {h_inst.__class__.__name__ if h_inst else 'None'}")
        except Exception as e_h:
            harness_status[h_name] = {"error": str(e_h)}
            print(f"    [{h_name.upper()}] Validation failed: {e_h}")

    # 4. Model B Check in Ollama
    print("\n" + "-" * 80)
    print(f" STEP 3: 'B' MODEL CHECK IN OLLAMA (Port {B_model_port})")
    print("-" * 80)
    from sword.b_model import ModelBJudge, ModelBJudgeConfig
    b_model_name = kwargs.get("B_model_name", "qwen3.6:35b-a3b")
    b_judge_cfg = ModelBJudgeConfig(
        host=kwargs.get("B_model_host", "127.0.0.1"),
        port=B_model_port,
        model_name=b_model_name,
        reasoning=kwargs.get("B_model_thinking", True),
        show_thinking=True,
        timeout=float(kwargs.get("B_model_timeout", 30.0)),
        auto_unload=kwargs.get("auto_unload_B_model", False),
    )
    b_judge = ModelBJudge(config=b_judge_cfg)
    b_diag = b_judge.check_connection()
    print(f"    Ollama URL:         {b_diag['url']}")
    print(f"    Reachable:          {b_diag['reachable']}")
    print(f"    Installed Models:   {b_diag['installed_models']}")
    print(f"    Target Model:       {b_model_name} (found={b_diag['model_found']})")
    if b_diag.get("error"):
        print(f"    Notice:             {b_diag['error']}")

    # Perform quick thinking call test on Model B if reachable
    b_eval_test = None
    if b_diag["reachable"]:
        print(f"    Executing quick thinking verification on {b_model_name}...")
        try:
            b_eval_test = b_judge.evaluate(
                problem="What is 2 + 2?",
                solution="4",
                reasoning_rule="Show deduction.",
                answer_rule="Final answer 4.",
                candidate_response="<think>2 plus 2 equals 4.</think>4",
            )
            print(f"    Model B Score Test: truth={b_eval_test.truth}, formatted={b_eval_test.formatted}, rule={b_eval_test.rule_reasoning_follow}")
        except Exception as e_beval:
            print(f"    Model B Eval Test Notice: {e_beval}")

    # 5. Build 5-sample balanced dataset satisfying dataset schema
    print("\n" + "-" * 80)
    print(" STEP 4: PREPARE 5-STEP TEST DATASET (2 Code + 3 General/Math)")
    print("-" * 80)
    test_5_samples = [
        # Sample 1: Code (OpenCode / Pi)
        {
            "problem": "Write a Python function `is_prime(n: int) -> bool` that returns True if n is prime and False otherwise.",
            "solution": "def is_prime(n: int) -> bool:\n    if n <= 1: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True",
            "reasoning_rule": "Enclose algorithmic logic inside <think> tags.",
            "answer_rule": "Provide clean python function in ```python ... ``` block.",
            "can_be_verify": True,
            "difficulty": "medium",
            "category": "code",
            "harness": "opencode",
        },
        # Sample 2: Code (Codex / Gemini)
        {
            "problem": "Write a Python function `reverse_words(s: str) -> str` that reverses the order of words in a sentence.",
            "solution": "def reverse_words(s: str) -> str:\n    return ' '.join(s.strip().split()[::-1])",
            "reasoning_rule": "State boundary handling in reasoning trace.",
            "answer_rule": "Return exact single function definition.",
            "can_be_verify": True,
            "difficulty": "easy",
            "category": "code",
            "harness": "codex",
        },
        # Sample 3: Math (Calculus / Algebra)
        {
            "problem": "Solve for x: 3*x + 15 = 42.",
            "solution": "9",
            "reasoning_rule": "Isolate x step by step.",
            "answer_rule": "State the exact numeric integer solution.",
            "can_be_verify": True,
            "difficulty": "easy",
            "category": "math",
            "harness": "gemini",
        },
        # Sample 4: General Reasoning (Logic / Deduction)
        {
            "problem": "If all roses are flowers and some flowers fade quickly, can we deduce that all roses fade quickly? Answer Yes or No.",
            "solution": "No",
            "reasoning_rule": "Analyze categorical syllogism validity.",
            "answer_rule": "State Yes or No clearly.",
            "can_be_verify": False,
            "difficulty": "medium",
            "category": "reasoning",
            "harness": "pi",
        },
        # Sample 5: Science / General knowledge
        {
            "problem": "What is the primary gas found in Earth's atmosphere?",
            "solution": "Nitrogen",
            "reasoning_rule": "Identify gas constituent percentages.",
            "answer_rule": "Give the name of the gas.",
            "can_be_verify": False,
            "difficulty": "easy",
            "category": "science",
            "harness": "base",
        },
    ]

    for idx, s in enumerate(test_5_samples, 1):
        print(f"    Sample {idx}: [{s['category'].upper()}] Harness: {s['harness']} | Problem: {s['problem'][:50]}...")

    # 6. Execute 5-step simple test training run
    print("\n" + "-" * 80)
    print(" STEP 5: EXECUTE 5-STEP TEST TRAINING RUN (Loss & Gradient Tracking)")
    print("-" * 80)
    test_arg = dict(Argument or {})
    test_arg.setdefault("steps", 5)
    test_arg.setdefault("concurrency_g", concurrent)
    test_arg.setdefault("max_seq_len", 16384)
    test_arg.setdefault("max_new_tokens", 256)
    test_arg.setdefault("save_steps", 10)

    # In diagnostic test mode, ensure test dataset is evaluated fresh without stale deduplication locks
    if test:
        test_arg.setdefault("allow_retest", True)

    # Configure 8-bit QLoRA and vLLM parameters
    train_model = model if model != "Qwen/Qwen3.5-0.8B" else "google/gemma-4-31B-it"
    effective_lora = lora_cfg or {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    from sword.long_trainer import run_long_train
    train_result = run_long_train(
        dataset=test_5_samples,
        model=train_model,
        B_model_port=B_model_port,
        B_model_name=b_model_name,
        B_model_thinking=kwargs.get("B_model_thinking", True),
        custom_checkpoint_save=True,
        is_save_and_upload_checkpoint=False,
        Argument=test_arg,
        lora_cfg=effective_lora,
        show_detail_logs=True,
        score_logs=True,
        auto_unload_B_model=False,
        # vLLM generation and memory settings
        vllm_quantization="fp8" if load_in_8bit else None,
        kv_cache_dtype="fp8",
        speculative_ngram=True,
        num_speculative_tokens=4,
        ngram_prompt_lookup_max=4,
        hf_overrides=hf_overrides,
        mock_test=mock_test,
        **kwargs,
    )

    print("\n" + "=" * 80)
    print(" SWORD PREP & TEST RUN SUMMARY")
    print("=" * 80)
    print(f"[*] Training Status:    {train_result.get('status')}")
    print(f"[*] Completed Steps:    {train_result.get('final_step')} / 5")
    print(f"[*] Visited Rows:       {train_result.get('visited_rows')} / 5")
    print(f"[*] Dataset SHA-256:    {train_result.get('dataset_sha256')}")
    print(f"[*] Checkpoint Directory: {train_result.get('checkpoint_dir')}")
    print("=" * 80 + "\n")

    return {
        "ok": train_result.get("status") in ("completed", "partial"),
        "patch_result": patch_result,
        "harness_status": harness_status,
        "b_model_diagnostic": b_diag,
        "train_result": train_result,
    }
