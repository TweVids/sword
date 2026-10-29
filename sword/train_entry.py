import os
import time
from typing import Optional, Any, Dict, List, Union

import torch
from .defaults import DEFAULT_GRPO_MODEL, DEFAULT_INFERENCE_MODEL


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
    dataset: Optional[Union[str, List[Any], Dict[str, Any]]] = "math-ai/TemplateGSM",
    model: Optional[str] = None,
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
    load_in_8bit: Optional[bool] = None,
    compile: bool = False,
    speculative: Optional[bool] = None,
    speculative_k: int = 3,
    mock_test: bool = False,
    use_vllm: bool = False,
    prep_and_logs: bool = False,
    b_model: bool = True,
    setup: bool = False,
    model_lora_repo: str = "",
    train_from_lora_checkpoint: Optional[Union[int, str]] = None,
    gradio_logger: bool = False,
    chat_template_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified entry point for Sword training and concurrent serving benchmark.
    Usage for quick inference testing:
        sword.train(test=True, model="Qwen/Qwen3.5-0.8B", concurrent=8)

    Usage for setup and the live target-environment test suite:
        sword.train(prep_and_logs=True, setup=True, test=True, b_model=True)

    GRPO uses google/gemma-4-31B-it by default, with an 8-bit trainer and FP8
    rollouts. Model B uses qwen3.6:35b and retries scoring with qwen3.5:9b.
    Quick inference tests keep the small Qwen default. Explicit model and
    quantization settings override these defaults.

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
    grpo_mode = prep_and_logs or long_train or not test or bool(model_lora_repo)
    from .chat_templates import read_chat_template
    template_override = read_chat_template(chat_template_path)
    if template_override is not None:
        kwargs["_chat_template_override"] = template_override
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose either load_in_4bit or load_in_8bit.")
    if not model_lora_repo and train_from_lora_checkpoint not in (None, ""):
        raise ValueError("train_from_lora_checkpoint requires model_lora_repo")
    if model_lora_repo:
        if mock_test or kwargs.get("trainer_model") is not None or kwargs.get("tokenizer") is not None:
            raise ValueError("model_lora_repo requires loading the selected checkpoint; mock_test and injected models are incompatible")
        from .lora_source import resolve_lora_training_source
        source = resolve_lora_training_source(
            model_lora_repo, train_from_lora_checkpoint, base_model=model, token=HF_TOKEN,
            revision=kwargs.get("model_lora_revision"), base_revision=kwargs.get("model_revision"),
            cache_dir=kwargs.get("model_cache_dir"), source_cache_dir=kwargs.get("lora_cache_dir"),
            local_files_only=bool(kwargs.get("local_files_only", False)),
        )
        model = source.base_model
        kwargs["_lora_training_source"] = source
        kwargs["model_lora_repo"] = model_lora_repo
        kwargs["train_from_lora_checkpoint"] = train_from_lora_checkpoint
        kwargs["model_revision"] = source.manifest.get("base_revision")
    if model is None:
        model = DEFAULT_GRPO_MODEL if grpo_mode else DEFAULT_INFERENCE_MODEL
    if load_in_8bit is None:
        load_in_8bit = grpo_mode and model == DEFAULT_GRPO_MODEL and not load_in_4bit
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose either load_in_4bit or load_in_8bit.")
    if grpo_mode and load_in_8bit:
        kwargs.setdefault("vllm_quantization", "fp8")

    # Preparation and comprehensive system diagnostic logs mode.
    if prep_and_logs:
        return _run_prep_and_logs(
            test=test,
            long_train=long_train or bool(model_lora_repo),
            dataset=dataset,
            model=model,
            B_model_port=B_model_port,
            concurrent=concurrent,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            lora_cfg=lora_cfg,
            Argument=Argument,
            mock_test=mock_test,
            auto_unload_B_model=auto_unload_B_model,
            B_model_keep_alive=B_model_keep_alive,
            HF_TOKEN=HF_TOKEN,
            b_model=b_model,
            setup=setup,
            gradio_logger=gradio_logger,
            chat_template_path=chat_template_path,
            **kwargs,
        )

    setup_result = None
    if setup:
        from .runtime_setup import setup_environment
        setup_result = setup_environment(
            model=model, b_model=b_model, B_model_port=B_model_port, HF_TOKEN=HF_TOKEN,
            load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, mock_test=mock_test,
            auto_unload_B_model=auto_unload_B_model, B_model_keep_alive=B_model_keep_alive,
            **kwargs,
        )
        if not setup_result["ok"] or setup_result.get("restart_kernel_required"):
            status = "restart_required" if setup_result.get("restart_kernel_required") else "setup_failed"
            return {"ok": False, "status": status, "setup_result": setup_result,
                    "environment_validated": False}

    if long_train or not test or model_lora_repo:
        from sword.long_trainer import run_long_train
        result = run_long_train(
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
            mock_test=mock_test,
            b_model=b_model,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            gradio_logger=gradio_logger,
            chat_template_path=chat_template_path,
            **kwargs,
        )
        if setup_result is not None:
            result["setup_result"] = setup_result
        return result

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
                if template_override is not None:
                    outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                                       sampling_params=sampling_params, chat_template=template_override)
                else:
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
                    "setup_result": setup_result,
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

        if template_override is not None:
            from .grpo_bridge import format_rollout_prompt
            tokenizer.chat_template = template_override
            if not callable(getattr(tokenizer, "apply_chat_template", None)):
                raise ValueError("chat_template_path requires a tokenizer with apply_chat_template; use a real model for serving")
            prompts = [format_rollout_prompt(tokenizer, p)[0] for p in prompts]
            from .chat_templates import RenderedChatTokenizer
            server.tokenizer = RenderedChatTokenizer(tokenizer)
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
            "setup_result": setup_result,
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
    model: str = DEFAULT_GRPO_MODEL,
    B_model_port: int = 11343,
    concurrent: int = 8,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    lora_cfg: Optional[Dict[str, Any]] = None,
    Argument: Optional[Dict[str, Any]] = None,
    mock_test: bool = False,
    b_model: bool = True,
    setup: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Provision when requested and run component checks in the training environment."""
    from .diagnostics import run_target_diagnostics
    return run_target_diagnostics(
        test=test, long_train=long_train, dataset=dataset, model=model,
        B_model_port=B_model_port, concurrent=concurrent, load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit, lora_cfg=lora_cfg, Argument=Argument,
        mock_test=mock_test, b_model=b_model, setup=setup, **kwargs,
    )
