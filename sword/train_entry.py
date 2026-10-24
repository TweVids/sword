import os
import sys
import time
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
    model: str = "Qwen/Qwen3.5-0.8B",
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
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified entry point for Sword training and concurrent inference testing.
    Usage:
        import sword
        # Using Sword Native Speed Engine:
        results = sword.train(test=True, model="Qwen/Qwen3.5-0.8B", concurrent=8, prompt=8)

        # Using patched vLLM engine:
        results = sword.train(test=True, model="Qwen/Qwen3.5-0.8B", concurrent=8, prompt=8, use_vllm=True)
    """
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

                # In notebook environments, ensure spawn method and optional V0 fallback
                os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
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

                llm_kwargs = {
                    "model": model,
                    "trust_remote_code": True,
                    "max_model_len": kwargs.get("max_model_len", 2048),
                    "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.8),
                    "dtype": kwargs.get("dtype", "auto"),
                    "hf_overrides": hf_overrides,
                }
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
