"""
Sword High-Throughput MoE Serving & Benchmark Engine.

Target Architecture:
- Model: tencent/Hy-MT2-30B-A3B-FP8 (30B Total Params, 3B Active Tokens, 128 Experts, Top-8 Active)
- Precision: Native FP8 (Fast Tensor Cores on NVIDIA Blackwell / Ada / Hopper)
- Speedup: Sword Pure FlashAttention SDPA + Zero-Allocation Static KV Cache + Multi-Stream Async Pipeline

Usage:
  # 1. Run 4-stream concurrent benchmark (BEFORE vs AFTER speedup comparison):
  python serve_moe.py --benchmark

  # 2. Run high-speed serving with 4 concurrent streams:
  python serve_moe.py --concurrency 4 --max-new-tokens 64

  # 3. Custom model or sequence length:
  python serve_moe.py --model tencent/Hy-MT2-30B-A3B-FP8 --concurrency 8 --max-seq-len 4096
"""

import argparse
import sys
import os
import time
import torch

import sword
from sword import (
    FastMoEServer,
    setup_blackwell_environment,
    patch_moe,
)

# Apply GPU allocation and Blackwell Tensor Core settings
setup_blackwell_environment()


def main():
    parser = argparse.ArgumentParser(description="Sword MoE High-Throughput Serving & RL Rollout Engine")
    parser.add_argument(
        "--model",
        type=str,
        default="inclusionAI/Ling-3.0-tiny",
        help="HuggingFace model ID or local path (default: inclusionAI/Ling-3.0-tiny, supports -fp8, -int4, Hy-MT2)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent generation streams (default: 4)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens generated per stream (default: 128)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence context length for Static KV Cache (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (Ling-3.0 recommended: 1.0, 0.0 for greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling (Ling-3.0 recommended: 0.95)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (Ling-3.0 recommended: 20)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="Enable Ling-3.0 native thinking mode (default: True)",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_false",
        dest="enable_thinking",
        help="Disable Ling-3.0 thinking mode for fast routine answers",
    )
    parser.add_argument(
        "--rl-rollouts",
        type=int,
        default=0,
        help="Run high-speed RL rollout generation with N trajectories per prompt (e.g. 4 for PPO/GRPO)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comprehensive BEFORE vs AFTER speed benchmark",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive prompt generation session",
    )
    args = parser.parse_args()

    is_ling = any(x in args.model.lower() for x in ["ling", "bailing"])

    print("=" * 72)
    print(" ⚡ SWORD HIGH-THROUGHPUT MoE & RL ROLLOUT ENGINE ⚡")
    print("=" * 72)
    print(f"Model ID:          {args.model}")
    print(f"Architecture:      {'Bailing Hybrid (KDA + MLA + Sparse MoE)' if is_ling else 'MoE Standard'}")
    print(f"Target GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Concurrency:       {args.concurrency} concurrent streams")
    print(f"Max New Tokens:    {args.max_new_tokens} tokens/stream")
    print(f"Max Sequence Len:  {args.max_seq_len} tokens")
    print(f"Thinking Mode:     {'Enabled (<think> active)' if args.enable_thinking else 'Disabled'}")
    print("=" * 72)


    # 1. Initialize Server (loads FP8 model, patches attention with Flash SDPA, sets up Static KV Cache)
    print("\n[*] Initializing FastMoEServer with Sword Speed Engine...")
    t0 = time.perf_counter()
    server = FastMoEServer.from_pretrained(
        model_name_or_path=args.model,
        max_concurrency=args.concurrency,
        max_seq_len=args.max_seq_len,
    )
    load_time = time.perf_counter() - t0
    print(f"[*] FastMoEServer ready in {load_time:.2f}s.\n")

    # 2. Benchmark Mode
    if args.benchmark:
        prompts = [
            "Translate the following text into fluent Chinese: 'The architecture of NVIDIA Blackwell GPUs features second-generation Transformer Engines with native FP8 and FP4 Tensor Cores for massive LLM speedups.'",
            "Translate into English: '腾讯混元大模型采用了混合专家架构，在翻译任务和长文本推理中具有高吞吐与低延迟表现。'",
            "Explain how Mixture-of-Experts (MoE) routing works with 128 experts and top-8 active routing.",
            "Write a fast Python function using PyTorch scaled dot-product attention for static key-value caching.",
        ]
        # Match batch size to concurrency
        if len(prompts) < args.concurrency:
            prompts = (prompts * ((args.concurrency // len(prompts)) + 1))[: args.concurrency]
        elif len(prompts) > args.concurrency:
            prompts = prompts[: args.concurrency]

        results = server.benchmark_before_after(
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )
        return

    # 3. RL Rollout Mode (e.g. GRPO / PPO multi-trajectory generation)
    if args.rl_rollouts > 0:
        rl_prompts = [
            "Calculate 17 * 23 and output only the final result.",
            "Write a fast Python script for zero-copy tensor slicing.",
        ]
        num_trajectories = args.rl_rollouts
        total_streams = len(rl_prompts) * num_trajectories
        print(f"[*] Executing RL Rollout Phase: {len(rl_prompts)} prompts x {num_trajectories} trajectories = {total_streams} streams...")
        t0 = time.perf_counter()
        if hasattr(server, "generate_rollouts"):
            rollouts = server.generate_rollouts(
                prompts=rl_prompts,
                num_rollouts_per_prompt=num_trajectories,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        else:
            flat_prompts = [p for p in rl_prompts for _ in range(num_trajectories)]
            res = server.serve(flat_prompts, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            rollouts = [res["responses"][i * num_trajectories : (i + 1) * num_trajectories] for i in range(len(rl_prompts))]
        elapsed = time.perf_counter() - t0
        total_tokens = sum(len(r.split()) for group in rollouts for r in group)

        print("\n" + "=" * 72)
        print(" 🎯 RL ROLLOUT GENERATION SUMMARY (GRPO / PPO)")
        print("=" * 72)
        print(f"Prompts:                {len(rl_prompts)}")
        print(f"Trajectories / Prompt:  {num_trajectories}")
        print(f"Total Rollout Streams:  {total_streams}")
        print(f"Elapsed Time:           {elapsed:.3f} s")
        print(f"Rollout Output Speed:   {total_tokens / elapsed:.2f} words/sec")
        print("=" * 72)
        for p_i, group in enumerate(rollouts):
            print(f"\n[Prompt {p_i + 1}]: {rl_prompts[p_i]}")
            for t_i, traj in enumerate(group):
                print(f"  --> Trajectory {t_i + 1}: {traj[:120].strip()}...")
        return

    # 4. Standard Serving Mode
    test_prompts = [
        "Translate to Chinese: 'Sword is a pure-PyTorch high-throughput attention and generation engine.'",
        "Translate to English: '混合专家模型在保持高参数容量的同时，显著降低了每个token的计算成本。'",
        "Explain the advantage of FP8 Tensor Cores in Large Language Model inference.",
        "How does static KV caching eliminate memory reallocation stalls?",
    ]
    if len(test_prompts) < args.concurrency:
        test_prompts = (test_prompts * ((args.concurrency // len(test_prompts)) + 1))[: args.concurrency]
    elif len(test_prompts) > args.concurrency:
        test_prompts = test_prompts[: args.concurrency]

    print(f"[*] Serving {len(test_prompts)} concurrent requests with Sword Speed Engine...")
    serve_kwargs = {"prompts": test_prompts, "max_new_tokens": args.max_new_tokens, "temperature": args.temperature}
    if is_ling and hasattr(server, "format_prompt"):
        serve_kwargs["enable_thinking"] = args.enable_thinking
        serve_kwargs["top_p"] = args.top_p
        serve_kwargs["top_k"] = args.top_k

    results = server.serve(**serve_kwargs)


    print("\n" + "=" * 72)
    print(" 🚀 SERVING PERFORMANCE SUMMARY")
    print("=" * 72)
    print(f"Batch Concurrency:      {args.concurrency} streams")
    print(f"Total Tokens Generated: {results['total_tokens']} tokens")
    print(f"Latency:                {results['latency_s']:.3f} s")
    print(f"Throughput (Total):     {results['total_tps']:.2f} tokens/sec")
    print(f"Throughput (Per Stream):{results['stream_tps'][0]:.2f} tokens/sec")
    print("=" * 72)

    print("\n📄 Generated Responses:")
    for i, resp in enumerate(results["responses"]):
        print(f"\n[Stream {i+1}]:")
        print("-" * 50)
        print(resp.strip())
        print("-" * 50)

    # 4. Optional Interactive Session
    if args.interactive:
        print("\n" + "=" * 72)
        print(" 💬 INTERACTIVE MoE SERVING SESSION (type 'exit' to quit)")
        print("=" * 72)
        while True:
            try:
                user_prompt = input("\nEnter prompt > ")
                if not user_prompt.strip():
                    continue
                if user_prompt.strip().lower() in ["exit", "quit", "q"]:
                    break

                batch = [user_prompt]
                out = server.serve(batch, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                print(f"\nResponse ({out['total_tps']:.1f} tok/s):")
                print(out["responses"][0])
            except KeyboardInterrupt:
                break
        print("\n[Sword] Session ended.")


if __name__ == "__main__":
    main()
