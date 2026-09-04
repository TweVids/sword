"""
Benchmark script for the Sword Speed Engine.
Tests throughput (Tokens Per Second) across concurrency levels (1, 2, 4)
to verify performance targets on the target GPU architecture (e.g. Blackwell).
"""

import sys
import torch
from sword.model import FastTransformerModel, FastTransformerConfig
from sword.engine import SpeedEngine


def run_benchmark(concurrencies=[1, 2, 4], prompt_len=64, gen_tokens=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    print("=" * 65)
    print(" SWORD SPEED ENGINE BENCHMARK")
    print("=" * 65)
    print(f"Device:            {device}")
    if device == "cuda":
        print(f"GPU Name:          {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability:   {torch.cuda.get_device_capability(0)}")
    print(f"Data Type:         {dtype}")
    print(f"Prompt Length:     {prompt_len} tokens")
    print(f"Generated Tokens:  {gen_tokens} tokens per sequence")
    print("=" * 65)

    # Standard architecture config (~1.5B parameters model size for fast testing)
    config = FastTransformerConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=4096,
        dtype=dtype,
    )

    print("\nInitializing FastTransformerModel...")
    model = FastTransformerModel(config).to(device=device, dtype=dtype)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Model Parameters: {num_params:.2f}M")

    print("\nInitializing SpeedEngine with Static KV Cache...")
    engine = SpeedEngine(
        model=model,
        max_batch_size=max(concurrencies),
        max_seq_len=prompt_len + gen_tokens + 128,
        device=device,
        dtype=dtype,
    )

    # Warmup
    print("Warming up engine...")
    warmup_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
    engine.generate(warmup_ids, max_new_tokens=8)

    print("\n" + "-" * 65)
    print(f"{'Concurrency':<14}{'Decode Time':<15}{'Total Time':<14}{'Decode TPS':<12}{'Total TPS':<10}")
    print("-" * 65)

    for c in concurrencies:
        input_ids = torch.randint(0, config.vocab_size, (c, prompt_len), device=device)
        _, stats = engine.generate(input_ids, max_new_tokens=gen_tokens, temperature=0.0)

        print(
            f"{c:<14}"
            f"{stats['decode_time_s']:<15.3f}"
            f"{stats['total_time_s']:<14.3f}"
            f"{stats['decode_tps']:<12.1f}"
            f"{stats['total_tps']:<10.1f}"
        )

    print("-" * 65)
    print("Benchmark complete.\n")


if __name__ == "__main__":
    run_benchmark()
