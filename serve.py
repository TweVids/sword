"""
Fast serving interface for Sword Speed Engine.
Processes batched requests with target 4+ concurrency.
"""

import argparse
import sys
import torch
from sword.model import FastTransformerModel, FastTransformerConfig
from sword.engine import SpeedEngine


def main():
    parser = argparse.ArgumentParser(description="Sword Speed Engine Serving")
    parser.add_argument("--concurrency", type=int, default=4, help="Target batch concurrency")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to generate per stream")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for decode loop")
    args = parser.parse_args()

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    print(f"[*] Initializing Sword Speed Engine on {args.device} (concurrency={args.concurrency})...")
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

    model = FastTransformerModel(config).to(device=args.device, dtype=dtype)
    engine = SpeedEngine(
        model=model,
        max_batch_size=args.concurrency,
        max_seq_len=2048,
        compile_decode=args.compile,
        device=args.device,
        dtype=dtype,
    )

    print("[*] Engine ready. Running 4 concurrent test prompts...")
    # Example 4 concurrent prompt inputs (represented as token IDs)
    sample_prompts = torch.randint(0, config.vocab_size, (args.concurrency, 32), device=args.device)

    output_ids, stats = engine.generate(
        sample_prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
    )

    print("\n--- Serving Execution Stats ---")
    print(f"Batch concurrency:      {stats['batch_size']}")
    print(f"Tokens per sequence:    {stats['tokens_per_stream']}")
    print(f"Total tokens generated: {stats['total_tokens']}")
    print(f"Prefill latency:        {stats['prefill_time_s'] * 1000:.2f} ms")
    print(f"Decode throughput:      {stats['decode_tps']:.2f} tokens/sec")
    print(f"Total throughput:       {stats['total_tps']:.2f} tokens/sec")
    print("--------------------------------")


if __name__ == "__main__":
    main()
