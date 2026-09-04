"""
===================================================================================
Colab / Modal / Marimo Cell Script: 8k Long-Context Fine-Tuning Benchmark
===================================================================================
Demonstrates how Sword Pure FlashAttention SDPA transforms O(N^2) memory scaling
into O(N) tiled SRAM execution for 8k (8192) context length training.

Usage:
  python test/finetune.py --model-id Qwen/Qwen3.5-9B-Instruct --seq-len 8192 --num-samples 5
===================================================================================
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sword import load_qwen_model
from sword.finetune import benchmark_finetune_8k


def run_finetune_benchmark():
    parser = argparse.ArgumentParser(description="Sword 8k Long-Context Fine-Tuning Benchmark")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3.5-9B-Instruct",
        help="HuggingFace model repo or path (e.g. Qwen/Qwen3.5-9B-Instruct)",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Load base model in 4-bit NF4")
    parser.add_argument("--seq-len", type=int, default=8192, help="Context length per row (default: 8192)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of training rows (default: 5)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per step (default: 1)")
    parser.add_argument("--mock-test", action="store_true", help="Run with small synthetic model for quick verification")
    args = parser.parse_args()

    if not args.mock_test and torch.cuda.is_available():
        print(f"\n[*] Loading {args.model_id} with BitsAndBytes 4-bit + Sword Pure FlashAttention...")
        model, tokenizer = load_qwen_model(
            model_name_or_path=args.model_id,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        print("\n[*] Initializing synthetic test model (local or --mock-test mode)...")
        from transformers import AutoConfig, AutoModelForCausalLM
        from sword.patcher import patch_qwen

        cfg = AutoConfig.for_model(
            "qwen2",
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=args.seq_len + 1024,
        )
        model = AutoModelForCausalLM.from_config(cfg)
        model = patch_qwen(model)
        tokenizer = None

    metrics = benchmark_finetune_8k(
        model=model,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    return metrics


if __name__ == "__main__":
    run_finetune_benchmark()
