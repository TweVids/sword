"""
===================================================================================
Colab / Modal Cell Script: 4-Concurrency Serving & Benchmark for Qwen 3.5 9B
===================================================================================
Usage in Colab / Modal:
  !git clone https://github.com/TweVids/sword.git
  %cd sword
  !pip install -e .
  !python test/serve.py --model-id Qwen/Qwen3.5-9B-Instruct --load-in-4bit
===================================================================================
"""

import sys
import argparse
import torch

from sword import FastQwenServer, load_qwen_model


def run_colab_serving():
    parser = argparse.ArgumentParser(description="Sword 4-Concurrency Speed Engine Serving")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3.5-9B-Instruct",
        help="HuggingFace model repo or path (e.g. Qwen/Qwen3.5-9B-Instruct or Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Load with bitsandbytes 4-bit NF4")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of new tokens to generate per stream")
    parser.add_argument("--mock-test", action="store_true", help="Run with synthetic weights for quick verification")
    args = parser.parse_args()

    print("=" * 72)
    print(" SWORD SPEED ENGINE - QWEN 3.5 4-CONCURRENCY SERVING")
    print("=" * 72)
    print(f"Device:           {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU Accelerator:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"Model ID:         {args.model_id}")
    print(f"BitsAndBytes 4-bit: {args.load_in_4bit}")
    print(f"Concurrency:      4 streams")
    print("=" * 72)

    # 4 concurrent rollout prompts
    prompts = [
        "What are the main performance advantages of NVIDIA Blackwell architecture?",
        "Explain how pure-PyTorch SDPA achieves FlashAttention performance without external wheels.",
        "How does static KV caching eliminate tensor allocation bottlenecks in high-concurrency decode?",
        "Write a Python function to demonstrate Grouped-Query Attention (GQA) head expansion.",
    ]

    if not args.mock_test and torch.cuda.is_available():
        print(f"\n[*] Loading {args.model_id} with Unsloth / BitsAndBytes + Sword Pure FlashAttention...")
        model, tokenizer = load_qwen_model(
            model_name_or_path=args.model_id,
            load_in_4bit=args.load_in_4bit,
        )
        server = FastQwenServer(
            model=model,
            tokenizer=tokenizer,
            max_concurrency=4,
        )
    else:
        print("\n[*] Initializing synthetic test harness (local or --mock-test mode)...")
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        from sword.patcher import patch_qwen

        class MockTokenizer:
            vocab_size = 1000
            pad_token_id = 0
            eos_token_id = 1
            def __call__(self, texts, padding=True, return_tensors="pt", **kwargs):
                # Generates synthetic tokens without any download
                max_len = 16
                input_ids = torch.randint(2, 990, (len(texts), max_len))
                attention_mask = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}
            def batch_decode(self, token_ids, skip_special_tokens=True):
                return [f"Response sequence {i+1} (generated {token_ids.shape[1]} tokens)" for i in range(token_ids.shape[0])]

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
        model = AutoModelForCausalLM.from_config(cfg)
        model = patch_qwen(model)
        server = FastQwenServer(model=model, tokenizer=tokenizer, max_concurrency=4)

    # 1. Direct 4-stream concurrent serve
    print("\n[*] Serving 4 concurrent requests...")
    results = server.serve(prompts, max_new_tokens=args.max_new_tokens, temperature=0.7)
    
    print("\n--- Generated Sample Responses ---")
    for idx, (p, r) in enumerate(zip(prompts, results["responses"])):
        print(f"\n[Stream {idx+1}] Prompt: {p[:60]}...")
        print(f"[Stream {idx+1}] Output: {r[:100]}...")

    # 2. Comparative Before vs After Speed Benchmark
    print("\n[*] Running comparative benchmark (BEFORE vs AFTER)...")
    server.benchmark_before_after(prompts, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    run_colab_serving()
