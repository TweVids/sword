# Sword: Pure-PyTorch High-Throughput Attention & Generation Engine

A pure PyTorch implementation of high-throughput LLM attention and batched generation designed to eliminate fragile external C++/CUDA dependencies (such as standalone `flash-attn` wheels or `vllm`).

Specifically targeted for high-concurrency rollout generation (4+ concurrent streams) on modern cloud GPU architectures like the **NVIDIA PRO 6000 Blackwell Workstation**.

---

## Key Features

1. **Pure PyTorch FlashAttention**: Uses PyTorch's native `F.scaled_dot_product_attention` (SDPA) with `SDPBackend.FLASH_ATTENTION` / `EFFICIENT_ATTENTION`. Automatically targets hardware tensor cores without compilation fragility.
2. **Zero-Allocation Static KV Cache**: Pre-allocates static tensors `[num_layers, max_batch, num_kv_heads, max_seq_len, head_dim]`. Slice updates eliminate memory churn and fragmentation.
3. **Grouped-Query Attention (GQA)**: Native support for modern GQA ratios (e.g. 4:1 KV head compression).
4. **RoPE (Rotary Position Embeddings)**: Precomputed frequency tables with half-rotation embedding application.
5. **CUDA Graph / `torch.compile` Ready**: Supports `torch.compile(mode="reduce-overhead")` for the single-token decode loop, removing Python interpreter overhead.
6. **Batched Concurrency Engine**: Optimized for serving 4 concurrent streams with high throughput (>20 TPS).

---

## Hardware Target: NVIDIA Blackwell Workstation

* **Transformer Engine**: Accelerated matrix operations on `bfloat16`.
* **Bandwidth Optimization**: Generation decode is memory bandwidth-bound; the static KV cache avoids memory allocation latency.
* **PyTorch SDPA**: Automatically dispatches to the fastest FlashAttention-2/3 kernels on Blackwell.

---

## Quickstart

### 1. Environment Setup
The cloud environment (e.g. Modal / Colab with NVIDIA Blackwell) already provides PyTorch compiled for CUDA. Do not overwrite torch. Simply install the standalone dependencies:
```bash
pip install --no-deps -e .
# or install without modifying existing torch:
pip install -r requirements.txt
```

### 2. Run 4-Concurrency Serving & Benchmark
```bash
# Qwen 3.5 Dense Serving
python test/serve.py --model-id Qwen/Qwen3.5-9B-Instruct --load-in-4bit --max-new-tokens 64

# Ling-3.0-tiny (BF16, FP8, or INT4) High-Throughput Serving & RL Rollout:
python serve_moe.py --model inclusionAI/Ling-3.0-tiny --concurrency 4 --max-new-tokens 128

# Ling-3.0-tiny FP8 Tensor-Core Serving:
python serve_moe.py --model inclusionAI/Ling-3.0-tiny-fp8 --concurrency 8

# Multi-Trajectory RL Rollout Generation (PPO / GRPO):
python serve_moe.py --model inclusionAI/Ling-3.0-tiny --rl-rollouts 4 --max-new-tokens 256
```

---

## Supported Architectures

* **Dense & MoE Transformer Family:**
  * **Qwen Family:** Qwen2, Qwen2.5, Qwen3, Qwen3.5 (QK-norm, dual-projection gating).
  * **Hunyuan / HYV3:** MoE with fused FP8 tensor cores.
* **Hybrid Linear-MoE Family (Ling-3.0 Series):**
  * `inclusionAI/Ling-3.0-tiny` (BF16, 7.9B total params, 1.3B active).
  * `inclusionAI/Ling-3.0-tiny-fp8` (Pre-quantized FP8 for Blackwell / Hopper).
  * `inclusionAI/Ling-3.0-tiny-int4` (Compressed-tensors INT4).
  * **3:1 Alternating KDA-MLA Attention Stack:** 3 Kimi Delta Attention linear recurrent layers followed by 1 Multi-Head Latent Attention layer.
  * **Zero-Sync MoE Dispatch:** Eliminates host-device synchronization stalls (`.cpu().numpy()`), evaluating only activated experts out of 128.
  * **Native Thinking Mode:** Configurable per-request reasoning mode (`<think>`).
  * **RL Rollout Engine:** Built-in multi-trajectory parallel rollout generation for RLHF, PPO, and GRPO.

---

## Repository Structure

* `sword/`
  * `attention.py`: Pure PyTorch FlashAttention block with RoPE and GQA.
  * `kv_cache.py`: Pre-allocated StaticKVCache for zero-allocation decode.
  * `ling.py`: Specialized Ling-3.0-tiny Flash MLA, Zero-Sync MoE dispatch, FLA shims, and FastLingServer.
  * `model.py`: Transformer architecture with RMSNorm, SwiGLU, and precomputed RoPE.
  * `patcher.py`: Model patcher routing attention and MoE through Flash SDPA.
  * `loader.py`: Hardware-accelerated model loading for Qwen, MoE, and Ling-3.0 models.
  * `server.py`: High-concurrency serving engine.
  * `engine.py`: Batched rollout and decode engine.
* `serve_moe.py`: Serving and RL rollout entry point for Ling-3.0 and MoE models.
* `benchmark.py`: Throughput benchmarking suite.
* `test/test_ling.py`: Unit test suite verifying Ling-3.0 optimizations.
* `requirements.txt`: Environment package specifications.

