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
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Benchmark
```powershell
python benchmark.py
```

### 3. Run 4-Concurrency Serving
```powershell
python serve.py --concurrency 4 --max-new-tokens 64
```

---

## Repository Structure

* `sword/`
  * `attention.py`: Pure PyTorch FlashAttention block with RoPE and GQA.
  * `kv_cache.py`: Pre-allocated StaticKVCache for zero-allocation decode.
  * `model.py`: Transformer architecture with RMSNorm, SwiGLU, and precomputed RoPE.
  * `engine.py`: High-throughput batched rollout and decode engine.
* `benchmark.py`: Throughput benchmarking suite (TPS across concurrency 1, 2, 4).
* `serve.py`: Serving entry point for concurrent batched generation.
* `requirements.txt`: Environment package specifications.
