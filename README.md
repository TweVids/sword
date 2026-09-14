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
# Qwen3 MoE (Qwen3 30B A3B) High-Throughput Serving & RL Rollout:
python serve_moe.py --model Qwen/Qwen3-30B-A3B --concurrency 4 --max-new-tokens 128

# Qwen3 MoE Multi-Trajectory RL Rollout Generation (PPO / GRPO):
python serve_moe.py --model Qwen/Qwen3-30B-A3B --rl-rollouts 8 --max-new-tokens 256

# Run Comprehensive BEFORE vs AFTER Speedup Benchmark:
python serve_moe.py --model Qwen/Qwen3-30B-A3B --benchmark
```

---

## Supported Architectures

* **Dense & MoE Transformer Family:**
  * **Qwen3 MoE (specially Qwen3 30B A3B):** 128 experts, top-8 active routing, 8:1 GQA ratio (32 query heads, 4 KV heads), QK RMSNorm, Zero-Sync MoE dispatch, router logits instrumentation for GRPO aux loss.
  * **Qwen Family:** Qwen2, Qwen2.5, Qwen3, Qwen3.5 (QK-norm, dual-projection gating).
  * **Hunyuan / HYV3:** MoE with fused FP8 tensor cores.
* **Faster & Smarter KV Cache:**
  * **O(1) Metadata Reset:** Bypasses multi-gigabyte HBM zeroing between generation batches.
  * **Auto-Clear on New Rollouts:** Automatically resets pointers and masks when new rollout requests arrive.
  * **Prefix KV Broadcast:** Replicates prefilled prompt KV states across $G$ sibling trajectories in GRPO in O(1) time.
* **Unsloth RL Engine (GRPO / PPO):**
  * Built-in `FastLanguageModel` memory-efficient LoRA adapters and gradient checkpointing.
  * Fused negative cross-entropy chunked token logprob calculation (eliminates full $[B, L, V]$ tensor allocation).
  * Zero-copy reference model evaluation via LoRA adapter bypass (saves 50% VRAM).

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
* `serve_gym.py`: CLI server hosting the local Docker Coding Gym with public zrok tunneling.
* `benchmark.py`: Throughput benchmarking suite.
* `test/`: Unit test suite verifying attention, KV cache, RL engine, MoE, and Docker gym.
* `requirements.txt`: Environment package specifications.

---

## Docker Coding Gym for SWE / Coding RL (OpenSWE & Scale-SWE)

When training coding models with GRPO in cloud environments like Molab / Colab where Docker is not permitted, Sword provides a distributed **Client/Server Docker Gym** architecture:

```
[ Local Machine (Docker Host) ]                       [ Molab / Remote GPU (Trainer) ]
┌──────────────────────────────┐                      ┌──────────────────────────────┐
│  sword-coding-gym:latest     │                      │  Qwen3-30B-A3B (Unsloth GRPO)│
│  - Python 3.10, git, pytest  │                      │  - Multi-stream rollouts (G) │
│  - Warm container pool       │  ◄── zrok Tunnel ──► │  - RemoteCodingGym Client    │
│  - OpenSWE (eval.sh)         │      (HTTPS)         │  - Chunked GRPOLoss backward │
│  - Scale-SWE (F2P + P2P)     │                      │  - Zero Docker needed!       │
└──────────────────────────────┘                      └──────────────────────────────┘
```

### 1. On Local Machine: Host the Docker Gym
```bash
# Start Docker gym server and automatically expose via secure zrok tunnel:
python serve_gym.py --port 8765 --share-zrok

# Output:
# 🌐 PUBLIC ZROK ENDPOINT READY FOR MOLAB / COLAB:
# 👉 https://<token>.shares.zrok.io
```

### 2. In Molab / Colab: Train with Remote Execution Rewards
```python
import sword

# Connect to your local machine's Docker gym over the public zrok endpoint
trainer = sword.start_grpo(
    model_name_or_path="Qwen/Qwen3-30B-A3B",
    data=["path/to/openswe_oss.jsonl"],
    remote_gym_endpoint="https://<token>.shares.zrok.io",
    num_rollouts_per_prompt=8,
)

# Step continuous RL loop
trainer.step()
```


