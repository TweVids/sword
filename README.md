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

### 2. Run High-Throughput Serving & 8-Concurrency Benchmark
```bash
# Liquid AI LFM 2.5 24B A2B (Hybrid Conv1d + GQA + Sparse MoE) 8-Concurrency Serving:
python serve_moe.py --model LiquidAI/LFM2.5-24B-A2B --concurrency 8 --max-new-tokens 128

# Liquid AI LFM 2.5 24B A2B Comprehensive BEFORE vs AFTER Speedup Benchmark:
python serve_moe.py --model LiquidAI/LFM2.5-24B-A2B --benchmark --concurrency 8

# Qwen3 MoE (Qwen3 30B A3B) High-Throughput Serving & RL Rollout:
python serve_moe.py --model Qwen/Qwen3-30B-A3B --concurrency 8 --max-new-tokens 128

# Qwen3 MoE Multi-Trajectory RL Rollout Generation (PPO / GRPO):
python serve_moe.py --model Qwen/Qwen3-30B-A3B --rl-rollouts 8 --max-new-tokens 256
```

---

## Supported Architectures

* **Liquid AI Foundation Models (LFM2 / LFM2.5 MoE Family):**
  * **LFM 2.5 24B A2B (`LiquidAI/LFM2.5-24B-A2B`):** 40 layers total (30 Gated Short Convolution blocks + 10 Grouped-Query Attention layers), 64 experts with top-4 active routing, routed scaling factor = 1.0, expert bias.
  * **Pure-PyTorch Causal Conv1d from Scratch:** Zero external C++/CUDA extension dependencies (runs on Blackwell without `causal-conv1d` or `xformers`).
  * **Zero-Sync MoE Dispatch:** Eliminates 1,500+ host-device synchronization stalls (`.nonzero()` / `torch.where()`) per token with pure batched BMM dispatch.
  * **Static Hybrid Cache (`StaticLfmCache`):** Pre-allocates both static KV cache (for attention layers) and rolling conv buffer (for short convolution layers) for 8 concurrent streams with O(1) metadata reset.
  * **Unsloth Compatibility Layer:** Full support for Unsloth's `FastLanguageModel.from_pretrained`, `FastModel`, and LoRA adapters.
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

## Automatic Ollama Judge Setup

`sword.train(long_train=True, ...)` automatically installs Ollama on Linux if
needed, starts its API on `B_model_port` (default `11343`), and pulls
`B_model_name` (default `qwen3.6:35b`). The judge loads
when reward scoring begins, after rollout VRAM is released. If the primary model
cannot download, load, or evaluate a response, the same reward is retried with
`qwen3.5:9b`. The selected fallback remains active for subsequent rewards and is
unloaded before the training phase when `auto_unload_B_model=True` (the default).

Installation uses `~/.local/share/sword/ollama`; no systemd is needed. Linux amd64
and arm64 archives are supported. Missing `tar`/`zstd` are installed with apt-get
in root Debian/Ubuntu containers; other environments need those tools available.
Existing local or remote servers are reused. Their existing model-storage settings
continue to apply. Windows/macOS can use an installed Ollama or a remote server.

Optional training arguments are `B_model_auto_setup=False` for a caller-managed
server, `B_model_fallback_name=None` to disable the smaller-model retry,
`B_model_models_dir`, `B_model_startup_timeout`, `B_model_pull_stall_s`,
`B_model_timeout`, `B_model_num_ctx`, `B_model_force_gpu`, and `B_model_log_path`.
The model-directory default is `/marimo/.ollama` in Marimo containers and
`~/.ollama/models` elsewhere. Pulls print streaming progress and report stalled
layers. Mock training disables automatic setup by default. If both judges fail,
training can still use deterministic rewards; `train_result["b_model"]["evaluation_counts"]` records
that explicitly and the real compatibility runner reports `ok=False`.

### Target environment setup and test suite

Run this in the Linux GPU environment where you will train:

```python
import os
import sword

result = sword.train(
    prep_and_logs=True,
    setup=True,
    test=True,
    b_model=True,
    HF_TOKEN=os.environ.get("HF_TOKEN"),
)
assert result["ok"], result["test_suite"]
```

The GRPO target defaults to `google/gemma-4-31B-it`, with an 8-bit LoRA trainer
and FP8 vLLM rollouts. The primary Ollama judge defaults to `qwen3.6:35b`; a
primary download, load, or scoring failure retries the reward with `qwen3.5:9b`.
The successful fallback remains active for subsequent rewards. Override the
models with `model`, `B_model_name`, and `B_model_fallback_name`. Set
`load_in_8bit=False` for an unquantized trainer or `load_in_4bit=True` for 4-bit
LoRA training. Use `vllm_quantization` to override rollout quantization.

`setup=True` prepares missing Python dependencies and harness tools, installs
and patches `vllm==0.23.0`, fetches the four pinned harness repositories, caches
the target model, and starts/reuses Ollama on the configured port. It downloads
both judges before training. Repeated setup reuses installed packages, cached
models, and the running server. `vllm_version`, `model_cache_dir`,
`model_revision`, and `B_model_models_dir` let you select versions and storage.
Pass `HF_TOKEN` when the target model requires authenticated access. If setup
changes packages already imported into a notebook, it returns a restart request
and waits to run training until you restart the kernel and repeat the call.

`prep_and_logs=True, test=True` checks CUDA forward/backward, dependency imports,
vLLM imports and patches, the target architecture/tokenizer, and all four pinned
harness repositories. Each harness executes real write/edit/read or patch tools
and Python code in a temporary workspace. With `b_model=True`, the suite verifies
live reward scoring, GPU residency, VRAM release, and a controlled primary-load
failure followed by two actual fallback scoring requests. It then runs five GRPO
steps (two coding problems and three math/general problems), verifies finite
losses and gradient norms, checks LoRA policy updates, and reads back the adapter,
optimizer state, and dataset completion log. Diagnostic checkpoints stay local
under `checkpoints/diagnostics`; `Argument["output_dir"]` overrides the location.

The result includes `setup_result`, `patch_result`, `harness_status`,
`b_model_diagnostic`, `train_result`, and `test_suite`. Each entry in
`test_suite["checks"]` has a `passed`, `failed`, or `skipped` status and diagnostic
details. A missing dependency, failed harness, PyTorch rollout fallback, or
deterministic judge fallback makes the live suite return `ok=False`. Failed
preflight checks prevent large model loading and training.

Use `prep_and_logs=True, setup=True, test=False` to prepare and validate readiness
without starting GRPO. `sword.setup_environment()` also exposes provisioning
alone. With `setup=False`, diagnostics validate an already prepared environment
without package installation, repository cloning, or model downloads.
`b_model=False` disables Ollama setup/checks and model-based reward judging.
`mock_test=True` skips provisioning and live model checks, exercises the small
mock trainer and real harness tools, and reports `environment_validated=False`.
Quick `sword.train(test=True)` inference benchmarks retain the small Qwen default.
