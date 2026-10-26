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
Set `B_model_thinking=False` to explicitly disable judge thinking. When a thinking
request returns no usable scores, the judge retries the same model with
`think: false` and at least `B_model_num_predict_no_think` tokens (default `2048`)
before trying the smaller fallback. Omitting `think` would keep Qwen's default
thinking enabled.
The model-directory default is `/marimo/.ollama` in Marimo containers and
`~/.ollama/models` elsewhere. Pulls print streaming progress and report stalled
layers. Mock training disables automatic setup by default. If both judges fail,
training can still use deterministic rewards; `train_result["b_model"]["evaluation_counts"]` records
that explicitly and the real compatibility runner reports `ok=False`.

Training logs show `RewardStd` (within-group reward standard deviation), `AdvMax`
(largest absolute advantage), `TrainTurns`, `Judge`, and whether the optimizer
update ran. Identical rewards produce zero GRPO
advantages; a constant reward across steps still supplies no learning signal.
Zero-gradient steps preserve weights and optimizer state instead of applying
Adam momentum or weight decay. Loss and gradient metrics retain their precision;
zero loss alone can be valid when gradients are nonzero. Live diagnostics reject
runs with no reward-driven gradients. Results also report `optimizer_steps` and
`skipped_optimizer_steps`, separately from the number of completed loop steps.

The long trainer shares the bridge's optional historical reward baseline. Set
`Argument={"use_cross_step_baseline": False, ...}` for strict group-centered GRPO.
The coding diagnostic tasks use Model B's semantic truth scores; numeric answer
verification remains enabled for the math task. `b_model=False` uses local
heuristics without constructing an Ollama judge.

Instruction-tuned rollouts use the tokenizer's chat template and assistant
generation prefix. The bridge sends exact prompt token IDs to vLLM so BOS is
not inserted twice, and reserves template overhead in the harness prompt budget.
The five-step diagnostic defaults to `max_new_tokens=512`; an explicit
`Argument["max_new_tokens"]` still overrides it.

When rewards are uniform or an update is skipped, logs print up to two response previews and raw judge
JSON results, plus counts of unique, empty, and truncated responses and tool
calls. `score_logs=True` also writes `rollout_details.jsonl` in the output
directory. Diagnostics enable `trace_rollouts=True` to include complete first
and final responses, sampled token IDs, rollout logprobs, and tool history;
normal long training records previews unless tracing is explicitly enabled.
Blank output earns no formatting credit. Live judge responses missing truth,
formatting, reasoning-rule, or answer-rule scores are retried without thinking
and then with the configured fallback instead of being reported as successful
live scoring with missing values filled by zero.

Harnesses execute tools, return their results to the model, and continue for up
to eight turns per rollout. Successful tools alone do not end the interaction.
An explicit `Final Answer:` or `Final Solution:` with a result ends the loop; a labelled final code answer can
also finish after its implicit code checks pass. A rollout still calling tools
at the turn limit is marked incomplete and supplies an empty final answer to
the scorer. Set `Argument["harness_max_turns"]` to change the limit. Follow-up
prompts retain the original task and recent tool observations within the model
context budget; older turns are dropped when necessary.

Thinking-only responses, plans, and incomplete tool calls receive feedback and
continue the loop. Tool parsing accepts tagged, fenced, bare, and function-style
JSON with nested or stringified arguments. Tool-shaped text inside thinking is
excluded from execution. These policy-driven adapters execute real filesystem,
patch, Python, and shell operations using the upstream harness tool conventions.

Training uses every sampled assistant turn, including follow-up actions and final
answers, with each turn's exact prompt tokens and rollout logprobs. One group
advantage is assigned per trajectory and applied to its assistant tokens; tool
observations remain context. This avoids discarding the distinguishing later
turns when first tool calls are identical. Training work grows with the number
of turns; microbatching still bounds the forward/backward batch size.

Nonzero reward variation can still produce zero gradients when the trained
trajectories are identical, the adapters are disconnected, or importance weights
underflow. Skipped steps report the largest advantage, unique first turns and
trajectories, token counts, gradient-buffer connectivity, policy/rollout logprobs,
and zero importance-weight fraction. Tracing also saves the exact data for all
training turns in `rollout_details.jsonl`. The bridge startup stamp for this
change is `sword-grpo-bridge text-lora-connectivity/v6`; load this updated checkout
and restart the target runtime before rerunning.

Text GRPO attaches Gemma's adapters to its language decoder. Mixed projection
types retain their directly supported layers and redirect only the individual
wrappers, so vision-only `.linear` matches cannot displace the language layers.
Startup prints the selected scope and trainable tensor coverage and rejects a
model with no trainable language parameters. This setup is shared by diagnostic
training (`prep_and_logs=True, test=True`) and ordinary training (`test=False`).
Restart from the base model when replacing the older vision-only adapters.

Model output logging is enabled by default for detailed long training, including
healthy steps. `[ModelA Request]` and `[ModelA Output]` show the active backend,
sampled text, token counts, and truncation for initial and follow-up generations.
`[Agent Turn]` and `[Agent Tool]` show each iteration and its actual tool result.
`[ModelB Request]`, `[ModelB Output]`, and `[ModelB Scores]` show every judge
attempt and its returned JSON before parsing. Thinking-only judge responses show
a thinking preview, and failed retries or deterministic scoring are labelled.
`trace_rollouts=True` also saves judge attempts and every training turn.

Use `log_model_outputs=False` to disable the output bodies and tool previews.
`model_output_chars` defaults to `1200` and retains the beginning and end of long
responses; `0` prints the full response. Judge-only overrides are
`B_model_log_outputs` and `B_model_output_chars`. Keep `show_detail_logs=True`
to see turn labels and step summaries.

Each rollout creates a fresh directory under `<output_dir>/harness_workspaces`.
Files persist across that rollout's turns, and the directory is removed when
the harness exits. Set `Argument["keep_harness_workspaces"] = True` to retain
the directories for inspection. A harness constructed directly with an explicit
`workspace_dir` reuses that caller-owned project and never automatically wipes
it. Logs and rollout details record turn counts, stop reasons, workspace paths,
and whether cleanup succeeded; tracing also saves all assistant turns.

Sword suppresses the repetitive bitsandbytes `MatMul8bitLt: inputs will be cast
from ... to float16 during quantization` notice. Other warnings and errors remain
visible. The guard is reinstalled after dependency imports and at generation/
training boundaries, and handles logger-filter and warning-policy resets.
Negative GRPO policy loss is valid because it minimizes a negated
advantage-weighted objective. Its sign or proximity to zero does not establish
learning: check reward variation, nonzero gradients, optimizer updates, and truth
scores. Uniform-reward logs separate policy loss from the weighted entropy
penalty, which can produce a small nonzero total loss without reward signal.

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
