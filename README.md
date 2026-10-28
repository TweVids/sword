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
Model B judges correctness, reasoning rules, and answer rules for both coding
and math tasks. Live math correctness uses the judge's full 0–1 score, including
equivalent expressions, units, and answers in words. Numeric verification supplies
the math correctness fallback when the judge is disabled or unavailable.
`b_model=False` uses local heuristics without constructing an Ollama judge.
Longer thinking and repeated calculations earn no extra credit; the judge assesses
concision when the dataset's reasoning rule requires it and respects requested
detailed derivations.

Instruction-tuned rollouts use the tokenizer's chat template and assistant
generation prefix. The bridge sends exact prompt token IDs to vLLM so BOS is
not inserted twice, and reserves template overhead in the harness prompt budget.
The live five-step diagnostic defaults to `max_new_tokens=2048` (64 in CPU mock mode); an explicit
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
change is `sword-grpo-bridge math-llm-judge/v8`; load this updated checkout
and restart the target runtime before rerunning.

Text GRPO attaches Gemma's adapters to its language decoder. Mixed projection
types retain their directly supported layers and redirect only the individual
wrappers, so vision-only `.linear` matches cannot displace the language layers.
Startup prints the selected scope and trainable tensor coverage and rejects a
model with no trainable language parameters. This setup is shared by diagnostic
training (`prep_and_logs=True, test=True`) and ordinary training (`test=False`).
Restart from the base model when replacing the older vision-only adapters.

The policy uses a **12,288-token context**, with **2,048 tokens per response**,
for both diagnostic and ordinary GRPO. Dataset reasoning and answer rules are
included in the initial task and every follow-up prompt; the reference solution
is reserved for evaluation. Explicit `Argument["max_seq_len"]` and
`Argument["max_new_tokens"]` override these defaults. Training microbatches
default to one turn. vLLM schedules at most two sequences concurrently and uses
2,048-token chunked prefill; all requested rollouts still run. Trainer offloading,
vLLM sleep, and judge unloading keep the GPU phases separate. Larger GPU contexts
still require validation on the target hardware.
CPU mock runs reserve 64 response tokens by default; explicit response limits
still apply in mock mode.

Datasets accept an optional `effort` column: `dynamic`, `low`, `medium`, or `high`.
Missing or blank values default to `dynamic`; invalid values raise an error.
The column is supported in Hugging Face datasets, in-memory rows, JSON/JSONL,
and CSV/TSV with column headers, and is included in dataset deduplication hashes.
For example: `{"problem": "What is 2+2?", "solution": "4", "effort": "low"}`.

Policy thinking is requested through the native chat template by default, with
the row's effort passed to the initial response and every tool follow-up.
Set `chat_template_path` to use an explicit local Jinja template:

```python
sword.train(
    test=False, dataset=rows,
    model_lora_repo="your-user/lora-model",
    train_from_lora_checkpoint=900,
    chat_template_path=r"D:\sword\chat_template.jinja",
)
```

The path is optional and also works when training directly from a base model.
Its contents override the tokenizer/checkpoint template, are validated before
model loading, participate in merged-tokenizer caching and dataset deduplication,
and are saved with subsequent checkpoints. A file in the working directory is
used only when its path is supplied. vLLM and PyTorch rollout prompts use the same
overridden tokenizer template, including harness follow-up turns.

All four harnesses accept native Gemma calls such as
`<|tool_call>call:write_file{path:<|"|>state.txt<|"|>,content:<|"|>hello<|"|>}<tool_call|>`
using each harness's supported tool names and argument schemas. The parser accepts
nested objects/arrays, booleans, nulls, numbers, and multiline quoted strings.
JSON tool calls remain supported. Tool-shaped text in thoughts or quoted
arguments is excluded from execution; incomplete/malformed native calls receive
normal repair feedback. Code/patch content inside arguments is executed only by
the requested tool, avoiding duplicate execution by code-block heuristics.

The supplied `chat_template.jinja` renders `<|think|>` and `Effort: <value>` in
the system turn. With `dynamic`, thinking stays enabled and the model decides
whether to emit a thought section; the dataset rules govern the judge's scores.
Use `A_model_thinking=False` to disable the request explicitly. The tokenizer's
chat template must support `enable_thinking` and `effort` to control them.
Reasoning markers are preserved, including a thought opener placed in the prompt,
so the harness does not execute tools mentioned inside thinking. Sampled thought
tokens remain part of GRPO. `[ModelA Request]` shows `thinking_requested` and
`effort`; each output shows `thought_present`, and `ThoughtTurns` counts nonempty
tagged thought sections across all trained turns. A request alone does not prove the model
actually emitted a thought section.

Model B uses a **32,768-token context** by default; `B_model_num_ctx` overrides
it. The judge receives the row's effort, complete dataset rules, all emitted
thoughts with their original markers, earlier assistant turns, the answer outside
thought channels separately, and measured tool counts. Unfinished harness turns
remain reasoning evidence even when no final answer was produced. The judge
checks thought boundaries against the rules and never counts a result present
only in thoughts as a final answer. Dynamic effort alone requires no thought section.
Its rubric evaluates both rules explicitly and avoids inventing an unstated
reasoning requirement. Ollama structured output requires all six score fields,
with numeric scores and a boolean `cheating`. Missing or invalid fields trigger
a repair request that names the problem; valid zero scores remain zero. Judge
request logs show the configured context, and responses record prompt token counts
when Ollama reports them.

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

Coding rows without a resolved harness now use a one-shot code executor. Coding
categories, an explicit `language`, dataset tests, or supported code fences enable
it; unfamiliar category names can use `language` or `tests` to select execution.
Python is the default; JavaScript (`node`) and shell (`bash`) require their runtime
on the training host. Set `auto_code_executor=False` to disable the fallback.
Execution alone does not establish answer correctness.

Dataset rows can define `tests` as Python assertion code, a list of assertion
strings, or a list of objects containing either `code` or `command`. Test objects
also support `name`, `timeout_sec` (default 10, maximum 300), `stdin`, and
`expected_stdout`. Command tests can use `{python}` for the active interpreter.
`test_commands` is an alias for a list of command tests. JSON strings in CSV/TSV
columns are accepted. `initial_files` maps relative filenames to initial contents.
Tests and initial files participate in dataset deduplication.

```python
rows = [{
    "problem": "Implement add(a, b) in Python.",
    "category": "custom_coding_task",
    "language": "python",
    "tests": [
        {"name": "addition", "code": "assert add(2, 3) == 5"},
        {"name": "negative", "code": "assert add(-2, 1) == -1"},
    ],
}]
sword.train(
    test=False, dataset=rows, gradio_logger=True,
    Argument={"concurrency_g": 8},
)
```

Python assertion tests load `solution.py` into their scope when it exists, so they
can call solution functions directly; command tests can run project test suites.
The same checks run for explicit harnesses before workspace cleanup. Dataset tests
are counted separately from model tool calls. With a live judge, the passed-test
fraction caps its truth score; without live judging, that fraction supplies the
truth score. No final answer still receives zero truth credit.

The judge receives exact tool success/failure totals and dataset-test totals,
plus bounded samples of runner-captured output. Defaults are a 4,000-character
JSON budget, up to four failed samples and two successful samples, with dataset
tests prioritized. Configure `judge_execution_chars` (512–16,000),
`judge_execution_failed_samples`, and `judge_execution_success_samples` through
`sword.train(...)`. Omitted samples are explicitly counted; full traces remain
available with `trace_rollouts=True`. stdout/stderr are treated as program output,
not judge instructions.

Install the optional dashboard dependency with `pip install -e ".[logger]"`.
`sword.train(gradio_logger=True, ...)` launches Gradio with `share=True` and prints
the public URL. The read-only dashboard shows live generation, current rollout
rewards, task totals/means, overall mean reward, a bounded score history, parsed
judge scores, raw judge JSON, and measured execution evidence. It displays up to
eight rollouts by default; `gradio_max_rollouts` changes only the display count,
while `Argument["concurrency_g"]` still controls training. `gradio_refresh_interval`
(default 1 second, minimum 0.5) and `gradio_port` are optional overrides.

vLLM streaming observes its existing engine loop and keeps final token IDs and
logprobs in the trainer flow. The PyTorch fallback also publishes live updates.
Generation updates are throttled, text and event buffers are bounded, and Gradio
rendering/launch run outside the training thread. The UI has no training controls
and dashboard failures do not stop training. `gradio_logger=False` loads no Gradio
dependency and installs no generation observer. The display remains available
after training finishes for inspection while the Python process remains alive.

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

To start GRPO from a published LoRA checkpoint, select its repository and folder:

```python
result = sword.train(
    test=False,
    dataset="your-user/training-data",
    model_lora_repo="your-user/gemma-lora",
    train_from_lora_checkpoint="checkpoint-900",
    HF_TOKEN=os.environ.get("HF_TOKEN"),
    Argument={"steps": 200, "output_dir": "checkpoints/grpo_from_900"},
)
```

`model_lora_repo` accepts a Hugging Face repo or local adapter repository. A
checkpoint can be a step number, `checkpoint-900`, a complete nested folder,
`latest`, or `root`; omitting it selects the repo root. Missing checkpoints and
ambiguous nested names fail explicitly. The source adapter's
`adapter_config.json` identifies the base model. An explicit `model` must agree
with that base, or supplies it if the adapter config has no base name. Remote
adapter and base revisions are pinned. `model_lora_revision` selects a source
repo revision when needed.

Sword downloads only the selected adapter and its config/tokenizer/template
assets, borrowing a root adapter config if the checkpoint lacks one. It loads
the base without k-bit quantization on CPU, merges the selected adapter with
PEFT's safe merge, and saves a full model under
`Argument["output_dir"]/merged_bases`. This one-time merge needs enough CPU RAM
and disk space for the full base. `merged_base_cache_dir`, `model_cache_dir`, and
`lora_cache_dir` let you choose storage locations. A matching merged base is
reused on subsequent runs. Merge failures stop training.

The trainer loads the merged base with the requested training quantization and
attaches a **new** trainable LoRA. vLLM loads the same merged-base directory and
receives that new adapter through the existing versioned `LoRARequest` flow.
The merged source weights stay frozen. For each problem, the policy generates
the rollout group, Model B scores complete thoughts and answers against the
dataset rules, GRPO computes advantages and updates the new adapter, and Sword
exports the updated adapter for the next rollout. Tool follow-ups, effort
control, logprobs, and GPU handovers use the existing bridge.

New checkpoints include `sword_lora_source.json` and the tokenizer/chat template
alongside the new adapter. The manifest records the original base and all source
adapters, so a later `model_lora_repo` run can reconstruct the merged base even
when the earlier machine's local directory is unavailable. Source repositories
must remain accessible. Dataset completion hashes also include the selected
source, so training a different checkpoint does not reuse another source's
completion log. `train_from_lora_checkpoint` starts a fresh GRPO adapter and
optimizer; `resume_from_checkpoint` restores a GRPO run and must use the same
merged-base manifest. Supplying a LoRA source selects GRPO, including when the
ordinary API's `test=True` default is left unchanged; `prep_and_logs=True`
continues to run its diagnostic workflow on that source.

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

Without a LoRA source, use `prep_and_logs=True, setup=True, test=False` to prepare
and validate readiness without starting GRPO. `sword.setup_environment()` also exposes provisioning
alone. With `setup=False`, diagnostics validate an already prepared environment
without package installation, repository cloning, or model downloads.
`b_model=False` disables Ollama setup/checks and model-based reward judging.
`mock_test=True` skips provisioning and live model checks, exercises the small
mock trainer and real harness tools, and reports `environment_validated=False`.
Quick `sword.train(test=True)` inference benchmarks retain the small Qwen default.
