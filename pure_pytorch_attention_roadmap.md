# Pure-PyTorch Attention + Throughput Stack (No flash-attn / No vLLM)

Goal: replace `flash-attn` and `vllm` dependencies with pure PyTorch code for a
private RL/GRPO training loop, avoiding install-fragility across environments.

License note: PyTorch itself = BSD-3. flash-attn = BSD-3. Unsloth = Apache 2.0.
For a **private, non-distributed** project, license obligations barely apply —
but keep attribution comments in source files anyway as good practice.

---

## Build Order

### 1. Core attention op
- Use `torch.nn.functional.scaled_dot_product_attention` (SDPA).
- Built into PyTorch ≥2.0 (2.2+ better kernel coverage). Dispatches to a
  FlashAttention-2-equivalent kernel automatically on Ampere+ GPUs, bf16/fp16.
- Source to reference: [PyTorch SDPA docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- Force fast backend (fail loud instead of silently falling back to slow math):
```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 2. RoPE (rotary position embeddings)
- Not covered by SDPA — apply to Q/K before calling it.
- Must match your base model's exact RoPE convention (interleaved vs
  half-rotation) or outputs will be wrong.
- Source to reference: HuggingFace `transformers` model source for your
  specific architecture (e.g. `modeling_llama.py`, `modeling_qwen2.py`) —
  copy their `apply_rotary_pos_emb` function directly, it's small and exact.

### 3. GQA / MQA head expansion
- If KV heads < Q heads (most modern models), expand before SDPA:
```python
k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
```
- Or use `enable_gqa=True` in SDPA if your PyTorch version supports it (check
  release notes — added in newer 2.x versions).

### 4. Causal + padding mask construction
- `is_causal=True` alone doesn't handle padded batches.
- Build a combined mask (causal AND not-padding) and pass via `attn_mask=`.
- For variable-length RL rollouts, consider a block-diagonal mask to pack
  multiple sequences into one tensor without padding waste (mimics flash-attn's
  `varlen` API without needing the special kernel).

### 5. KV cache for decode
- Pre-allocate a fixed-size buffer per sequence; slice-assign new K/V each
  step. Avoid growing/concatenating tensors every token — this silently kills
  TPS.
- Attend new Q against the full cache each decode step.

### 6. `torch.compile` on the forward pass
- Highest ROI single change for decode speed (tiny per-token forward passes
  are overhead-bound, not compute-bound).
```python
model.forward = torch.compile(model.forward, mode="reduce-overhead")
```
- `reduce-overhead` uses CUDA graphs internally.
- Source: [PyTorch compile docs](https://pytorch.org/docs/stable/torch.compiler.html)

### 7. Batched rollout generation
- Don't generate one sequence at a time — batch rollouts together (static or
  semi-static batching is enough for bounded RL rollout sizes; you don't need
  full continuous batching like vLLM unless serving unpredictable concurrent
  traffic).
- Hand-roll: track active sequences in a batch, evict on EOS, refill from a
  queue if you want continuous batching later.

### 8. Precision
- Use bf16/fp16 for the model, not fp32 — halves memory bandwidth (decode is
  bandwidth-bound).
- Keep fp32 only where numerically necessary (loss/log-softmax accumulation).

### 9. Chunked/fused log-prob computation (GRPO-specific)
- Avoid materializing full `(batch, seq_len, vocab_size)` logits tensor when
  computing per-token log-probs for the policy ratio.
- Compute log-softmax + gather in chunks along seq_len or vocab dim.
- Source to reference: Unsloth's fused cross-entropy implementation, and
  [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) /
  [cut-cross-entropy](https://github.com/apple/ml-cross-entropy) — both
  Apache/MIT licensed, algorithmically portable to plain PyTorch loops even if
  you don't use their Triton kernels directly.
- Note: GRPO needs *per-token log-probs*, not just a reduced scalar loss — the
  standard "fused loss" recipes need adapting for this.

### 10. Gradient checkpointing
- `torch.utils.checkpoint` on transformer blocks — trades recompute for VRAM,
  standard PyTorch, ~80% of Unsloth's hand-written-backward savings with far
  less code.

### 11. LoRA / PEFT (if applicable)
- If policy and reference model share frozen base weights (LoRA), you avoid
  holding a full second copy of the model for GRPO's reference model — an
  architecture choice, not a kernel trick.
- Source: [HuggingFace PEFT](https://github.com/huggingface/peft) (also usable
  standalone, doesn't require the fragile pieces either).

---

## What to skip (not worth reimplementing)
- **PagedAttention** (vLLM's core kernel) — custom CUDA, hard to replicate
  efficiently in pure PyTorch; only matters at high-concurrency serving scale,
  not typical bounded-batch RL rollouts.
- **Sliding window attention** — only implement if your specific base model
  architecture uses it (e.g. Mistral-style); build a banded mask + `attn_mask=`
  if needed.
- **Speculative decoding** — algorithmically doable, but real throughput gains
  need careful draft/verify batching; treat as a later optimization, not core.

---

## Summary Table

| Feature | Effort | Source to study |
|---|---|---|
| SDPA fast kernel | trivial | PyTorch docs |
| RoPE | small | HF `transformers` modeling files |
| GQA expansion | trivial | PyTorch SDPA `enable_gqa` / manual repeat_interleave |
| Causal+padding mask | small | write yourself |
| KV cache | medium | write yourself |
| `torch.compile` | trivial | PyTorch compiler docs |
| Batched rollouts | medium | write yourself |
| bf16/fp16 | trivial | — |
| Chunked log-prob (GRPO) | medium-high | Unsloth, Liger-Kernel, cut-cross-entropy |
| Gradient checkpointing | small | `torch.utils.checkpoint` docs |
| LoRA/PEFT | small-medium | HuggingFace PEFT |
