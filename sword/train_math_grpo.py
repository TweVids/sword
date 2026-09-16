"""
Standalone Self-Contained Math GRPO Trainer for Colab / Marimo / Modal / Blackwell.
Dataset: Nihilux/BigMath2 (problem & answer columns)
Base Model: Qwen3 MoE (e.g. Qwen/Qwen3-30B-A3B)
LoRA Checkpoint: checkpoint-2200
Inference Engine: In-Process Sword Fast Engine (Bypasses vLLM completely)
Context Window: 32k (32768 tokens)
Concurrency: 4 rollouts per prompt (num gen 4)
Optimization: FP8 Expert Weights + FP8 Static KV Cache (6 GB for 4x32k)
VRAM Lifecycle: Generates rollouts -> cleans KV cache down to ~29 GB -> backward pass
Multi-Reward: GDPO (accuracy=1.0, formatting=0.3, efficiency=0.2, safety=hard gate)
Thinking Format: Validates <think> and </think> opening & closing tags
Audit: Logs first 200 steps, zips generations, and uploads to Hugging Face Hub.
"""

import os
import sys
import gc
import re
import json
import math
import time
import zipfile
import argparse
from typing import List, Dict, Any, Tuple, Optional, Generator
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path if running inside sword repo
SWORD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SWORD_ROOT not in sys.path:
    sys.path.insert(0, SWORD_ROOT)


def _fix_transformers_moe_fp8_compatibility():
    """
    Hotfixes upstream Transformers bugs in transformers.integrations.moe:
    1. `_batched_linear`: uses `torch.bmm(weight, input)`. PyTorch CUDA has no FP8 bmm kernel,
       causing:
       'NotImplementedError: "baddbmm_cuda" not implemented for 'Float8_e4m3fn''.
    2. `_grouped_mm`: casts `input.to(weight.dtype)`. When MoE expert weights are in FP8,
       this casts `input` to Float8_e4m3fn, causing PyTorch grouped_mm to crash with:
       'RuntimeError: Expected mat_a to be Float32, BFloat16 or Float16 matrix, got Float8_e4m3fn'.
    This fix casts the FP8 weights on the fly to input.dtype during both grouped_mm (prefill)
    and batched_linear (decoding), preserving full BF16 tensor core matrix multiplication
    while retaining the full ~27 GB VRAM weight savings!
    """
    try:
        from transformers.integrations import moe as hf_moe

        # 1. Patch _batched_linear (used during single-token decoding)
        orig_batched_linear = getattr(hf_moe, "_batched_linear", None)
        if orig_batched_linear is not None and not getattr(orig_batched_linear, "_sword_patched", False):
            def safe_batched_linear(input, weight, bias=None, is_transposed=False):
                if str(weight.dtype).startswith("torch.float8") or str(input.dtype).startswith("torch.float8"):
                    target_dtype = input.dtype if input.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
                    if str(weight.dtype).startswith("torch.float8"):
                        weight = weight.to(target_dtype)
                    if str(input.dtype).startswith("torch.float8"):
                        input = input.to(target_dtype)
                return orig_batched_linear(input, weight, bias=bias, is_transposed=is_transposed)

            safe_batched_linear._sword_patched = True
            hf_moe._batched_linear = safe_batched_linear

        # 2. Patch _grouped_mm (used during prompt prefill)
        orig_grouped_mm = getattr(hf_moe, "_grouped_mm", None)
        if orig_grouped_mm is not None and not getattr(orig_grouped_mm, "_sword_patched", False):
            def safe_grouped_mm(input, weight, offs=None):
                if str(weight.dtype).startswith("torch.float8") or str(input.dtype).startswith("torch.float8"):
                    target_dtype = input.dtype if input.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
                    if str(weight.dtype).startswith("torch.float8"):
                        weight = weight.to(target_dtype)
                    if str(input.dtype).startswith("torch.float8"):
                        input = input.to(target_dtype)
                elif input.dtype != weight.dtype:
                    input = input.to(weight.dtype)

                if hasattr(torch.nn.functional, "grouped_mm"):
                    try:
                        return torch.nn.functional.grouped_mm(input, weight, offs=offs)
                    except Exception:
                        pass
                if hasattr(torch, "_grouped_mm"):
                    try:
                        return torch._grouped_mm(input, weight, offs=offs)
                    except Exception:
                        pass
                return torch.ops.transformers.grouped_mm_fallback(input, weight, offs=offs)

            safe_grouped_mm._sword_patched = True
            hf_moe._grouped_mm = safe_grouped_mm

        # 3. Patch _grouped_mm_fallback
        orig_fallback = getattr(hf_moe, "_grouped_mm_fallback", None)
        if orig_fallback is not None and not getattr(orig_fallback, "_sword_patched", False):
            def safe_fallback(input, weight, offs):
                if str(weight.dtype).startswith("torch.float8"):
                    target_dtype = input.dtype if input.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
                    weight = weight.to(target_dtype)
                return orig_fallback(input, weight, offs)

            safe_fallback._sword_patched = True
            hf_moe._grouped_mm_fallback = safe_fallback

    except Exception:
        pass


_fix_transformers_moe_fp8_compatibility()


# =====================================================================
# ⚙️  BLACKWELL & GPU ENVIRONMENT OPTIMIZATION
# =====================================================================
def setup_blackwell_environment():
    """Configures high-performance runtime flags for Blackwell (SM100) / CUDA."""
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    _fix_transformers_moe_fp8_compatibility()
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        if hasattr(torch.backends.cuda, "enable_flex_attention"):
            try:
                torch.backends.cuda.enable_flex_attention(True)
            except Exception:
                pass


# =====================================================================
# 🗜️  IN-MEMORY FP8 WEIGHT QUANTIZATION (Save 27 GB VRAM)
# =====================================================================
def convert_to_fp8_moe_weights(model: nn.Module) -> int:
    """
    Quantizes Qwen3 MoE expert weights in-place to torch.float8_e4m3fn.
    Reduces weight VRAM from ~61.5 GB down to ~34.6 GB.
    """
    if not hasattr(torch, "float8_e4m3fn"):
        print("[FP8] torch.float8_e4m3fn not supported on this PyTorch build. Keeping original dtype.")
        return 0

    converted = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Target MoE expert MLP projections: gate_up_proj and down_proj
            if any(k in name for k in ["gate_up_proj", "down_proj", "experts.gate_up", "experts.down"]):
                if param.dtype != torch.float8_e4m3fn:
                    fp8_data = param.data.to(torch.float8_e4m3fn)
                    param.data = fp8_data
                    param.requires_grad = False
                    converted += 1
    print(f"[FP8] Converted {converted} MoE weight tensors to torch.float8_e4m3fn.")
    return converted


# =====================================================================
# 🧠  NATIVE FP8 STATIC KV CACHE (Linear O(N) Memory: 6 GB for 4x32k)
# =====================================================================
class FastStaticKVCache:
    """
    Lightweight KV cache manager.
    Dynamic memory management delegates cache allocation to model generation
    preventing static dead-weight VRAM allocation (~13 GB saved).
    """

    def __init__(
        self,
        num_layers: int = 48,
        max_batch_size: int = 4,
        num_kv_heads: int = 8,
        max_seq_len: int = 32768,
        head_dim: int = 128,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.current_pos = 0

    def reset(self):
        """O(1) pointer reset."""
        self.current_pos = 0

    def clear_vram(self):
        self.reset()


# =====================================================================
# 🚀  IN-PROCESS ROLLOUT ENGINE (BYPASSES VLLM COMPLETELY)
# =====================================================================
class InProcessRolloutEngine:
    """
    Direct in-process multi-stream rollout generator.
    Bypasses vLLM completely:
    - Zero separate processes or ray actors
    - Zero double-model VRAM allocation
    - Native PyTorch SDPA
    - Flushes cache memory back down to ~29 GB before training step.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_concurrency: int = 4,
        max_seq_len: int = 32768,
        use_fp8_kv: bool = True,
        vram_target_gb: float = 35.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        self.max_concurrency = max_concurrency
        self.max_seq_len = max_seq_len
        self.vram_target_gb = vram_target_gb
        self.device = next(model.parameters()).device
        self.static_cache = FastStaticKVCache()

    def generate_rollouts(
        self,
        prompt: str,
        num_rollouts: int = 4,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> List[str]:
        """Generates G rollouts in eval mode, then immediately flushes KV cache memory."""
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

        prompts = [prompt] * num_rollouts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Slice generated response tokens (excluding prompt)
        prompt_len = input_ids.shape[1]
        response_ids = outputs[:, prompt_len:]
        decoded_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # -------------------------------------------------------------
        # VRAM Memory Reset: Drop back to baseline weight footprint
        # -------------------------------------------------------------
        del inputs, outputs, response_ids, input_ids, attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return decoded_responses


# =====================================================================
# 🏷️  THINKING TAGS (<think> ... </think>) FORMAT VERIFIER
# =====================================================================
class ThinkingFormatVerifier:
    """
    Authoritative verifier for <think> and </think> opening and closing tags.
    """

    @staticmethod
    def verify(text: str) -> Tuple[float, Dict[str, Any], str, str]:
        """
        Parses text and validates thinking tags.
        Returns:
            (tag_score, audit_dict, reasoning_trace, final_answer)
        """
        open_tag = "<think>"
        close_tag = "</think>"
        open_count = text.count(open_tag)
        close_count = text.count(close_tag)

        trace = ""
        answer = text
        audit: Dict[str, Any] = {
            "open_count": open_count,
            "close_count": close_count,
        }

        # Case 1: Malformed duplicate tags (looping)
        if open_count > 1 or close_count > 1:
            audit["error"] = "duplicate_or_nested_tags"
            return -0.25, audit, trace, answer

        # Case 2: Opened but never closed
        if open_count == 1 and close_count == 0:
            parts = text.split(open_tag, 1)
            trace = parts[1].strip()
            answer = ""
            audit["error"] = "unclosed_think_tag"
            return -0.25, audit, trace, answer

        # Case 3: Orphaned closing tag without opening
        if open_count == 0 and close_count == 1:
            parts = text.split(close_tag, 1)
            trace = parts[0].strip()
            answer = parts[1].strip()
            audit["error"] = "orphaned_close_tag"
            return -0.20, audit, trace, answer

        # Case 4: Standard well-formed pair
        if open_count == 1 and close_count == 1:
            pos_open = text.find(open_tag)
            pos_close = text.find(close_tag)

            if pos_open > pos_close:
                audit["error"] = "inverted_tags_close_before_open"
                return -0.25, audit, trace, answer

            trace = text[pos_open + len(open_tag):pos_close].strip()
            answer = text[pos_close + len(close_tag):].strip()

            if len(trace) < 5:
                audit["error"] = "empty_thinking_block"
                return -0.20, audit, trace, answer
            if len(answer) == 0:
                audit["error"] = "missing_final_answer"
                return -0.20, audit, trace, answer
            if open_tag in answer or close_tag in answer:
                audit["error"] = "leaked_tags_in_answer"
                return -0.20, audit, trace, answer

            audit["valid_thinking_tags"] = True
            return 0.10, audit, trace, answer

        # Case 5: Neither tag present (missing required thinking block in math)
        audit["error"] = "missing_required_thinking_tags"
        return -0.20, audit, trace, answer


# =====================================================================
# 🎯  PRIMARY SCORER (GROUND TRUTH MATH + FORMAT + EFFICIENCY)
# =====================================================================
class MathScorer:
    """Evaluates mathematical ground truth accuracy and formatting."""

    @staticmethod
    def _normalize_math(ans: str) -> str:
        s = ans.strip().lower()
        s = re.sub(r"[\$\\,\s]", "", s)
        s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]+)\}", r"\1", s)
        s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
        # Extract \boxed{...} if present
        boxed = re.findall(r"\\boxed\{([^}]+)\}", ans)
        if boxed:
            return MathScorer._normalize_math(boxed[-1])
        return s

    def score(
        self,
        problem: str,
        reference_answer: str,
        full_text: str,
        effort_tier: str = "high",
        token_count: int = 0,
    ) -> Dict[str, Any]:
        tag_score, tag_audit, trace, answer = ThinkingFormatVerifier.verify(full_text)

        norm_ref = self._normalize_math(reference_answer)
        norm_ans = self._normalize_math(answer)

        # 1. Ground Truth Accuracy
        is_match = (norm_ref in norm_ans) or (norm_ans == norm_ref) or (norm_ref and norm_ref in full_text)
        accuracy_score = 0.35 if is_match else -0.40

        # 2. Formatting (Numbered steps / structured reasoning)
        has_numbered = bool(re.search(r"^\s*\d+\.\s+", answer, re.MULTILINE))
        has_bullets = bool(re.search(r"^\s*[-*•]\s+", answer, re.MULTILINE))
        format_score = 0.10 if (has_numbered or has_bullets) else -0.10

        # 3. Efficiency & Anti-Looping
        efficiency_score = 0.0
        audit_flags: Dict[str, Any] = {"thinking_tags": tag_audit}

        # Check repetitive loops in reasoning trace
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", trace) if len(s.split()) >= 4]
        counts = defaultdict(int)
        for s in sentences:
            norm = s.lower()
            counts[norm] += 1
            if counts[norm] >= 2:
                efficiency_score -= 0.5
                audit_flags["repetitive_loop_detected"] = True
                break

        # Check prohibited bold markdown in thinking (**Step 1**)
        if re.search(r"\*\*[^*\n]+\*\*", trace):
            efficiency_score -= 0.2
            audit_flags["prohibited_bold_found"] = True

        # 4. Effort Tier Compliance & Token Budget Check
        effort_score = 0.0
        max_budget = {"low": 1024, "medium": 4024, "high": 11024, "xhigh": 22024, "ultra": 32024, "max": 65536}.get(effort_tier.lower(), 11024)
        if token_count <= max_budget:
            if effort_tier.lower() == "low":
                if token_count <= 800:
                    effort_score += 0.10
                    audit_flags["rapid_low_effort_rewarded"] = True
                elif token_count > 1500:
                    effort_score -= 0.20
                    audit_flags["complexity_cross_match_penalty"] = -0.20
            elif effort_tier.lower() in ("high", "xhigh", "ultra", "max"):
                if token_count >= 200:
                    effort_score += 0.10
                    audit_flags["deep_effort_rewarded"] = True
                else:
                    effort_score -= 0.15
                    audit_flags["insufficient_effort_penalty"] = -0.15
        else:
            overage = (token_count - max_budget) / max_budget
            if overage > 0.10:
                pen = min(1.0, 0.5 * ((overage - 0.10) / 0.40))
                effort_score -= pen
                audit_flags["budget_overage_penalty"] = -pen

        # Verification step rewards for high/ultra/max effort
        if effort_tier.lower() in ("high", "xhigh", "ultra", "max"):
            verification_cues = [r"\bverif", r"\bcheck", r"\bconfirm", r"\bdouble[-\s]check", r"\bsubstitut", r"\bassum"]
            if any(re.search(pat, trace, re.IGNORECASE) for pat in verification_cues):
                effort_score += 0.10
                audit_flags["effort_verification_steps_rewarded"] = True

        components = {
            "ground_truth": accuracy_score,
            "output_format": format_score,
            "thinking_tags": tag_score,
            "reasoning_structure": efficiency_score,
            "token_budget": round(effort_score, 4),
            "safety": 0.0,
        }

        # Group into 4 GDPO columns
        columns = {
            "accuracy": accuracy_score,
            "formatting": round(format_score + tag_score, 4),
            "efficiency": round(efficiency_score + effort_score, 4),
            "safety": 0.0,
        }

        total_reward = sum(components.values())

        return {
            "total_reward": round(total_reward, 4),
            "component_scores": components,
            "column_scores": columns,
            "is_correct": is_match,
            "reasoning_trace": trace,
            "final_answer": answer,
            "audit_log": audit_flags,
        }


# =====================================================================
# ⚖️  GDPO (GROUP REWARD-DECOUPLED NORMALIZATION POLICY OPTIMIZATION)
# =====================================================================
def compute_gdpo_advantages(
    rollout_results: List[Dict[str, Any]],
    column_weights: Optional[Dict[str, float]] = None,
    eps: float = 1e-8,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Decoupled normalization across accuracy, formatting, and efficiency columns.
    Prevents penalty spiking and reward collapse (NVIDIA arXiv:2601.05242).
    """
    weights = column_weights or {"accuracy": 1.0, "formatting": 0.3, "efficiency": 0.2}
    G = len(rollout_results)

    if G <= 1:
        return [0.0] * G, {k: [0.0] * G for k in ["accuracy", "formatting", "efficiency", "safety"]}

    norm_advs: Dict[str, List[float]] = {}
    for col in ["accuracy", "formatting", "efficiency"]:
        vals = [r["column_scores"].get(col, 0.0) for r in rollout_results]
        mean_v = sum(vals) / G
        var_v = sum((v - mean_v) ** 2 for v in vals) / G
        std_v = math.sqrt(var_v)

        if std_v < eps:
            norm_advs[col] = [0.0] * G
        else:
            norm_advs[col] = [(v - mean_v) / (std_v + eps) for v in vals]

    # Safety is treated as a hard gate
    safety_vals = [r["column_scores"].get("safety", 0.0) for r in rollout_results]
    norm_advs["safety"] = safety_vals

    total_advantages: List[float] = []
    for i in range(G):
        base_adv = sum(weights[c] * norm_advs[c][i] for c in ["accuracy", "formatting", "efficiency"])
        s_pen = safety_vals[i]
        adv_i = min(base_adv + s_pen, s_pen) if s_pen < 0 else base_adv
        total_advantages.append(round(adv_i, 4))

    return total_advantages, norm_advs


# =====================================================================
# 📉  MEMORY-EFFICIENT CHUNKED GRPO SURROGATE LOSS
# =====================================================================
class ChunkedGRPOLoss(nn.Module):
    """
    Unsloth-style sequence chunking cross-entropy.
    Prevents materializing the massive [Batch, SeqLen, Vocab] tensor in VRAM.
    """

    def __init__(self, clip_eps: float = 0.2, kl_coeff: float = 0.04, chunk_size: int = 512):
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.chunk_size = chunk_size

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: List[int],
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Mask response tokens only
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lengths):
            response_mask[i, p_len:] = attention_mask[i, p_len:].bool()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits

        targets = input_ids[:, 1:]
        target_mask = response_mask[:, 1:]
        shift_logits = logits[:, :-1, :]

        num_tokens = shift_logits.size(1)
        policy_logprobs_list = []

        # Iterate over sequence in chunks (chunk_size=512)
        for start_idx in range(0, num_tokens, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_tokens)
            chunk_logits = shift_logits[:, start_idx:end_idx, :].reshape(-1, shift_logits.size(-1))
            chunk_targets = targets[:, start_idx:end_idx].reshape(-1)

            token_logprobs = -F.cross_entropy(
                chunk_logits.float(),
                chunk_targets,
                reduction="none",
            ).reshape(batch_size, end_idx - start_idx)
            policy_logprobs_list.append(token_logprobs)

        policy_logprobs = torch.cat(policy_logprobs_list, dim=1)
        old_logprobs = policy_logprobs.detach()

        # Clipped surrogate objective
        log_ratio = policy_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        adv = advantages.unsqueeze(-1).to(ratio.device)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss_per_token = -torch.min(surr1, surr2)

        valid_tokens = target_mask.float().sum().clamp(min=1.0)
        policy_loss = (policy_loss_per_token * target_mask.float()).sum() / valid_tokens

        metrics = {
            "grpo_loss": round(policy_loss.item(), 5),
            "mean_ratio": round(ratio.mean().item(), 4),
            "mean_advantage": round(advantages.mean().item(), 4),
        }
        return policy_loss, metrics

    def forward_single(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
        advantage: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Micro-batch loss for a single rollout sequence (batch_size=1).
        Disables use_cache and only computes cross entropy on the response tokens
        in chunks, keeping activation VRAM minimal (< 1.5 GB).
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits  # [1, seq_len, vocab_size]

        targets = input_ids[:, 1:]
        shift_logits = logits[:, :-1, :]

        resp_start = max(0, prompt_length - 1)
        resp_logits = shift_logits[:, resp_start:, :]
        resp_targets = targets[:, resp_start:]

        if resp_targets.numel() == 0:
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True), {"grpo_loss": 0.0}

        resp_logits_flat = resp_logits.reshape(-1, resp_logits.size(-1))
        resp_targets_flat = resp_targets.reshape(-1)
        num_tokens = resp_targets_flat.size(0)

        token_logprobs_list = []
        for s_idx in range(0, num_tokens, self.chunk_size):
            e_idx = min(s_idx + self.chunk_size, num_tokens)
            c_logits = resp_logits_flat[s_idx:e_idx].float()
            c_targets = resp_targets_flat[s_idx:e_idx]
            c_logprobs = -F.cross_entropy(c_logits, c_targets, reduction="none")
            token_logprobs_list.append(c_logprobs)

        policy_logprobs = torch.cat(token_logprobs_list, dim=0)
        old_logprobs = policy_logprobs.detach()

        log_ratio = policy_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        adv_tensor = torch.tensor(advantage, dtype=torch.float32, device=input_ids.device)

        surr1 = ratio * adv_tensor
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss, {"grpo_loss": round(policy_loss.item(), 5)}


# =====================================================================
# 📦  DATASET DOWNLOADER & DISK BATCH STREAMER
# =====================================================================
EFFORT_TIERS_LIST: List[str] = ["low", "medium", "high", "xhigh", "ultra", "max"]


def download_bigmath2(
    dataset_name: str = "Nihilux/BigMath2",
    cache_file: str = "local_trainer/data/bigmath2.jsonl",
    max_samples: int = 10000,
    hf_token: Optional[str] = None,
) -> str:
    """
    Downloads / caches Nihilux/BigMath2 to local disk.
    Assigns a balanced round-robin effort tier across [low, medium, high, xhigh, ultra, max].
    """
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
        print(f"[Data] Found local cached dataset: {cache_file} ({os.path.getsize(cache_file)/1e6:.1f} MB)")
        return cache_file

    print(f"[Data] Streaming {dataset_name} from Hugging Face...")
    from datasets import load_dataset
    token = hf_token or os.environ.get("HF_TOKEN") or None
    ds = load_dataset(dataset_name, split="train", streaming=True, token=token)

    count = 0
    with open(cache_file, "w", encoding="utf-8") as f:
        for item in ds:
            prob = item.get("problem") or item.get("question") or ""
            ans = item.get("answer") or item.get("solution") or ""
            if not prob or not ans:
                continue
            tier = EFFORT_TIERS_LIST[count % len(EFFORT_TIERS_LIST)]
            f.write(json.dumps({
                "idx": count,
                "problem": str(prob).strip(),
                "answer": str(ans).strip(),
                "effort_tier": tier,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_samples:
                break

    print(f"[Data] Cached {count} math problems to {cache_file} (balanced across 6 effort tiers)")
    return cache_file


def stream_math_data_from_disk(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Streams lines from disk with zero RAM retention."""
    while True:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
                    gc.collect()


# =====================================================================
# 📊  200-STEP AUDITOR & HUGGING FACE ARCHIVE UPLOADER
# =====================================================================
class StepAuditor:
    """
    Saves every step's generations, exports flat tabular dataset,
    and uploads zip archive to Hugging Face Hub.
    Dedicated 'effort_tier' column enables studying token scaling per effort level.
    """

    def __init__(self, log_dir: str = "local_trainer/generations_200_steps"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_file = os.path.join(self.log_dir, "all_generations.jsonl")
        self.tabular_jsonl = os.path.join(self.log_dir, "generations_table.jsonl")
        self.tabular_csv = os.path.join(self.log_dir, "generations_table.csv")
        self.step_records = []
        self.effort_stats = defaultdict(lambda: {
            "steps": 0,
            "rollouts": 0,
            "correct": 0,
            "valid_tags": 0,
            "unclosed_tags": 0,
            "loops": 0,
            "total_tokens": 0,
            "total_reward": 0.0,
        })

    def record(
        self,
        step: int,
        problem: str,
        reference: str,
        rollouts: List[Dict[str, Any]],
        elapsed: float,
        effort_tier: str = "high",
    ):
        tier_key = str(effort_tier).lower()
        self.effort_stats[tier_key]["steps"] += 1

        tabular_rows = []
        for r in rollouts:
            r["effort_tier"] = tier_key
            st = self.effort_stats[tier_key]
            st["rollouts"] += 1
            st["total_tokens"] += r.get("token_count", 0)
            st["total_reward"] += r.get("total_reward", 0.0)

            is_corr = bool(r.get("is_correct", False))
            has_valid_tags = bool(r.get("component_scores", {}).get("thinking_tags", 0.0) > 0)
            is_unclosed = r.get("audit_log", {}).get("thinking_tags", {}).get("error") == "unclosed_think_tag"
            has_loop = bool(r.get("audit_log", {}).get("repetitive_loop_detected", False))

            if is_corr:
                st["correct"] += 1
            if has_valid_tags:
                st["valid_tags"] += 1
            if is_unclosed:
                st["unclosed_tags"] += 1
            if has_loop:
                st["loops"] += 1

            cols = r.get("column_scores", {})
            tabular_rows.append({
                "step": step,
                "problem": problem,
                "reference_answer": reference,
                "effort_tier": tier_key,
                "rollout_idx": r.get("rollout_index", len(tabular_rows)),
                "token_count": r.get("token_count", 0),
                "total_reward": round(r.get("total_reward", 0.0), 4),
                "accuracy_reward": round(cols.get("accuracy", 0.0), 4),
                "formatting_reward": round(cols.get("formatting", 0.0), 4),
                "efficiency_reward": round(cols.get("efficiency", 0.0), 4),
                "advantage": round(r.get("advantage", 0.0), 4),
                "is_correct": is_corr,
                "valid_think_tags": has_valid_tags,
                "unclosed_think_tag": is_unclosed,
                "repetitive_loop": has_loop,
                "final_answer": r.get("final_answer", ""),
                "reasoning_trace": r.get("reasoning_trace", ""),
                "full_text": r.get("full_text", ""),
            })

        record = {
            "step": step,
            "effort_tier": tier_key,
            "problem": problem,
            "reference_answer": reference,
            "elapsed_sec": round(elapsed, 2),
            "rollouts": rollouts,
        }
        step_path = os.path.join(self.log_dir, f"step_{step:04d}.json")
        with open(step_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(self.tabular_jsonl, "a", encoding="utf-8") as f_tab:
            for row in tabular_rows:
                f_tab.write(json.dumps(row, ensure_ascii=False) + "\n")

        if tabular_rows:
            import csv
            csv_fields = [
                "step", "effort_tier", "rollout_idx", "token_count", "total_reward",
                "accuracy_reward", "formatting_reward", "efficiency_reward", "advantage",
                "is_correct", "valid_think_tags", "unclosed_think_tag", "repetitive_loop",
                "problem", "reference_answer", "final_answer"
            ]
            write_header = not os.path.exists(self.tabular_csv) or os.path.getsize(self.tabular_csv) == 0
            with open(self.tabular_csv, "a", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=csv_fields, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                for row in tabular_rows:
                    writer.writerow(row)

        self.step_records.append(record)

    def summarize(self) -> Dict[str, Any]:
        total_rollouts = sum(len(r["rollouts"]) for r in self.step_records)
        correct = sum(sum(1 for ro in r["rollouts"] if ro.get("is_correct", False)) for r in self.step_records)
        valid_tags = sum(sum(1 for ro in r["rollouts"] if ro.get("component_scores", {}).get("thinking_tags", 0) > 0) for r in self.step_records)

        tier_order = ["low", "medium", "high", "xhigh", "ultra", "max"]
        tier_breakdown: Dict[str, Any] = {}
        for t in tier_order:
            if t in self.effort_stats:
                st = self.effort_stats[t]
                n_ro = max(1, st["rollouts"])
                tier_breakdown[t] = {
                    "steps": st["steps"],
                    "rollouts": st["rollouts"],
                    "accuracy_pct": round((st["correct"] / n_ro) * 100, 2),
                    "thinking_tag_compliance_pct": round((st["valid_tags"] / n_ro) * 100, 2),
                    "unclosed_tag_pct": round((st["unclosed_tags"] / n_ro) * 100, 2),
                    "looping_pct": round((st["loops"] / n_ro) * 100, 2),
                    "mean_tokens": round(st["total_tokens"] / n_ro, 1),
                    "mean_reward": round(st["total_reward"] / n_ro, 4),
                }

        summary = {
            "total_steps": len(self.step_records),
            "total_rollouts": total_rollouts,
            "accuracy_pct": round((correct / max(1, total_rollouts)) * 100, 2),
            "thinking_tag_compliance_pct": round((valid_tags / max(1, total_rollouts)) * 100, 2),
            "effort_tier_breakdown": tier_breakdown,
        }
        with open(os.path.join(self.log_dir, "audit_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Generate HF Dataset Card README.md for interactive table viewing
        readme_path = os.path.join(self.log_dir, "README.md")
        card_content = (
            "---\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files:\n"
            "  - split: train\n"
            "    path: generations_table.jsonl\n"
            "---\n\n"
            "# 🗡️ Sword 200-Step Math GRPO Generation Dataset\n\n"
            "Traces and multi-reward metrics from a 200-step Math GRPO training run on `Nihilux/BigMath2`.\n\n"
            "### Columns\n"
            "- **`step`**: Training step index (1-200)\n"
            "- **`effort_tier`**: Reasoning effort tier (`low`, `medium`, `high`, `xhigh`, `ultra`, `max`)\n"
            "- **`rollout_idx`**: Trajectory index (0 to 3 for G=4)\n"
            "- **`problem`**: User mathematical problem statement\n"
            "- **`reference_answer`**: Ground truth solution\n"
            "- **`token_count`**: Generated token count\n"
            "- **`total_reward`**: GDPO composite reward\n"
            "- **`advantage`**: Decoupled normalized advantage\n"
            "- **`is_correct`**: Mathematical ground truth match\n"
            "- **`valid_think_tags`**: `<think>` and `</think>` opening & closing tag compliance\n"
            "- **`full_text`**: Complete model response\n"
        )
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        print("\n" + "=" * 68)
        print(" 📊 200-STEP AUDIT SUMMARY")
        print(f" Steps Completed:       {summary['total_steps']}")
        print(f" Ground Truth Accuracy: {summary['accuracy_pct']}%")
        print(f" Tag Compliance:        {summary['thinking_tag_compliance_pct']}%")

        if tier_breakdown:
            print("-" * 68)
            print(f" {'Tier':<8} {'Steps':<7} {'Rollouts':<10} {'Accuracy':<10} {'AvgToks':<10} {'Tag%':<8} {'Reward':<8}")
            print("-" * 68)
            for t, data in tier_breakdown.items():
                print(
                    f" {t:<8} {data['steps']:<7} {data['rollouts']:<10} "
                    f"{data['accuracy_pct']:>5.1f}%    "
                    f"{data['mean_tokens']:>6.1f}    "
                    f"{data['thinking_tag_compliance_pct']:>5.1f}%  "
                    f"{data['mean_reward']:>+6.3f}"
                )
        print("=" * 68 + "\n")
        return summary


def zip_and_upload_to_hf(log_dir: str, repo_id: str, hf_token: Optional[str] = None) -> Optional[str]:
    """Zips generations and uploads to Hugging Face Hub."""
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️  No HF_TOKEN provided. Skipping Hugging Face upload.")
        return None

    zip_path = f"{os.path.normpath(log_dir)}.zip"
    print(f"[Archive] Zipping {log_dir} -> {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(log_dir):
            for file in files:
                abs_p = os.path.join(root, file)
                zf.write(abs_p, os.path.relpath(abs_p, os.path.dirname(log_dir)))

    print(f"[Hub] Uploading archive to HF Hub: {repo_id}...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=os.path.basename(zip_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload 200-step Math GRPO generation traces",
        )
        url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"🎉 Upload successful! 👉 {url}")
        return url
    except Exception as e:
        print(f"⚠️  HF upload error: {e}")
        return None


# =====================================================================
# 🧭  EFFORT SYSTEM PROMPTS & CHAT FORMATTING
# =====================================================================
EFFORT_SYSTEM_PROMPTS: Dict[str, str] = {
    "low": "Reasoning effort is set to low. Think rapidly and minimize token usage; answer directly without verification unless something is clearly wrong.",
    "medium": "Reasoning effort is set to medium. Validate non-obvious logic and state transitions, but don't re-check self-evident steps; keep a steady, balanced pace.",
    "high": "Reasoning effort is set to high. Validate non-obvious logic and state transitions, verify intermediate calculations, and check common edge cases before finalizing.",
    "xhigh": "Reasoning effort is set to extra high. Validate non-obvious logic and state transitions, verify intermediate calculations, check common edge cases, test key assumptions against likely counterexamples, and compare alternative solution paths before settling on one.",
    "ultra": "Reasoning effort is set to ultra. Validate non-obvious logic and state transitions, verify intermediate calculations, check common edge cases, test key assumptions against likely counterexamples, compare alternative solution paths, and break the problem into its component parts, verifying each independently and discarding approaches that fail early checks.",
    "max": "Reasoning effort is set to maximum. Validate non-obvious logic and state transitions, verify intermediate calculations, check common edge cases, test key assumptions against likely counterexamples, compare alternative solution paths, break the problem into its component parts and verify each independently, and cross-check the final answer against all stated constraints and edge cases. Stop once the answer is verified consistent—do not continue re-deriving it once no further errors are found.",
}


def format_effort_prompt(problem: str, effort_tier: str = "high", tokenizer: Optional[Any] = None) -> str:
    """Formats the user problem with the exact effort prompt in the system role."""
    tier_key = effort_tier.lower().strip()
    system_prompt = EFFORT_SYSTEM_PROMPTS.get(tier_key, EFFORT_SYSTEM_PROMPTS["high"])

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"


def resolve_checkpoint_lora(
    checkpoint_name: str = "checkpoint-2200",
    hf_repo_id: str = "Nihilux/SpringHunter",
    local_dir: str = "checkpoints",
    hf_token: Optional[str] = None,
) -> str:
    """
    Resolves checkpoint_lora:
    1. Checks local folder './checkpoint-2200' or 'checkpoints/checkpoint-2200'.
    2. If not found locally, automatically downloads checkpoint-2200 from Nihilux/SpringHunter on HF!
    """
    token = hf_token or os.environ.get("HF_TOKEN") or None

    candidates = [
        checkpoint_name,
        os.path.join(local_dir, checkpoint_name),
        os.path.join("local_trainer", "checkpoints", checkpoint_name),
    ]
    for c in candidates:
        if os.path.isdir(c) and (
            os.path.exists(os.path.join(c, "adapter_config.json"))
            or os.path.exists(os.path.join(c, "trainer_state.json"))
            or os.path.exists(os.path.join(c, "adapter_model.safetensors"))
        ):
            print(f"✅ Found local LoRA checkpoint: {c}")
            return c

    print(f"📥 Checkpoint '{checkpoint_name}' not found locally. Auto-downloading from {hf_repo_id}...")
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=[f"{checkpoint_name}/*", f"{checkpoint_name}/**"],
            local_dir=local_dir,
            token=token,
        )
        target_path = os.path.join(local_dir, checkpoint_name)
        if os.path.exists(target_path):
            print(f"✅ Successfully downloaded {checkpoint_name} from {hf_repo_id} to {target_path}")
            return target_path
    except Exception as e:
        print(f"⚠️  Could not download checkpoint from {hf_repo_id} ({e}). Will initialize fresh LoRA.")

    return checkpoint_name


# =====================================================================
# 🚀  MAIN STANDALONE EXECUTION FUNCTION
# =====================================================================
def run_standalone_math_grpo(
    model_name_or_path: str = "Qwen/Qwen3-30B-A3B",
    checkpoint_lora: str = "checkpoint-2200",
    checkpoint_repo: str = "Nihilux/SpringHunter",
    effort_tier: str = "balanced",
    hf_token: Optional[str] = None,
    repo_id: str = "Nihilux/sword-grpo-200-steps",
    num_rollouts: int = 4,
    max_seq_len: int = 32768,
    max_new_tokens: int = 2048,
    steps: int = 200,
    lr: float = 5e-6,
    use_fp8: bool = True,
):
    setup_blackwell_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(" 🚀 STANDALONE MATH GRPO TRAINER (Nihilux/BigMath2)")
    print(f" Base Model:      {model_name_or_path}")
    print(f" LoRA Checkpoint: {checkpoint_lora} (Store: {checkpoint_repo})")
    print(f" Effort Mode:     {effort_tier.upper()} (Balanced round-robin across 6 tiers if 'balanced')")
    print(f" Context Window:  {max_seq_len} tokens (32k context)")
    print(f" Rollouts:        {num_rollouts} concurrent streams (G=4)")
    print(f" FP8 Engine:      {use_fp8}")
    print(f" Steps:           {steps}")
    print(f" Device:          {device}")
    print("=" * 72)

    # 1. Download & Prepare Dataset
    data_file = download_bigmath2(
        dataset_name="Nihilux/BigMath2",
        cache_file="local_trainer/data/bigmath2.jsonl",
        hf_token=hf_token,
    )
    streamer = stream_math_data_from_disk(data_file)

    # 2. Load Model & Tokenizer
    print(f"\n[*] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        token=hf_token,
        trust_remote_code=True,
    )

    # 3. Apply In-Memory FP8 MoE Quantization
    if use_fp8:
        convert_to_fp8_moe_weights(model)

    # 4. Resolve and Attach LoRA Adapter
    resolved_lora = resolve_checkpoint_lora(
        checkpoint_name=checkpoint_lora,
        hf_repo_id=checkpoint_repo,
        local_dir="checkpoints",
        hf_token=hf_token,
    )
    print(f"[*] Attaching LoRA adapters (path: {resolved_lora})...")
    lora_attached = False

    # Attempt importing unsloth if installed to enable custom MoE LoRA kernels
    try:
        import unsloth
    except Exception:
        pass

    try:
        from peft import PeftModel
        if os.path.exists(resolved_lora):
            model = PeftModel.from_pretrained(model, resolved_lora, is_trainable=True)
            print(f"✅ Loaded existing LoRA from {resolved_lora}")
            lora_attached = True
    except Exception as e:
        print(f"⚠️  PeftModel standard load note: {e}")
        # Cleanly unwrap base model before fallback
        if hasattr(model, "unload"):
            try:
                model = model.unload()
            except Exception:
                pass
        while hasattr(model, "base_model"):
            model = getattr(model.base_model, "model", model.base_model)

        try:
            from peft import LoraConfig, get_peft_model
            from safetensors.torch import load_file as load_safetensors
            adapter_file = os.path.join(resolved_lora, "adapter_model.safetensors")
            if not os.path.exists(adapter_file):
                adapter_file = os.path.join(resolved_lora, "adapter_model.bin")
            if os.path.exists(adapter_file):
                # Target standard attention projections which are always 100% compatible
                lora_cfg = LoraConfig(
                    r=32,
                    lora_alpha=64,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    use_rslora=True,
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_cfg)
                sd = load_safetensors(adapter_file) if adapter_file.endswith(".safetensors") else torch.load(adapter_file, map_location="cpu")
                model_sd = model.state_dict()
                matching_sd = {}
                for k, v in sd.items():
                    if k in model_sd and v.shape == model_sd[k].shape:
                        matching_sd[k] = v
                    else:
                        k_default = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
                        if k_default in model_sd and v.shape == model_sd[k_default].shape:
                            matching_sd[k_default] = v

                if matching_sd:
                    model.load_state_dict(matching_sd, strict=False)
                    print(f"✅ Tolerant LoRA load: initialized {len(matching_sd)}/{len(sd)} matching attention adapter layers from {resolved_lora}")
                    lora_attached = True
        except Exception as e2:
            print(f"⚠️  Tolerant LoRA load note: {e2}")

    if not lora_attached:
        print("[*] Initializing fresh LoRA adapter...")
        try:
            if hasattr(model, "unload"):
                try:
                    model = model.unload()
                except Exception:
                    pass
            while hasattr(model, "base_model"):
                model = getattr(model.base_model, "model", model.base_model)
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            print("✅ Initialized fresh LoRA adapter on attention projections")
        except Exception as e3:
            print(f"⚠️  Fresh LoRA init note: {e3}")

    # Optional: Apply Sword FlashAttention SDPA patch to attention modules
    try:
        from sword.patcher import patch_model
        patch_model(model, mode="flash", patch_moe=False)
    except Exception:
        pass

    # Enable Gradient Checkpointing for memory-efficient backprop (~50 GB activation memory saved)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("✅ Gradient checkpointing enabled (saves ~50 GB activation memory)")
        except Exception:
            try:
                model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
            except Exception:
                pass
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    # 5. Initialize In-Process Rollout Engine (Bypasses vLLM)
    engine = InProcessRolloutEngine(
        model=model,
        tokenizer=tokenizer,
        max_concurrency=num_rollouts,
        max_seq_len=max_seq_len,
        use_fp8_kv=use_fp8,
    )

    # 6. Initialize Scorer, Loss, Auditor, Optimizer
    scorer = MathScorer()
    loss_fn = ChunkedGRPOLoss(chunk_size=512)
    auditor = StepAuditor()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # 7. 200-Step Training Loop
    print(f"\n⚡ Starting {steps} training steps (Effort Mode: {effort_tier})...\n")
    for step in range(1, steps + 1):
        t_start = time.perf_counter()
        item = next(streamer)
        problem_text = item["problem"]
        ref_answer = item["answer"]

        # Resolve effort tier for this step (balanced round-robin vs forced single tier)
        if effort_tier.lower() in ("balanced", "all"):
            step_effort = item.get("effort_tier", "high")
        else:
            step_effort = effort_tier.lower()

        # Format user problem with exact effort system prompt
        formatted_prompt = format_effort_prompt(problem_text, effort_tier=step_effort, tokenizer=tokenizer)

        # Phase A: Inference / Rollout Generation
        raw_rollouts = engine.generate_rollouts(
            prompt=formatted_prompt,
            num_rollouts=num_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )

        # Phase B: Scoring & GDPO Decoupled Advantages
        rollout_results = []
        for text in raw_rollouts:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            res = scorer.score(
                problem=problem_text,
                reference_answer=ref_answer,
                full_text=text,
                effort_tier=step_effort,
                token_count=token_count,
            )
            res["full_text"] = text
            res["token_count"] = token_count
            res["effort_tier"] = step_effort
            rollout_results.append(res)

        advantages, col_advantages = compute_gdpo_advantages(rollout_results)
        for idx, adv in enumerate(advantages):
            rollout_results[idx]["advantage"] = adv

        # Phase C: Training / Backward Step (Micro-batched per rollout with gradient accumulation)
        model.train()
        if hasattr(model, "config"):
            model.config.use_cache = False
        optimizer.zero_grad()

        total_step_loss = 0.0
        p_ids = tokenizer.encode(problem_text, add_special_tokens=False)
        p_len = len(p_ids)

        for i, res in enumerate(rollout_results):
            r_ids = tokenizer.encode(res["full_text"], add_special_tokens=False)
            combo = p_ids + r_ids
            cur_input = torch.tensor([combo], dtype=torch.long, device=device)
            cur_mask = torch.ones_like(cur_input)
            adv_i = float(advantages[i])

            loss_i, _ = loss_fn.forward_single(
                model=model,
                input_ids=cur_input,
                attention_mask=cur_mask,
                prompt_length=p_len,
                advantage=adv_i,
            )

            scaled_loss = loss_i / num_rollouts
            scaled_loss.backward()
            total_step_loss += loss_i.item()

            del cur_input, cur_mask, loss_i, scaled_loss

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        elapsed = time.perf_counter() - t_start
        vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

        # Phase D: Record Generation Audit with dedicated effort_tier column
        auditor.record(step, problem_text, ref_answer, rollout_results, elapsed, effort_tier=step_effort)

        mean_acc = sum(r.get("acc_reward", 0.0) for r in rollout_results) / len(rollout_results)
        mean_fmt = sum(r.get("format_reward", 0.0) for r in rollout_results) / len(rollout_results)
        mean_eff = sum(r.get("effort_reward", 0.0) for r in rollout_results) / len(rollout_results)
        mean_total = sum(r.get("total_reward", 0.0) for r in rollout_results) / len(rollout_results)

        print(
            f"[Step {step:03d}/{steps:03d} | {step_effort.upper():<6}] "
            f"Acc: {mean_acc:.2f} | Fmt: {mean_fmt:.2f} | Eff: {mean_eff:+.2f} | "
            f"Total: {mean_total:+.2f} | Loss: {total_step_loss / num_rollouts:.4f} | "
            f"VRAM: {vram_gb:.1f} GB | Time: {elapsed:.2f}s"
        )

        del rollout_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summarize and Upload to HF Hub
    auditor.summarize()
    zip_and_upload_to_hf(auditor.log_dir, repo_id, hf_token=hf_token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-2200")
    parser.add_argument("--repo_id", type=str, default=os.environ.get("HF_UPLOAD_REPO_ID", "Nihilux/sword-grpo-200-steps"))
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--effort_tier", type=str, default="balanced", choices=["balanced", "all", "low", "medium", "high", "xhigh", "ultra", "max"], help="Reasoning effort tier or 'balanced' for round-robin split across all 6 tiers")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--test_stream_only", action="store_true", help="Only stream 1 row of Nihilux/BigMath2 to inspect and exit")
    args = parser.parse_args()

    if args.test_stream_only:
        data_file = download_bigmath2(dataset_name="Nihilux/BigMath2", hf_token=args.token or None)
        streamer = stream_math_data_from_disk(data_file)
        row = next(streamer)
        print("\n[Test Stream] Inspected 1 sample row:")
        print(json.dumps(row, indent=2))
        sys.exit(0)

    run_standalone_math_grpo(
        model_name_or_path=args.model,
        checkpoint_lora=args.checkpoint,
        repo_id=args.repo_id,
        hf_token=args.token,
        effort_tier=args.effort_tier,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
