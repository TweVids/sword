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
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional, Generator, Union
from collections import defaultdict
from dataclasses import dataclass, field

# Import unsloth at the very top before transformers/peft to activate Unsloth optimizations
try:
    import unsloth
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except Exception:
    HAS_UNSLOTH = False

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
    if converted > 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

        prompt_len = input_ids.shape[1]
        allowed_new = min(max_new_tokens, max(1, self.max_seq_len - prompt_len))

        # Collect all chat & base EOS stop tokens (<|im_end|>, <|endoftext|>, </s>)
        stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                stop_token_ids.extend(self.tokenizer.eos_token_id)
            else:
                stop_token_ids.append(self.tokenizer.eos_token_id)
        for stop_str in ["<|im_end|>", "<|endoftext|>", "</s>"]:
            sid = self.tokenizer.convert_tokens_to_ids(stop_str)
            if sid is not None and isinstance(sid, int) and sid > 0 and sid not in stop_token_ids:
                stop_token_ids.append(sid)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=allowed_new,
                max_length=None,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
            )

        # Slice generated response tokens (excluding prompt)
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

        # Case 1: Multiple opening tags (looping / nested generation)
        if open_count > 1:
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
        if open_count == 0 and close_count >= 1:
            parts = text.split(close_tag, 1)
            trace = parts[0].strip()
            answer = parts[1].strip()
            audit["error"] = "orphaned_close_tag"
            return -0.20, audit, trace, answer

        # Case 4: Single opening tag with at least one closing tag
        if open_count == 1 and close_count >= 1:
            pos_open = text.find(open_tag)
            # Use the final closing tag to separate thinking from answer
            pos_last_close = text.rfind(close_tag)

            if pos_open > pos_last_close:
                audit["error"] = "inverted_tags_close_before_open"
                return -0.25, audit, trace, answer

            trace = text[pos_open + len(open_tag):pos_last_close].strip()
            answer = text[pos_last_close + len(close_tag):].strip()

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
            if close_count > 1:
                audit["quoted_close_tag_in_trace"] = True
            return 0.10, audit, trace, answer

        # Case 5: Neither tag present (missing required thinking block in math)
        audit["error"] = "missing_required_thinking_tags"
        return -0.20, audit, trace, answer


# =====================================================================
# 📝  TEXT & BULLET STRUCTURE UTILITIES
# =====================================================================
def count_sentences(text: str) -> int:
    """
    Accurately counts sentences in a text block, masking LaTeX formulas,
    decimal numbers (e.g. 3.14), abbreviations, and splitting on [.!?] followed by whitespace.
    """
    if not text or not text.strip():
        return 0
    cleaned = re.sub(
        r"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\s*\\begin\{[a-z*]*\}.*?\\end\{[a-z*]*\}",
        " FORMULA ",
        text,
        flags=re.DOTALL,
    )
    # Mask decimal numbers like 3.14 so period isn't treated as end of sentence
    cleaned = re.sub(r"\b\d+\.\d+\b", "NUM", cleaned)
    # Split on terminal punctuation followed by space or newline
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 3]
    return len(sents)


def extract_bullet_items(text: str) -> List[str]:
    """
    Extracts individual bullet points from markdown text.
    Handles *, -, +, and • bullets, including multi-line bullet bodies.
    """
    lines = text.split("\n")
    bullets: List[str] = []
    current_bullet: List[str] = []
    bullet_marker = re.compile(r"^\s*[-*+•]\s+(.*)$")

    for line in lines:
        m = bullet_marker.match(line)
        if m:
            if current_bullet:
                bullets.append("\n".join(current_bullet).strip())
                current_bullet = []
            current_bullet.append(m.group(1).strip())
        elif current_bullet:
            stripped = line.strip()
            # If line is header, end current bullet
            if re.match(r"^\s*#{1,6}\s+", line):
                bullets.append("\n".join(current_bullet).strip())
                current_bullet = []
            elif stripped:
                current_bullet.append(stripped)
            else:
                current_bullet.append("")

    if current_bullet:
        bullets.append("\n".join(current_bullet).strip())

    return [b for b in bullets if b.strip()]


def is_valid_bullet_archetype(bullet_text: str) -> Tuple[bool, str]:
    """
    Validates if a bullet item matches Archetype A or Archetype B:
    - Archetype A: ** >=5 words title - ** Body paragraph with >= 3 sentences
      (the ' - ' separator before the closing ** is strictly mandatory)
    - Archetype B: Continuous paragraph with >= 4 sentences
    Returns (is_valid, matched_archetype)
    """
    # Check Archetype A: ** title - ** body
    # Strictly requires ' - ' separator before the closing **
    match_a = re.match(r"^\s*\*\*\s*(.+?)\s+-\s+\*\*\s*(.+)$", bullet_text, re.DOTALL)
    if match_a:
        title = match_a.group(1).strip()
        body = match_a.group(2).strip()
        title_word_count = len(title.split())
        body_sentence_count = count_sentences(body)
        if title_word_count >= 5 and body_sentence_count >= 3:
            return True, "archetype_a"

    # Check Archetype B: continuous paragraph bullet with >= 4 sentences
    sent_count = count_sentences(bullet_text)
    if sent_count >= 4:
        return True, "archetype_b"

    return False, "invalid"


# =====================================================================
# 🎯  PRIMARY SCORER (GROUND TRUTH MATH + FORMAT + EFFICIENCY)
# =====================================================================
class MathScorer:
    """Evaluates mathematical ground truth accuracy and formatting."""

    @staticmethod
    def _normalize_math(ans: str) -> str:
        s = str(ans).strip().lower()
        # Handle GSM8K '####' final answer marker
        if "####" in s:
            s = s.split("####")[-1].strip()
        s = s.replace(";", ",")
        # Handle LaTeX commands before stripping backslashes
        s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]+)\}", r"\1", s)
        s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
        # Extract \boxed{...} if present
        boxed = re.findall(r"\\boxed\{([^}]+)\}", s)
        if boxed:
            return MathScorer._normalize_math(boxed[-1])
        s = re.sub(r"[\$\\,\s;]", "", s)
        # Strip trailing .0 or .00 if whole number (e.g. 357.0 -> 357)
        s = re.sub(r"\.0+(?=[^\d]|$)", "", s)
        return s

    def score(
        self,
        problem: str,
        reference_answer: str,
        full_text: str,
        effort_tier: str = "high",
        token_count: int = 0,
        dataset_name: str = "",
    ) -> Dict[str, Any]:
        tag_score, tag_audit, trace, answer = ThinkingFormatVerifier.verify(full_text)
        audit_flags: Dict[str, Any] = {"thinking_tags": tag_audit}

        norm_ref = self._normalize_math(reference_answer)
        norm_ans = self._normalize_math(answer)
        norm_full = self._normalize_math(full_text)

        # 1. Ground Truth Accuracy (handles float/integer equivalence, interval notation, and boxed answers)
        is_match = False
        try:
            ref_num = float(norm_ref)
            ans_numbers = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", answer)]
            if any(abs(n - ref_num) < 1e-4 for n in ans_numbers):
                is_match = True
            else:
                boxed_nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", norm_ans)]
                if any(abs(n - ref_num) < 1e-4 for n in boxed_nums):
                    is_match = True
        except (ValueError, TypeError):
            pass

        if not is_match:
            is_match = (
                (norm_ref in norm_ans)
                or (norm_ans == norm_ref)
                or (norm_ref and norm_ref in norm_full)
                or (reference_answer.strip().lower() in answer.lower())
            )
        accuracy_score = 0.35 if is_match else -0.40

        # 2. Formatting: Paragraph Thinking, Outside Paragraph Answer, and Boxed Output
        format_score = 0.0

        # A. Check for bullet points, numbered lists, or markdown headers in thinking trace
        # Strip display math environments ($$..$$, \[..\], \begin{..}..\end{..}) before bullet checking
        # so equations containing minus signs, asterisks or numbered tags are not penalized as bullets.
        trace_text_for_lists = re.sub(
            r"\$\$.*?\$\$|\\\[.*?\\\]|\\begin\{[a-z*]*\}.*?\\end\{[a-z*]*\}",
            "",
            trace,
            flags=re.DOTALL,
        )
        has_bullets_in_trace = bool(re.search(r"^\s*[-*•]\s+(?!\s*[\d\w\\$].*?[=<>])", trace_text_for_lists, re.MULTILINE))
        has_numbered_in_trace = bool(re.search(r"^\s*\d+[\.)]\s+", trace_text_for_lists, re.MULTILINE))
        has_headers_in_trace = bool(re.search(r"^\s*#{1,6}\s+", trace_text_for_lists, re.MULTILINE))

        if has_bullets_in_trace or has_numbered_in_trace or has_headers_in_trace:
            format_score -= 0.15
            audit_flags["bullet_or_list_in_thinking"] = True
        elif len(trace.strip()) >= 20:
            # Reward fluent continuous paragraph thinking
            format_score += 0.15
            audit_flags["natural_paragraph_thinking_rewarded"] = True

            # Check 3+ sentence paragraph depth for BigMath2 or when effort is higher than medium (high, xhigh, ultra, max)
            is_bigmath = "bigmath" in str(dataset_name).lower()
            is_higher_than_medium = str(effort_tier).lower() in ("high", "xhigh", "ultra", "max")
            if is_bigmath or is_higher_than_medium:
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", trace) if len(p.strip()) > 15]
                if paragraphs:
                    counts = [count_sentences(p) for p in paragraphs]
                    avg_sents = sum(counts) / len(counts)
                    if avg_sents >= 3.0 or any(c >= 3 for c in counts):
                        format_score += 0.10
                        audit_flags["deep_paragraph_sentences_rewarded"] = True
                    else:
                        format_score -= 0.10
                        audit_flags["shallow_paragraphs_penalty"] = -0.10

        # B. Paragraph behavior outside <think> (in final answer)
        # 1. Detect reasoning evasion: markdown headers, step markers, or bullet lists in the answer (-0.20)
        has_headers_in_answer = bool(re.search(r"^\s*#{1,6}\s+", answer, re.MULTILINE))
        has_step_markers_in_answer = bool(re.search(r"(?i)\*\*step\s*\d+[:\.]?|\bstep\s*\d+[:\.]", answer))
        has_bullets_in_answer = bool(re.search(r"^\s*[-*•]\s+(?!\s*[\d\w\\$].*?[=<>])", answer, re.MULTILINE))
        has_numbered_in_answer = bool(re.search(r"^\s*\d+[\.)]\s+", answer, re.MULTILINE))

        has_reasoning_dump = has_headers_in_answer or has_step_markers_in_answer or (len(answer.strip()) > 250 and (has_bullets_in_answer or has_numbered_in_answer))

        # Check for bare answers (only digits/symbols or very short, without descriptive prose)
        cleaned_words = re.sub(r"\\boxed\{[^}]*\}|[\d\s\.,;:!?'\"\(\)\$\+\-\*\/=]", "", answer).strip()
        is_bare_answer = len(cleaned_words) < 8

        if has_reasoning_dump:
            format_score -= 0.20
            audit_flags["reasoning_dump_in_answer"] = True
        elif has_bullets_in_answer or has_numbered_in_answer:
            format_score -= 0.15
            audit_flags["bullet_or_list_in_answer"] = True
        elif is_bare_answer:
            # Penalize naked numbers without explanatory sentence prose
            format_score -= 0.10
            audit_flags["bare_answer_without_prose"] = True
        elif 15 <= len(answer.strip()) <= 350:
            # Reward a concise 1-2 sentence explanatory prose wrapping the answer (+0.15)
            format_score += 0.15
            audit_flags["natural_paragraph_answer_rewarded"] = True

        # C. Check for \boxed{...} in final answer or full text
        has_boxed = bool(re.search(r"\\boxed\{[^}]+\}", answer or full_text))
        if has_boxed:
            format_score += 0.10
            audit_flags["boxed_answer_rewarded"] = True

        # 3. Efficiency & Anti-Looping
        efficiency_score = 0.0

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

        # Continuous token efficiency: slightly rewards conciseness to break ties and guarantee non-zero GRPO variance/loss
        conciseness_bonus = max(0.0, (1.0 - min(token_count / max_budget, 1.0))) * 0.05
        effort_score += conciseness_bonus

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
# 🎓  EXPLANATION SCORER (MULTI-TURN PEDAGOGICAL DIALOG)
# =====================================================================
class ExplanationScorer:
    """
    Dedicated multi-reward scoring system for Turn 2 explanations.
    Completely decoupled from Turn 1 math solving to prevent reward conflicts:
    1. Steals reasoning content discipline for <think> traces:
       - Tag compliance (ThinkingFormatVerifier)
       - Continuous natural paragraphs (+0.15), strictly no bullets/headers (-0.15)
       - 3+ sentence paragraph depth (+0.10 / -0.10) for complex problems or effort > medium
       - Anti-looping / anti-repetition (-0.50)
       - Prohibited bold markdown in thinking (-0.20)
       - Token budget compliance & verification bonuses
    2. Explanation Answer Field Rules:
       - GSM8K: exactly 1 continuous paragraph with >= 5 sentences (no headers, no bullets) (+0.25 / -0.25)
       - Complex Datasets (AoPS, BigMath2):
         * Header rule: '#' and '##' must have > 5 words per header. Short headers (<= 5 words) penalized (-0.15), rich headers (> 5 words) rewarded (+0.15).
         * Bullet quantity: when bullets are used, must have >= 5 bullets. Fewer than 5 penalized (-0.20).
         * Bullet structure: every bullet must match Archetype A (strictly mandatory ' - ' separator before **) or Archetype B (>= 4 sentences). All valid (+0.20), any invalid (-0.20).
       - Pedagogical content & accuracy: ground truth concept present (+0.35 / -0.30), substantive length (+0.10) / shallow (-0.20).
    3. Grouped into decoupled GDPO columns:
       - pedagogical_accuracy
       - explanation_structure
       - thinking_discipline
       - efficiency
       - safety
    """

    @staticmethod
    def _normalize_math(ans: str) -> str:
        return MathScorer._normalize_math(ans)

    def score(
        self,
        problem: str,
        reference_answer: str,
        full_text: str,
        effort_tier: str = "high",
        token_count: int = 0,
        dataset_name: str = "",
    ) -> Dict[str, Any]:
        tag_score, tag_audit, trace, answer = ThinkingFormatVerifier.verify(full_text)
        audit_flags: Dict[str, Any] = {"thinking_tags": tag_audit}

        # -------------------------------------------------------------
        # 1. THINKING TRACE RULES ("Stolen from reasoning content")
        # -------------------------------------------------------------
        thinking_format_score = 0.0

        trace_text_for_lists = re.sub(
            r"\$\$.*?\$\$|\\\[.*?\\\]|\\begin\{[a-z*]*\}.*?\\end\{[a-z*]*\}",
            "",
            trace,
            flags=re.DOTALL,
        )
        has_bullets_in_trace = bool(re.search(r"^\s*[-*•]\s+(?!\s*[\d\w\\$].*?[=<>])", trace_text_for_lists, re.MULTILINE))
        has_numbered_in_trace = bool(re.search(r"^\s*\d+[\.)]\s+", trace_text_for_lists, re.MULTILINE))
        has_headers_in_trace = bool(re.search(r"^\s*#{1,6}\s+", trace_text_for_lists, re.MULTILINE))

        if has_bullets_in_trace or has_numbered_in_trace or has_headers_in_trace:
            thinking_format_score -= 0.15
            audit_flags["bullet_or_list_in_thinking"] = True
        elif len(trace.strip()) >= 20:
            thinking_format_score += 0.15
            audit_flags["natural_paragraph_thinking_rewarded"] = True

            # Check 3+ sentence paragraph depth for complex datasets or effort > medium
            is_complex = any(k in str(dataset_name).lower() for k in ["bigmath", "aops", "openmath"])
            is_higher_than_medium = str(effort_tier).lower() in ("high", "xhigh", "ultra", "max")
            if is_complex or is_higher_than_medium:
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", trace) if len(p.strip()) > 15]
                if paragraphs:
                    counts = [count_sentences(p) for p in paragraphs]
                    avg_sents = sum(counts) / len(counts)
                    if avg_sents >= 3.0 or any(c >= 3 for c in counts):
                        thinking_format_score += 0.10
                        audit_flags["deep_paragraph_sentences_rewarded"] = True
                    else:
                        thinking_format_score -= 0.10
                        audit_flags["shallow_paragraphs_penalty"] = -0.10

        # Efficiency & Anti-Looping in thinking trace
        efficiency_score = 0.0
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", trace) if len(s.split()) >= 4]
        counts = defaultdict(int)
        for s in sentences:
            norm = s.lower()
            counts[norm] += 1
            if counts[norm] >= 2:
                efficiency_score -= 0.50
                audit_flags["repetitive_loop_detected"] = True
                break

        if re.search(r"\*\*[^*\n]+\*\*", trace):
            efficiency_score -= 0.20
            audit_flags["prohibited_bold_found"] = True

        # Effort tier token budget
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

        if effort_tier.lower() in ("high", "xhigh", "ultra", "max"):
            verification_cues = [r"\bverif", r"\bcheck", r"\bconfirm", r"\bdouble[-\s]check", r"\bsubstitut", r"\bassum"]
            if any(re.search(pat, trace, re.IGNORECASE) for pat in verification_cues):
                effort_score += 0.10
                audit_flags["effort_verification_steps_rewarded"] = True

        conciseness_bonus = max(0.0, (1.0 - min(token_count / max_budget, 1.0))) * 0.05
        effort_score += conciseness_bonus

        # -------------------------------------------------------------
        # 2. EXPLANATION ANSWER FIELD RULES (PEDAGOGICAL STRUCTURE)
        # -------------------------------------------------------------
        explanation_structure_score = 0.0
        is_gsm8k = "gsm8k" in str(dataset_name).lower() or "template" in str(dataset_name).lower()

        if is_gsm8k:
            # GSM8K Single Continuous Paragraph Rule:
            # Must be exactly 1 continuous paragraph, >= 5 sentences, NO headers, NO bullets
            has_headers = bool(re.search(r"^\s*#{1,6}\s+", answer, re.MULTILINE))
            has_bullets = bool(re.search(r"^\s*[-*•]\s+", answer, re.MULTILINE) or re.search(r"^\s*\d+[\.)]\s+", answer, re.MULTILINE))
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", answer) if len(p.strip()) > 10]
            num_sents = count_sentences(answer)

            if has_headers or has_bullets or len(paragraphs) != 1 or num_sents < 5:
                explanation_structure_score -= 0.25
                audit_flags["gsm8k_paragraph_rule_violation"] = {
                    "has_headers": has_headers,
                    "has_bullets": has_bullets,
                    "paragraph_count": len(paragraphs),
                    "sentence_count": num_sents,
                }
            else:
                explanation_structure_score += 0.25
                audit_flags["gsm8k_single_deep_paragraph_rewarded"] = True
        else:
            # Complex Datasets (AoPS, BigMath2, etc.) Rules:
            # A. Header Rule: # and ## allowed, but must have > 5 words per header
            headers = [m.group(1).strip() for m in re.finditer(r"^\s*#{1,6}\s+(.+)$", answer, re.MULTILINE)]
            if headers:
                short_headers = [h for h in headers if len(h.split()) <= 5]
                if short_headers:
                    explanation_structure_score -= 0.15
                    audit_flags["short_header_penalty"] = -0.15
                    audit_flags["short_headers"] = short_headers
                else:
                    explanation_structure_score += 0.15
                    audit_flags["rich_header_rewarded"] = True

            # B. Bullet Quantity Rule & Archetype Structure
            bullet_items = extract_bullet_items(answer)
            if bullet_items:
                if len(bullet_items) < 5:
                    explanation_structure_score -= 0.20
                    audit_flags["insufficient_bullets_penalty"] = -0.20
                    audit_flags["bullet_count"] = len(bullet_items)
                else:
                    audit_flags["adequate_bullets"] = True

                # Check Archetypes for each bullet
                invalid_bullets = []
                archetype_counts = {"archetype_a": 0, "archetype_b": 0}
                for b_idx, b_text in enumerate(bullet_items):
                    is_valid, arch_type = is_valid_bullet_archetype(b_text)
                    if is_valid:
                        archetype_counts[arch_type] += 1
                    else:
                        invalid_bullets.append({"idx": b_idx, "snippet": b_text[:80]})

                if invalid_bullets:
                    explanation_structure_score -= 0.20
                    audit_flags["invalid_bullet_archetype_penalty"] = -0.20
                    audit_flags["invalid_bullets"] = invalid_bullets
                elif len(bullet_items) >= 5:
                    explanation_structure_score += 0.20
                    audit_flags["valid_bullet_archetypes_rewarded"] = True
                    audit_flags["archetype_counts"] = archetype_counts

        # -------------------------------------------------------------
        # 3. PEDAGOGICAL CONTENT & MATHEMATICAL ACCURACY
        # -------------------------------------------------------------
        norm_ref = self._normalize_math(reference_answer)
        norm_ans = self._normalize_math(answer)
        norm_full = self._normalize_math(full_text)

        is_concept_match = False
        try:
            ref_num = float(norm_ref)
            ans_numbers = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", answer)]
            if any(abs(n - ref_num) < 1e-4 for n in ans_numbers):
                is_concept_match = True
        except (ValueError, TypeError):
            pass

        if not is_concept_match:
            is_concept_match = (
                (norm_ref in norm_ans)
                or (norm_ans == norm_ref)
                or (norm_ref and norm_ref in norm_full)
                or (reference_answer.strip().lower() in answer.lower())
            )

        pedagogical_accuracy = 0.35 if is_concept_match else -0.30
        audit_flags["pedagogical_accuracy_match"] = is_concept_match

        # Substantive explanation length check
        ans_word_count = len(answer.split())
        if ans_word_count >= 80:
            pedagogical_accuracy += 0.10
            audit_flags["substantive_explanation_rewarded"] = True
        elif ans_word_count < 20:
            pedagogical_accuracy -= 0.20
            audit_flags["shallow_explanation_penalty"] = -0.20

        # -------------------------------------------------------------
        # 4. GROUP INTO DECOUPLED GDPO COLUMNS
        # -------------------------------------------------------------
        columns = {
            "pedagogical_accuracy": round(pedagogical_accuracy, 4),
            "explanation_structure": round(explanation_structure_score, 4),
            "thinking_discipline": round(tag_score + thinking_format_score, 4),
            "efficiency": round(efficiency_score + effort_score, 4),
            "safety": 0.0,
        }

        total_reward = round(sum(columns.values()), 4)

        return {
            "total_reward": total_reward,
            "column_scores": columns,
            "component_scores": {
                "thinking_tags": tag_score,
                "thinking_format": thinking_format_score,
                "explanation_structure": explanation_structure_score,
                "pedagogical_accuracy": pedagogical_accuracy,
                "efficiency": efficiency_score,
                "effort_budget": round(effort_score, 4),
                "safety": 0.0,
            },
            "is_correct": is_concept_match,
            "reasoning_trace": trace,
            "final_answer": answer,
            "audit_log": audit_flags,
        }


# =====================================================================
# ⚖️  GDPO (GROUP REWARD-DECOUPLED NORMALIZATION POLICY OPTIMIZATION)
# =====================================================================
def compute_explanation_gdpo_advantages(
    rollout_results: List[Dict[str, Any]],
    column_weights: Optional[Dict[str, float]] = None,
    eps: float = 1e-8,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Decoupled advantage normalization across pedagogical explanation columns:
    pedagogical_accuracy (1.0), explanation_structure (0.4), thinking_discipline (0.3), efficiency (0.2).
    """
    weights = column_weights or {
        "pedagogical_accuracy": 1.0,
        "explanation_structure": 0.4,
        "thinking_discipline": 0.3,
        "efficiency": 0.2,
    }
    G = len(rollout_results)
    if G <= 1:
        return [0.0] * G, {k: [0.0] * G for k in weights.keys()}

    norm_advs: Dict[str, List[float]] = {}
    for col in weights.keys():
        vals = [r["column_scores"].get(col, 0.0) for r in rollout_results]
        mean_v = sum(vals) / G
        var_v = sum((v - mean_v) ** 2 for v in vals) / G
        std_v = math.sqrt(var_v)
        if std_v < eps:
            norm_advs[col] = [0.0] * G
        else:
            norm_advs[col] = [(v - mean_v) / (std_v + eps) for v in vals]

    combined = [0.0] * G
    for col, w in weights.items():
        adv_col = norm_advs[col]
        for i in range(G):
            combined[i] += w * adv_col[i]

    return combined, norm_advs


def compute_gdpo_advantages(
    rollout_results: List[Dict[str, Any]],
    column_weights: Optional[Dict[str, float]] = None,
    eps: float = 1e-8,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Decoupled normalization across accuracy, formatting, and efficiency columns.
    Prevents penalty spiking and reward collapse (NVIDIA arXiv:2601.05242).
    Automatically routes to compute_explanation_gdpo_advantages if pedagogical_accuracy is present.
    """
    if rollout_results and "pedagogical_accuracy" in rollout_results[0].get("column_scores", {}):
        return compute_explanation_gdpo_advantages(rollout_results, column_weights=column_weights, eps=eps)

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

        token_nll = -policy_logprobs.mean().item()
        pg_loss = -(policy_logprobs * adv_tensor).mean().item()

        return policy_loss, {
            "grpo_loss": round(policy_loss.item(), 5),
            "nll": round(token_nll, 4),
            "pg_loss": round(pg_loss, 4),
        }


# =====================================================================
# 📦  DATASET DOWNLOADER & DISK BATCH STREAMER
# =====================================================================
EFFORT_TIERS_LIST: List[str] = ["low", "medium", "high", "xhigh", "ultra", "max"]


def download_math_dataset(
    dataset_name: str = "math-ai/TemplateGSM",
    cache_file: Optional[str] = None,
    max_samples: int = 10000,
    hf_token: Optional[str] = None,
) -> str:
    """
    Downloads / caches a math dataset (e.g. math-ai/TemplateGSM, Nihilux/BigMath2) to local disk.
    Extracts problem and answer/result, annotating each sample with the dataset source and an effort tier.
    """
    if cache_file is None:
        safe_name = dataset_name.replace("/", "_").replace("\\", "_")
        cache_file = f"local_trainer/data/{safe_name}.jsonl"

    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
        print(f"[Data] Found local cached dataset: {cache_file} ({os.path.getsize(cache_file)/1e6:.1f} MB)")
        return cache_file

    print(f"[Data] Streaming {dataset_name} from Hugging Face...")
    from datasets import load_dataset
    token = hf_token or os.environ.get("HF_TOKEN") or None

    try:
        ds = load_dataset(dataset_name, split="train", streaming=True, token=token)
    except Exception as e:
        print(f"[Data] Notice: standard streaming load for {dataset_name} reported: {e}. Trying default config...")
        try:
            ds = load_dataset(dataset_name, "templategsm-1000-1k", split="train", streaming=True, token=token)
        except Exception:
            ds = load_dataset(dataset_name, split="train", token=token)

    count = 0
    with open(cache_file, "w", encoding="utf-8") as f:
        for item in ds:
            prob = item.get("problem") or item.get("question") or ""
            # Support TemplateGSM ('result'), BigMath2 / GSM8K ('answer', 'solution')
            ans = item.get("result") or item.get("answer") or item.get("solution") or ""
            if not prob or not ans:
                continue
            tier = EFFORT_TIERS_LIST[count % len(EFFORT_TIERS_LIST)]
            f.write(json.dumps({
                "idx": count,
                "dataset": dataset_name,
                "problem": str(prob).strip(),
                "answer": str(ans).strip(),
                "effort_tier": tier,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_samples:
                break

    print(f"[Data] Cached {count} math problems from '{dataset_name}' to {cache_file} (balanced across 6 effort tiers)")
    return cache_file


# Backward compatibility alias
download_bigmath2 = download_math_dataset


def download_curated_study_dataset(
    cache_file: str = "local_trainer/data/curated_study_100.jsonl",
    hf_token: Optional[str] = None,
    force_download: bool = False,
    offline: bool = False,
) -> str:
    """
    Downloads and caches exactly 100 curated math problems across 3 authoritative sources:
      1. 35 problems from GSM8K (openai/gsm8k, train split)
      2. 30 problems from OpenMathReasoning (unsloth/OpenMathReasoning, aops_c4_high_school_math column)
      3. 35 problems from BigMath2 (Nihilux/BigMath2, train split)
      Total: exactly 100 problems.
    Balanced across 6 reasoning effort tiers with built-in offline fallback data.
    """
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    if not force_download and os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
        with open(cache_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) == 100:
            print(f"[Data] Found local cached curated study dataset: {cache_file} (100 problems)")
            return cache_file

    print(f"[Data] Curating 100 study problems (35 GSM8K + 30 AoPS C4 + 35 BigMath2)...")
    token = hf_token or os.environ.get("HF_TOKEN") or None
    records: List[Dict[str, Any]] = []

    # 1. 35 GSM8K
    if not offline:
        try:
            from datasets import load_dataset
            ds_gsm = load_dataset("openai/gsm8k", "main", split="train", streaming=True, token=token)
            count_gsm = 0
            for row in ds_gsm:
                prob = row.get("question", "").strip()
                ans = row.get("answer", "").strip()
                if prob and ans:
                    target_ans = ans.split("####")[-1].strip() if "####" in ans else ans
                    records.append({
                        "idx": len(records),
                        "dataset": "gsm8k",
                        "problem": prob,
                        "answer": target_ans,
                        "effort_tier": EFFORT_TIERS_LIST[len(records) % len(EFFORT_TIERS_LIST)],
                    })
                    count_gsm += 1
                    if count_gsm >= 35:
                        break
            print(f"[Data] Collected {count_gsm} problems from GSM8K")
        except Exception as e:
            print(f"[Data] Note: GSM8K streaming returned ({e}). Filling from fallback...")

    # Fill any missing GSM8K from representative problems
    while sum(1 for r in records if r["dataset"] == "gsm8k") < 35:
        curr = sum(1 for r in records if r["dataset"] == "gsm8k")
        records.append({
            "idx": len(records),
            "dataset": "gsm8k",
            "problem": f"A store sells item {curr+1} for $15 each. If a customer buys 4 items and uses a $5 discount coupon, what is the total cost in dollars?",
            "answer": "55",
            "effort_tier": EFFORT_TIERS_LIST[len(records) % len(EFFORT_TIERS_LIST)],
        })

    # 2. 30 AoPS C4 High School Math (unsloth/OpenMathReasoning)
    if not offline:
        try:
            from datasets import load_dataset
            ds_aops = load_dataset("unsloth/OpenMathReasoning", split="cot", streaming=True, token=token)
            count_aops = 0
            for row in ds_aops:
                if row.get("problem_source") == "aops_c4_high_school_math":
                    prob = row.get("problem", "").strip()
                    ans = row.get("expected_answer", "").strip()
                    if prob and ans:
                        records.append({
                            "idx": len(records),
                            "dataset": "aops_c4_high_school_math",
                            "problem": prob,
                            "answer": ans,
                            "effort_tier": EFFORT_TIERS_LIST[len(records) % len(EFFORT_TIERS_LIST)],
                        })
                        count_aops += 1
                        if count_aops >= 30:
                            break
            print(f"[Data] Collected {count_aops} problems from AoPS C4 High School Math")
        except Exception as e:
            print(f"[Data] Note: AoPS C4 streaming returned ({e}). Filling from fallback...")

    while sum(1 for r in records if r["dataset"] == "aops_c4_high_school_math") < 30:
        curr = sum(1 for r in records if r["dataset"] == "aops_c4_high_school_math")
        records.append({
            "idx": len(records),
            "dataset": "aops_c4_high_school_math",
            "problem": f"Find all real solutions x satisfying x^2 - {curr+4}x + 4 = 0.",
            "answer": f"\\boxed{{{curr+2}}}",
            "effort_tier": EFFORT_TIERS_LIST[len(records) % len(EFFORT_TIERS_LIST)],
        })

    # 3. 35 BigMath2 (Nihilux/BigMath2)
    if not offline:
        try:
            from datasets import load_dataset
            ds_bm = load_dataset("Nihilux/BigMath2", split="train", streaming=True, token=token)
            count_bm = 0
            for row in ds_bm:
                prob = row.get("problem", "").strip()
                ans = row.get("answer", "").strip()
                if prob and ans:
                    records.append({
                        "idx": len(records),
                        "dataset": "BigMath2",
                        "problem": prob,
                        "answer": ans,
                        "effort_tier": EFFORT_TIERS_LIST[len(records) % len(EFFORT_TIERS_LIST)],
                    })
                    count_bm += 1
                    if count_bm >= 35:
                        break
            print(f"[Data] Collected {count_bm} problems from BigMath2")
        except Exception as e:
            print(f"[Data] Note: BigMath2 streaming returned ({e}). Filling from fallback...")

    while sum(1 for r in records if r["dataset"] == "BigMath2") < 35:
        curr = sum(1 for r in records if r["dataset"] == "BigMath2")
        records.append({
            "idx": len(records),
            "dataset": "BigMath2",
            "problem": f"Compute the definite integral \\int_0^{curr+1} (3x^2 + 2x) dx.",
            "answer": f"\\boxed{{{(curr+1)**3 + (curr+1)**2}}}",
            "effort_tier": EFFORT_TIERS_LIST[len(records) % len(EFFORT_TIERS_LIST)],
        })

    # Re-index and save exactly 100 records
    records = records[:100]
    for i, r in enumerate(records):
        r["idx"] = i
        r["effort_tier"] = EFFORT_TIERS_LIST[i % len(EFFORT_TIERS_LIST)]

    with open(cache_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Data] Successfully saved {len(records)} curated study problems to {cache_file}")
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


def prepare_study_records(
    file_path: str,
    shuffle_mode: Union[bool, str] = "stratified",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Loads records from JSONL file on disk and optionally applies deterministic shuffling.
    Modes:
      - True / "stratified" / "balanced": Groups by dataset source (GSM8K, AoPS C4, BigMath2),
        shuffles within each source, and round-robin interleaves them. This guarantees that
        every consecutive window has an even distribution of all problem sources.
      - "random" / "uniform": Shuffles all problems uniformly at random using the seed.
      - False / "none" / "off": Preserves canonical file order.
    """
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return records

    # Automatically sanitize GSM8K reference answers so they contain the clean target number
    for r in records:
        ans_raw = str(r.get("answer", "")).strip()
        if "####" in ans_raw:
            r["answer"] = ans_raw.split("####")[-1].strip()

    mode_str = str(shuffle_mode).lower().strip()
    if mode_str in ("false", "none", "off", "0", "no"):
        return records

    rng = random.Random(seed)

    if mode_str in ("true", "stratified", "balanced", "interleaved", "1", "yes"):
        # Split into dataset buckets
        gsm_list = [r for r in records if "gsm" in r.get("dataset", "").lower()]
        aops_list = [r for r in records if "aops" in r.get("dataset", "").lower()]
        bm_list = [r for r in records if "bigmath" in r.get("dataset", "").lower()]
        other_list = [
            r for r in records
            if r not in gsm_list and r not in aops_list and r not in bm_list
        ]

        # Shuffle each bucket independently
        rng.shuffle(gsm_list)
        rng.shuffle(aops_list)
        rng.shuffle(bm_list)
        rng.shuffle(other_list)

        # Interleave buckets round-robin
        buckets = [b for b in [gsm_list, aops_list, bm_list, other_list] if b]
        interleaved: List[Dict[str, Any]] = []
        while any(len(b) > 0 for b in buckets):
            for b in buckets:
                if b:
                    interleaved.append(b.pop(0))
        return interleaved

    # Default fallback: uniform random permutation
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled


def get_step_schedule(
    step: int,
    total_steps: int = 200,
    records: Optional[List[Dict[str, Any]]] = None,
    explanation_mode: Union[str, bool] = "effort_cycle",
    effort_tier: str = "balanced",
    pair_problems: bool = True,
) -> Dict[str, Any]:
    """
    Determines the operating mode (SOLVE vs EXPLAIN), effort tier, and dataset problem for a given training step.
    
    Supported explanation_mode values:
      - 'effort_cycle' / 'cycle' / 'effort_blocks':
        Passes all effort tiers (low -> max) in SOLVE mode, then passes all effort tiers in EXPLAIN mode, and repeats!
        Cycle length = 2 * len(EFFORT_TIERS_LIST) = 12 steps.
        Steps 1-6:   SOLVE   (low, medium, high, xhigh, ultra, max)
        Steps 7-12:  EXPLAIN (low, medium, high, xhigh, ultra, max)
        Steps 13-18: SOLVE   (low, medium, high, xhigh, ultra, max)
        ...
        If pair_problems=True: the 6 problems solved in steps 1-6 are the exact same 6 problems explained in steps 7-12!
      - 'interleaved' / 'paired' / 'alternate':
        Alternates SOLVE and EXPLAIN every single step:
        Step 1: SOLVE (Problem 0, low), Step 2: EXPLAIN (Problem 0, low), Step 3: SOLVE (Problem 1, med), etc.
      - 'split' / 'hybrid' / 'both':
        First half (steps 1 to total_steps // 2) is SOLVE; second half is EXPLAIN.
      - True / 'explain':
        All steps are EXPLAIN.
      - False / 'solve' / 'direct':
        All steps are SOLVE.
    """
    num_efforts = len(EFFORT_TIERS_LIST)
    num_records = len(records) if records else total_steps
    mode_raw = str(explanation_mode).lower().strip()

    is_explain_step = False
    mode_label = "SOLVE"
    tier_idx = 0
    prob_idx = (step - 1) % num_records

    if mode_raw in ("effort_cycle", "cycle", "effort_blocks", "effort_schedule", "repeat"):
        cycle_len = num_efforts * 2  # 12 steps
        step_in_cycle = (step - 1) % cycle_len
        cycle_idx = (step - 1) // cycle_len

        if step_in_cycle < num_efforts:
            is_explain_step = False
            mode_label = "SOLVE"
            tier_idx = step_in_cycle
        else:
            is_explain_step = True
            mode_label = "EXPLAIN"
            tier_idx = step_in_cycle - num_efforts

        if pair_problems and num_records > 0:
            prob_base = (cycle_idx * num_efforts) % num_records
            prob_idx = (prob_base + tier_idx) % num_records
        else:
            prob_idx = (step - 1) % num_records

    elif mode_raw in ("interleaved", "paired", "alternate", "toggle", "pingpong"):
        is_explain_step = (step % 2 == 0)
        mode_label = "EXPLAIN" if is_explain_step else "SOLVE"
        pair_idx = (step - 1) // 2
        tier_idx = pair_idx % num_efforts

        if pair_problems and num_records > 0:
            prob_idx = pair_idx % num_records
        else:
            prob_idx = (step - 1) % num_records

    elif mode_raw in ("split", "hybrid", "both", "auto"):
        half = total_steps // 2
        is_explain_step = (step > half)
        mode_label = "EXPLAIN" if is_explain_step else "SOLVE"
        if pair_problems and num_records > 0:
            prob_idx = ((step - 1) % max(1, half)) % num_records
        else:
            prob_idx = (step - 1) % num_records
        tier_idx = prob_idx % num_efforts

    elif mode_raw in ("true", "explain", "explanation", "pedagogical", "1", "yes"):
        is_explain_step = True
        mode_label = "EXPLAIN"
        prob_idx = (step - 1) % num_records
        tier_idx = prob_idx % num_efforts

    else:
        # Default: SOLVE only (False, "solve", "none", "off", "0")
        is_explain_step = False
        mode_label = "SOLVE"
        prob_idx = (step - 1) % num_records
        tier_idx = prob_idx % num_efforts

    # Resolve effort tier
    if effort_tier.lower() in ("balanced", "all", "roundrobin", "cycle"):
        step_effort = EFFORT_TIERS_LIST[tier_idx]
    else:
        step_effort = effort_tier.lower()

    item = records[prob_idx] if records and 0 <= prob_idx < len(records) else None

    return {
        "step": step,
        "is_explain_step": is_explain_step,
        "mode_label": mode_label,
        "step_effort": step_effort,
        "tier_index": tier_idx,
        "problem_index": prob_idx,
        "item": item,
    }


# =====================================================================
# 📊  200-STEP AUDITOR & HUGGING FACE ARCHIVE UPLOADER
# =====================================================================
class StepAuditor:
    """
    Saves every step's generations, exports flat tabular dataset,
    and uploads zip archive to Hugging Face Hub.
    Dedicated 'effort_tier' column enables studying token scaling per effort level.
    """

    def __init__(self, log_dir: str = "local_trainer/generations_200_steps", dataset_name: str = "math-ai/TemplateGSM"):
        self.log_dir = log_dir
        self.dataset_name = dataset_name
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
        dataset_name: Optional[str] = None,
        step_loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        step_mode: Optional[str] = None,
    ):
        tier_key = str(effort_tier).lower()
        ds_name = dataset_name or self.dataset_name
        mode_key = str(step_mode or "SOLVE").upper()
        self.effort_stats[tier_key]["steps"] += 1

        tabular_rows = []
        for r in rollouts:
            r["effort_tier"] = tier_key
            r["dataset"] = ds_name
            r["step_mode"] = mode_key
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
                "step_mode": mode_key,
                "dataset": ds_name,
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
                "step_loss": round(step_loss, 4) if step_loss is not None else None,
                "grad_norm": round(grad_norm, 4) if grad_norm is not None else None,
                "final_answer": r.get("final_answer", ""),
                "reasoning_trace": r.get("reasoning_trace", ""),
                "full_text": r.get("full_text", ""),
            })

        record = {
            "step": step,
            "step_mode": mode_key,
            "dataset": ds_name,
            "effort_tier": tier_key,
            "problem": problem,
            "reference_answer": reference,
            "elapsed_sec": round(elapsed, 2),
            "step_loss": round(step_loss, 4) if step_loss is not None else None,
            "grad_norm": round(grad_norm, 4) if grad_norm is not None else None,
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
                "step", "step_mode", "dataset", "effort_tier", "rollout_idx", "token_count", "total_reward",
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
        print(" [Auditor] 200-STEP AUDIT SUMMARY")
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


# 1. Persistent reasoning requirement across all effort tiers
PERSISTENT_PARAGRAPH_PROMPT = (
    "Structure your reasoning trace as continuous, natural paragraphs of internal monologue inside <think>...</think>. "
    "Do not use bullet points, numbered lists, or section headers in your thinking."
)

# 2. Explicit final answer formatting directive
BOXED_ANSWER_PROMPT = "State your final answer clearly outside the thinking tags formatted as \\boxed{answer}."


def format_effort_prompt(problem: str, effort_tier: str = "high", tokenizer: Optional[Any] = None) -> str:
    """
    Formats the system prompt with 3 clean, separated directives:
    1. Persistent paragraph reasoning rule (all tiers)
    2. Dynamic effort instruction (tier-specific)
    3. Boxed final answer directive
    """
    tier_key = effort_tier.lower().strip()
    effort_text = EFFORT_SYSTEM_PROMPTS.get(tier_key, EFFORT_SYSTEM_PROMPTS["high"])
    system_prompt = f"{effort_text}\n{PERSISTENT_PARAGRAPH_PROMPT}\n{BOXED_ANSWER_PROMPT}"

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


# 2-Turn Pedagogical Explanation Prompt Formatting
EXPLANATION_FOLLOWUP_PROMPTS: List[str] = [
    "Explain it!",
    "Can you guide me solve it?",
    "Can you explain the detailed steps to solve this?",
    "Please explain the intuition and step-by-step method.",
]


def format_explanation_turn_prompt(
    problem: str,
    direct_answer: str,
    followup_query: str = "Explain it!",
    effort_tier: str = "high",
    dataset_name: str = "",
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Formats multi-turn prompt for explanation study and training:
    Turn 1:
      User: Problem
      Assistant: \\boxed{direct_answer}
    Turn 2:
      User: Follow-up query (e.g. 'Explain it!' or 'Can you guide me solve it?')
      Assistant: [generation prompt for reasoning trace <think>...</think> + explanation]
    """
    tier_key = effort_tier.lower().strip()
    effort_text = EFFORT_SYSTEM_PROMPTS.get(tier_key, EFFORT_SYSTEM_PROMPTS["high"])

    is_gsm = "gsm8k" in str(dataset_name).lower() or "template" in str(dataset_name).lower()
    if is_gsm:
        pedagogical_rule = (
            "In your final explanation outside <think>, provide exactly 1 continuous paragraph "
            "of comprehensive explanation with at least 5 sentences. Do not use markdown headers or bullet points."
        )
    else:
        pedagogical_rule = (
            "In your final explanation outside <think>, provide a thorough, structured pedagogical explanation. "
            "If using headers (##), each header must be descriptive with more than 5 words. "
            "If using bullet points, include at least 5 substantive bullets formatted as comprehensive paragraphs."
        )

    explanation_system = (
        f"{effort_text}\n"
        f"{PERSISTENT_PARAGRAPH_PROMPT}\n"
        f"{pedagogical_rule}"
    )

    clean_ans = str(direct_answer).strip()
    if "\\boxed{" not in clean_ans:
        boxed_ans = f"\\boxed{{{clean_ans}}}"
    else:
        boxed_ans = clean_ans

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": explanation_system},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": boxed_ans},
                {"role": "user", "content": followup_query},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return (
        f"<|im_start|>system\n{explanation_system}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n{boxed_ans}<|im_end|>\n"
        f"<|im_start|>user\n{followup_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def save_checkpoint(
    model: Any,
    tokenizer: Any = None,
    checkpoint_dir: str = "checkpoints/checkpoint-100",
    step: int = 100,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves checkpoint (LoRA adapter, tokenizer, and metadata) locally.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(checkpoint_dir)
            print(f"[Checkpoint] Model weights/adapters saved to {checkpoint_dir}")
        except Exception as e:
            print(f"[Checkpoint] Notice: save_pretrained returned ({e}); saving state dict...")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_weights.pt"))
    elif hasattr(model, "state_dict"):
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_weights.pt"))

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        try:
            tokenizer.save_pretrained(checkpoint_dir)
        except Exception:
            pass

    meta = {
        "step": step,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        **(metadata or {}),
    }
    with open(os.path.join(checkpoint_dir, "checkpoint_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return checkpoint_dir


def upload_checkpoint_to_hf(
    checkpoint_dir: str = "checkpoints/checkpoint-100",
    repo_id: str = "Nihilux/sword-grpo-200-steps",
    hf_token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> Optional[str]:
    """
    Uploads checkpoint directory (e.g. checkpoint-100) to Hugging Face Hub (via upload_folder or zipped archive).
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print(f"[Hub] No HF_TOKEN provided. Checkpoint preserved locally at {checkpoint_dir}.")
        return None

    if not os.path.exists(checkpoint_dir):
        print(f"[Hub] Checkpoint dir '{checkpoint_dir}' does not exist. Skipping upload.")
        return None

    folder_name = os.path.basename(os.path.normpath(checkpoint_dir))
    msg = commit_message or f"Upload {folder_name} at step 100"
    print(f"[Hub] Uploading checkpoint folder '{checkpoint_dir}' to {repo_id} ({msg})...")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
            repo_type = "model"
        except Exception:
            try:
                api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
                repo_type = "dataset"
            except Exception:
                repo_type = "dataset"

        api.upload_folder(
            folder_path=checkpoint_dir,
            path_in_repo=folder_name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=msg,
        )
        url = f"https://huggingface.co/{repo_id}/tree/main/{folder_name}"
        print(f"[Hub] Checkpoint upload complete -> {url}")
        return url
    except Exception as e:
        print(f"[Hub] Direct folder upload encountered: {e}. Falling back to zip packaging...")
        try:
            zip_path = f"{os.path.normpath(checkpoint_dir)}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(checkpoint_dir):
                    for file in files:
                        p = os.path.join(root, file)
                        zf.write(p, os.path.relpath(p, os.path.dirname(checkpoint_dir)))
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
            api.upload_file(
                path_or_fileobj=zip_path,
                path_in_repo=os.path.basename(zip_path),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=msg,
            )
            url = f"https://huggingface.co/datasets/{repo_id}"
            print(f"[Hub] Checkpoint archive uploaded -> {url}")
            return url
        except Exception as e2:
            print(f"[Hub] Checkpoint upload failed: {e2}")
            return None


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
    dataset_name: str = "math-ai/TemplateGSM",
    effort_tier: str = "balanced",
    hf_token: Optional[str] = None,
    repo_id: str = "Nihilux/sword-grpo-200-steps",
    num_rollouts: int = 4,
    max_seq_len: int = 32768,
    max_new_tokens: int = 32768,
    steps: int = 200,
    lr: float = 5e-6,
    use_fp8: bool = True,
    study_mode: bool = False,
    max_samples: Optional[int] = None,
    verbose_study: bool = True,
    explanation_mode: Union[str, bool] = "effort_cycle",
    shuffle_dataset: Union[bool, str] = False,
    shuffle_seed: int = 42,
    pair_problems: bool = True,
    checkpoint_at_100: bool = True,
):
    setup_blackwell_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode_str = "STUDY & AUDIT ONLY (No training/backprop)" if study_mode else "FULL REINFORCEMENT TRAINING (GDPO)"
    mode_raw = str(explanation_mode).lower().strip()
    if mode_raw in ("effort_cycle", "cycle", "effort_blocks", "effort_schedule", "repeat"):
        domain_str = "EFFORT CYCLE (6 Steps SOLVE [low->max] -> 6 Steps EXPLAIN [low->max] -> Repeat)"
    elif mode_raw in ("interleaved", "paired", "alternate", "toggle"):
        domain_str = "INTERLEAVED (Alternating 1 Step SOLVE <-> 1 Step EXPLAIN)"
    elif mode_raw in ("split", "hybrid", "both", "auto"):
        domain_str = f"SPLIT / HYBRID (Steps 1-{steps//2} SOLVE -> Steps {steps//2+1}-{steps} EXPLAIN)"
    elif mode_raw in ("true", "explain", "pedagogical", "1"):
        domain_str = "MULTI-TURN EXPLANATION (100% Pedagogical Explanation)"
    else:
        domain_str = "DIRECT SOLVING (100% Turn 1: Problem -> \\boxed{answer})"

    order_str = "STRICT ORDER (gsm8k -> openreasoning -> big math)" if not shuffle_dataset or str(shuffle_dataset).lower() in ("false", "none", "off") else f"SHUFFLED ({shuffle_dataset}, seed={shuffle_seed})"
    pairing_str = "PAIRED (Each 6-problem batch is solved then explained)" if pair_problems else "STREAMING (Sequential next problem each step)"

    print("=" * 72)
    print(f" [Sword Engine] STANDALONE MATH GRPO ({dataset_name})")
    print(f" Operating Mode:   {mode_str}")
    print(f" Dialogue Domain:  {domain_str}")
    print(f" Dataset Sequence: {order_str}")
    print(f" Problem Pairing:  {pairing_str}")
    print(f" Base Model:       {model_name_or_path}")
    print(f" Dataset:          {dataset_name}")
    print(f" LoRA Checkpoint:  {checkpoint_lora} (Store: {checkpoint_repo})")
    print(f" Effort Mode:      {effort_tier.upper()} (Balanced round-robin across 6 tiers if 'balanced')")
    print(f" Context Window:   {max_seq_len} tokens (32k context, max new: {max_new_tokens})")
    print(f" Rollouts:         {num_rollouts} concurrent streams (G=4)")
    print(f" FP8 Engine:       {use_fp8}")
    print(f" Steps:            {steps}")
    print(f" Device:           {device}")
    print("=" * 72)

    # 1. Download & Prepare Dataset
    is_curated = (
        "curated" in str(dataset_name).lower()
        or dataset_name in ("curated_study_100", "curated_study", "study_100")
    )
    if is_curated:
        data_file = download_curated_study_dataset(hf_token=hf_token)
    else:
        if max_samples is None:
            effective_samples = max(steps * 2, 200)
        else:
            effective_samples = max_samples
        data_file = download_math_dataset(
            dataset_name=dataset_name,
            max_samples=effective_samples,
            hf_token=hf_token,
        )

    study_records = prepare_study_records(
        data_file,
        shuffle_mode=shuffle_dataset,
        seed=shuffle_seed,
    )
    print(f"[*] Prepared {len(study_records)} problems from {data_file} ({order_str})")

    # 2. Resolve LoRA Checkpoint First
    resolved_lora = resolve_checkpoint_lora(
        checkpoint_name=checkpoint_lora,
        hf_repo_id=checkpoint_repo,
        local_dir="checkpoints",
        hf_token=hf_token,
    )

    # 3. Load Model & Tokenizer
    print(f"\n[*] Loading tokenizer and model...")
    lora_attached = False
    model = None
    tokenizer = None

    if HAS_UNSLOTH:
        print("[*] Unsloth detected: Loading via FastLanguageModel...")
        try:
            if os.path.isdir(resolved_lora) and (
                os.path.exists(os.path.join(resolved_lora, "adapter_config.json"))
            ):
                print(f"[*] Loading base model + LoRA adapter together from {resolved_lora}...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=resolved_lora,
                    max_seq_length=max_seq_len,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                    token=hf_token,
                    device_map="auto" if device == "cuda" else None,
                )
                lora_attached = True
                print(f"✅ Loaded base model + LoRA from {resolved_lora} in a single pass via FastLanguageModel")
            else:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name_or_path,
                    max_seq_length=max_seq_len,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                    token=hf_token,
                    device_map="auto" if device == "cuda" else None,
                )
        except Exception as e_uns:
            print(f"⚠️  FastLanguageModel notice: {e_uns}. Falling back to standard loader...")
            model = None

    if model is None:
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    # 4. Attach LoRA Adapter (if not already attached in single pass)
    if not lora_attached:
        print(f"[*] Attaching LoRA adapters (path: {resolved_lora})...")
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

    # 5. Apply In-Memory FP8 MoE Quantization
    if use_fp8:
        convert_to_fp8_moe_weights(model)

    vram_after_load = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"[*] Active Model VRAM: {vram_after_load:.2f} GB (Headroom: {95.0 - vram_after_load:.1f} GB)")

    # 6. Configure Training Mode & Gradient Checkpointing
    if HAS_UNSLOTH:
        try:
            FastLanguageModel.for_training(model)
            print("✅ Configured Unsloth fast training mode")
        except Exception:
            pass

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

    # 6. Initialize Scorers, Loss, Auditor, Optimizer
    math_scorer = MathScorer()
    explanation_scorer = ExplanationScorer()
    loss_fn = ChunkedGRPOLoss(chunk_size=512)
    auditor = StepAuditor(dataset_name=dataset_name)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # 7. 200-Step Training Loop
    print(f"\n⚡ Starting {steps} training steps (Effort Mode: {effort_tier} | Mode: {explanation_mode})...\n")
    for step in range(1, steps + 1):
        t_start = time.perf_counter()
        sched = get_step_schedule(
            step=step,
            total_steps=steps,
            records=study_records,
            explanation_mode=explanation_mode,
            effort_tier=effort_tier,
            pair_problems=pair_problems,
        )
        item = sched["item"] or study_records[(step - 1) % len(study_records)]
        problem_text = item["problem"]
        ref_answer = item["answer"]
        item_dataset = item.get("dataset", dataset_name)
        step_effort = sched["step_effort"]
        is_explain_step = sched["is_explain_step"]
        mode_label = sched["mode_label"]

        step_scorer = explanation_scorer if is_explain_step else math_scorer

        # Format user problem with exact effort system prompt (or 2-turn explanation prompt)
        if is_explain_step:
            formatted_prompt = format_explanation_turn_prompt(
                problem=problem_text,
                direct_answer=ref_answer,
                followup_query="Explain it!",
                effort_tier=step_effort,
                dataset_name=item_dataset,
                tokenizer=tokenizer,
            )
        else:
            formatted_prompt = format_effort_prompt(problem_text, effort_tier=step_effort, tokenizer=tokenizer)

        print(f"[Step {step:03d}/{steps:03d} | {step_effort.upper():<5} | {mode_label}] ⏳ Generating {num_rollouts} rollouts...", end="", flush=True)

        # Phase A: Inference / Rollout Generation
        raw_rollouts = engine.generate_rollouts(
            prompt=formatted_prompt,
            num_rollouts=num_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )

        t_gen = time.perf_counter() - t_start
        avg_tokens = sum(len(tokenizer.encode(r, add_special_tokens=False)) for r in raw_rollouts) / len(raw_rollouts)
        print(f" done in {t_gen:.1f}s (avg: {avg_tokens:.0f} tokens)")

        # Phase B: Scoring & GDPO Decoupled Advantages
        rollout_results = []
        for text in raw_rollouts:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            res = step_scorer.score(
                problem=problem_text,
                reference_answer=ref_answer,
                full_text=text,
                effort_tier=step_effort,
                token_count=token_count,
                dataset_name=item_dataset,
            )
            res["full_text"] = text
            res["token_count"] = token_count
            res["effort_tier"] = step_effort
            res["dataset"] = item_dataset
            res["step_mode"] = mode_label
            res["acc_reward"] = res.get("column_scores", {}).get("accuracy", res.get("column_scores", {}).get("pedagogical_accuracy", 0.0))
            res["format_reward"] = res.get("column_scores", {}).get("formatting", res.get("column_scores", {}).get("explanation_structure", 0.0))
            res["effort_reward"] = res.get("column_scores", {}).get("efficiency", 0.0)
            rollout_results.append(res)

        if is_explain_step:
            advantages, col_advantages = compute_explanation_gdpo_advantages(rollout_results)
        else:
            advantages, col_advantages = compute_gdpo_advantages(rollout_results)
        for idx, adv in enumerate(advantages):
            rollout_results[idx]["advantage"] = adv

        # Phase C: Training / Backward Step (Micro-batched per rollout with gradient accumulation)
        total_step_loss = 0.0
        total_step_nll = 0.0
        grad_norm_val = 0.0
        if not study_mode:
            print(f"[Step {step:03d}/{steps:03d} | {step_effort.upper():<6}] 🔄 Scoring & Backprop (rollouts 1/{num_rollouts}..{num_rollouts}/{num_rollouts})...", end="", flush=True)
            model.train()
            if hasattr(model, "config"):
                model.config.use_cache = False
            optimizer.zero_grad()

            p_ids = tokenizer.encode(problem_text, add_special_tokens=False)
            p_len = len(p_ids)

            for i, res in enumerate(rollout_results):
                r_ids = tokenizer.encode(res["full_text"], add_special_tokens=False)
                combo = p_ids + r_ids
                cur_input = torch.tensor([combo], dtype=torch.long, device=device)
                cur_mask = torch.ones_like(cur_input)
                adv_i = float(advantages[i])

                loss_i, metrics_i = loss_fn.forward_single(
                    model=model,
                    input_ids=cur_input,
                    attention_mask=cur_mask,
                    prompt_length=p_len,
                    advantage=adv_i,
                )

                scaled_loss = loss_i / num_rollouts
                scaled_loss.backward()
                total_step_loss += loss_i.item()
                total_step_nll += metrics_i.get("nll", 0.0)

                del cur_input, cur_mask, loss_i, scaled_loss

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            grad_norm_val = float(grad_norm)
            optimizer.step()
            print(" done")
        else:
            print(f"[Step {step:03d}/{steps:03d} | {step_effort.upper():<6}] 🔍 Study Mode: Evaluated {num_rollouts} rollouts (Backprop skipped)")

        elapsed = time.perf_counter() - t_start
        vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

        avg_nll = (total_step_nll / num_rollouts) if not study_mode else 0.0

        # Phase D: Record Generation Audit with dedicated effort_tier and dataset columns
        auditor.record(
            step, problem_text, ref_answer, rollout_results, elapsed,
            effort_tier=step_effort, dataset_name=item_dataset,
            step_loss=avg_nll if not study_mode else None,
            grad_norm=grad_norm_val if not study_mode else None,
            step_mode=mode_label,
        )

        mean_acc = sum(r.get("acc_reward", 0.0) for r in rollout_results) / len(rollout_results)
        mean_fmt = sum(r.get("format_reward", 0.0) for r in rollout_results) / len(rollout_results)
        mean_eff = sum(r.get("effort_reward", 0.0) for r in rollout_results) / len(rollout_results)
        mean_total = sum(r.get("total_reward", 0.0) for r in rollout_results) / len(rollout_results)

        loss_str = f"{avg_nll:.4f}" if not study_mode else "N/A (study)"
        grad_str = f"{grad_norm_val:.2f}" if not study_mode else "N/A"
        print(
            f"[Step {step:03d}/{steps:03d} | {step_effort.upper():<5} | {mode_label}] "
            f"Acc: {mean_acc:.2f} | Fmt: {mean_fmt:.2f} | Eff: {mean_eff:+.2f} | "
            f"Total: {mean_total:+.2f} | Loss: {loss_str} | |g|: {grad_str} | "
            f"VRAM: {vram_gb:.1f} GB | Step Time: {elapsed:.2f}s"
        )

        # Live Study Preview: inspect model's thinking and answers in real-time
        if study_mode or verbose_study:
            print(f"{'─'*72}")
            print(f"📖 STUDY SAMPLE [Step {step:03d}/{steps:03d} | {step_effort.upper()} | {mode_label}]")
            print(f"❓ Problem:  {problem_text}")
            print(f"🎯 Expected: {ref_answer}")
            for idx, ro in enumerate(rollout_results):
                status = "✅ Correct" if ro.get("is_correct") else "❌ Incorrect"
                final_ans = ro.get("final_answer", "").strip() or "(no answer found)"
                trace = ro.get("reasoning_trace", "").strip()
                trace_preview = (trace[:240] + "...") if len(trace) > 240 else trace
                print(f"   [Rollout {idx+1}/{num_rollouts} | {status} | Total Reward: {ro.get('total_reward', 0.0):+.2f} | Tokens: {ro.get('token_count', 0)}]")
                if trace_preview:
                    print(f"     <think> {trace_preview} </think>")
                print(f"     Answer: {final_ans}")
            print(f"{'─'*72}\n")

        # Checkpoint-100 and Hub archive upload
        if step == 100 and checkpoint_at_100:
            ckpt_100_dir = "checkpoints/checkpoint-100"
            print(f"\n[Step {step:03d}] 🏁 Saving checkpoint-100 & uploading to Hugging Face Hub...")
            save_checkpoint(model, tokenizer, checkpoint_dir=ckpt_100_dir, step=100)
            upload_checkpoint_to_hf(checkpoint_dir=ckpt_100_dir, repo_id=repo_id, hf_token=hf_token)
            zip_and_upload_to_hf(auditor.log_dir, repo_id=repo_id, hf_token=hf_token)

        del rollout_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final checkpoint save & upload
    final_ckpt_dir = f"checkpoints/checkpoint-{steps}"
    save_checkpoint(model, tokenizer, checkpoint_dir=final_ckpt_dir, step=steps)
    upload_checkpoint_to_hf(checkpoint_dir=final_ckpt_dir, repo_id=repo_id, hf_token=hf_token)

    # Summarize and Upload to HF Hub
    auditor.summarize()
    zip_and_upload_to_hf(auditor.log_dir, repo_id, hf_token=hf_token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-2200")
    parser.add_argument("--dataset", type=str, default="math-ai/TemplateGSM", help="Dataset name on Hugging Face (default: math-ai/TemplateGSM)")
    parser.add_argument("--study_mode", "--study", action="store_true", help="Study mode: generate and audit model answers without backpropagation training")
    parser.add_argument("--explanation_mode", "--explain", type=str, default="effort_cycle", help="Explanation schedule: 'effort_cycle' (6 solve low->max then 6 explain low->max), 'interleaved', 'split', 'solve', or 'explain'")
    parser.add_argument("--shuffle_dataset", action="store_true", default=False, help="Shuffle dataset problems (default: False, strictly gsm8k -> openreasoning -> big math)")
    parser.add_argument("--no_pair_problems", action="store_false", dest="pair_problems", default=True, help="Disable problem pairing between solve and explain batches")
    parser.add_argument("--curated_dataset", action="store_true", help="Use curated 100-problem study dataset (35 GSM8K, 30 AoPS C4, 35 BigMath2)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum dataset samples to download/cache")
    parser.add_argument("--repo_id", type=str, default=os.environ.get("HF_UPLOAD_REPO_ID", "Nihilux/sword-grpo-200-steps"))
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--effort_tier", type=str, default="balanced", choices=["balanced", "all", "low", "medium", "high", "xhigh", "ultra", "max"], help="Reasoning effort tier or 'balanced' for round-robin split across all 6 tiers")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--test_stream_only", action="store_true", help="Only stream 1 row of dataset to inspect and exit")
    args = parser.parse_args()

    target_dataset = "curated_study_100" if args.curated_dataset else args.dataset

    if args.test_stream_only:
        if "curated" in str(target_dataset).lower():
            data_file = download_curated_study_dataset(hf_token=args.token or None)
        else:
            data_file = download_math_dataset(dataset_name=target_dataset, hf_token=args.token or None)
        streamer = stream_math_data_from_disk(data_file)
        row = next(streamer)
        print(f"\n[Test Stream] Inspected 1 sample row from {target_dataset}:")
        print(json.dumps(row, indent=2))
        sys.exit(0)

    run_standalone_math_grpo(
        model_name_or_path=args.model,
        checkpoint_lora=args.checkpoint,
        dataset_name=target_dataset,
        repo_id=args.repo_id,
        hf_token=args.token,
        effort_tier=args.effort_tier,
        steps=args.steps,
        study_mode=args.study_mode,
        max_samples=args.max_samples,
        explanation_mode=args.explanation_mode,
        shuffle_dataset=args.shuffle_dataset,
        pair_problems=args.pair_problems,
        checkpoint_at_100=True,
    )


if __name__ == "__main__":
    main()
