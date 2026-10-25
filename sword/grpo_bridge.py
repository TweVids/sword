"""
GRPO Bridge: Robust Training-Inference Bridge between Unsloth and vLLM for Dense Models.
Systematically addresses all failure modes:
- A1-A6: Truncated Importance Sampling (TIS) with detached reference policy, temperature alignment, and LoRA request tracking
- B1-B5: Standard PEFT adapter serialization, precision partitioning, calibrated clipping, and policy entropy floor
- C1-C5: Two-phase VRAM lifecycle handover (sleep/wake), global token-normalized micro-batching, and adapter pruning
- D1-D4: Runtime dynamic LoRA request synchronization without restarting vLLM
- E1-E4: Group-centered advantage estimation, group size verification and filtering, and gradient norm clipping
- F1-F4: Unsloth training mode activation, gradient checkpointing, and target module compatibility
- G1-G4: Resilient fallback engine if vLLM encounters runtime or memory exceptions
"""

import os
import gc
import sys
import json
import math
import time
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import unsloth
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except Exception:
    HAS_UNSLOTH = False

from transformers import AutoTokenizer, AutoModelForCausalLM
from .rl.policy import token_statistics, safe_optimizer_step
from .defaults import DEFAULT_GRPO_MODEL


@dataclass
class GRPOBridgeConfig:
    """
    Configuration for the Unsloth-vLLM GRPO Bridge.
    All parameters are tuned to prevent OOM, numerical divergence, and reward collapse.
    """
    model_name: str = DEFAULT_GRPO_MODEL
    concurrency_g: int = 4
    max_model_len: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    temperature_scale: bool = False
    logprobs_mode: str = "raw_logprobs"
    clip_eps: float = 0.2
    tis_bound_c: float = 2.0
    use_score_centering: bool = False
    use_calibrated_clipping: bool = False
    entropy_floor: float = 0.01
    entropy_coeff: float = 0.001
    max_grad_norm: float = 1.0
    micro_batch_size: int = 2
    max_kept_adapters: int = 2
    enable_vllm_sleep: bool = True
    offload_trainer_during_rollout: bool = False
    drop_incomplete_groups: bool = True
    vllm_gpu_utilization: float = 0.40
    vllm_quantization: Optional[str] = None
    kv_cache_dtype: str = "auto"
    enforce_eager: bool = True
    enable_lora: bool = True
    speculative_ngram: bool = False
    num_speculative_tokens: int = 4
    ngram_prompt_lookup_max: int = 4
    hf_overrides: Optional[Dict[str, Any]] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    chunk_size: int = 512
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    device: Optional[str] = None
    adapter_sync_dir: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    hf_token: Optional[str] = field(default=None, repr=False)
    model_cache_dir: Optional[str] = None
    model_revision: Optional[str] = None
    local_files_only: bool = False
    require_vllm: bool = False

    def __post_init__(self):
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Choose either load_in_4bit or load_in_8bit.")
        if self.concurrency_g < 1 or self.micro_batch_size < 1 or self.chunk_size < 1:
            raise ValueError("rollout count, microbatch size, and chunk size must be positive")
        if self.max_new_tokens < 1 or self.max_model_len <= self.max_new_tokens:
            raise ValueError("max_model_len must leave room for prompt and response tokens")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1]")
        # Both backends report raw model probabilities, independent of sampling warpers.
        if self.logprobs_mode != "raw_logprobs" or self.temperature_scale:
            raise ValueError("bridge importance correction requires raw_logprobs and temperature_scale=False")


@dataclass
class RolloutItem:
    prompt_text: str
    prompt_token_ids: List[int]
    response_text: str
    response_token_ids: List[int]
    rollout_logprobs: List[float]
    reward: float = 0.0
    advantage: float = 0.0
    group_id: int = 0
    policy_version: int = 0
    is_truncated: bool = False


@dataclass
class RolloutBatch:
    items: List[RolloutItem] = field(default_factory=list)
    policy_version: int = 0
    concurrency_g: int = 4

    def __len__(self) -> int:
        return len(self.items)

    def validate_groups(self) -> bool:
        """Verifies that every prompt group has exactly concurrency_g rollouts."""
        if not self.items:
            return False
        group_counts: Dict[int, int] = {}
        for item in self.items:
            group_counts[item.group_id] = group_counts.get(item.group_id, 0) + 1
        return all(c == self.concurrency_g for c in group_counts.values())

    def filter_valid_groups(self, required_g: int) -> "RolloutBatch":
        """Filters batch to keep only complete groups of size required_g."""
        group_counts: Dict[int, int] = {}
        for item in self.items:
            group_counts[item.group_id] = group_counts.get(item.group_id, 0) + 1
        valid_group_ids = {gid for gid, c in group_counts.items() if c == required_g}
        self.items = [it for it in self.items if it.group_id in valid_group_ids]
        return self


class VRAMLifecycleManager:
    """
    Manages GPU memory handover between rollout generation and training forward/backward.
    Supports vLLM sleep mode and trainer offloading to prevent co-location OOM on shared GPUs.
    """
    def __init__(
        self,
        device: str = "cuda",
        enable_sleep: bool = True,
        offload_trainer: bool = False,
    ):
        self.device = device
        self.enable_sleep = enable_sleep
        self.offload_trainer = offload_trainer
        self._sleeping_engine = None

    def enter_rollout_phase(
        self,
        trainer_model: Optional[nn.Module] = None,
        vllm_engine: Optional[Any] = None,
    ):
        """Prepares environment for vLLM rollout generation."""
        # Release trainer allocations before acquiring rollout memory.
        if trainer_model is not None:
            trainer_model.eval()
            if self.offload_trainer and hasattr(trainer_model, "to"):
                trainer_model.to("cpu")

        gc.collect()
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.empty_cache()

        if vllm_engine is not None and self.enable_sleep and hasattr(vllm_engine, "wake_up"):
            vllm_engine.wake_up()
            self._sleeping_engine = None

    def enter_score_phase(self, vllm_engine: Optional[Any] = None):
        """Release rollout allocations before starting the external judge."""
        if vllm_engine is not None and self.enable_sleep and hasattr(vllm_engine, "sleep"):
            if self._sleeping_engine is not vllm_engine:
                vllm_engine.sleep()
                self._sleeping_engine = vllm_engine
        gc.collect()
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.empty_cache()

    def enter_train_phase(
        self,
        trainer_model: Optional[nn.Module] = None,
        vllm_engine: Optional[Any] = None,
    ):
        """Puts vLLM to sleep and prepares trainer for forward and backward passes."""
        # 1. Sleep vLLM engine to free KV cache blocks and execution memory
        self.enter_score_phase(vllm_engine)

        # 2. Return trainer model to training device
        if trainer_model is not None:
            if self.offload_trainer and hasattr(trainer_model, "to"):
                trainer_model.to(self.device)
            trainer_model.train()

        gc.collect()
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.empty_cache()

    def get_memory_info(self) -> Dict[str, float]:
        """Returns allocated, reserved, and available GPU memory in gigabytes."""
        if not torch.cuda.is_available() or "cuda" not in self.device:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}
        dev_idx = torch.device(self.device).index
        if dev_idx is None:
            dev_idx = torch.cuda.current_device()
        free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
        total_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
        alloc_gb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
        res_gb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)
        return {
            "allocated_gb": round(alloc_gb, 3),
            "reserved_gb": round(res_gb, 3),
            "total_gb": round(total_gb, 3),
            "free_gb": round(free_bytes / (1024 ** 3), 3),
        }


class WeightSynchronizer:
    """
    Synchronizes LoRA weights from the active trainer model into vLLM using standard
    PEFT format (adapter_config.json and adapter_model.safetensors) and vLLM LoRARequest.
    Includes automated adapter directory pruning to prevent disk leaks.
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3.5-0.8B",
        lora_r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        max_kept_adapters: int = 2,
        adapter_sync_dir: Optional[str] = None,
    ):
        self.base_model_name = base_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.max_kept_adapters = max(1, max_kept_adapters)
        self.policy_version = 0
        if adapter_sync_dir is not None:
            os.makedirs(adapter_sync_dir, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix="sword_lora_sync_", dir=adapter_sync_dir)
        self.current_adapter_path: Optional[str] = None

    def extract_lora_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extracts trainable LoRA parameters while preserving standard PEFT key hierarchy."""
        lora_sd: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if "lora_" in name:
                name = name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
                if name.endswith((".lora_A", ".lora_B")):
                    name += ".weight"
                if not name.startswith("base_model.model."):
                    name = "base_model.model." + name
                lora_sd[name] = param.detach().cpu().contiguous().clone()
        return lora_sd

    def sync_to_vllm(self, trainer_model: nn.Module, vllm_engine: Optional[Any] = None) -> int:
        """
        Exports LoRA state dict and adapter_config.json in standard PEFT format.
        Updates active policy version stamp for dynamic LoRARequest dispatch.
        """
        next_version = self.policy_version + 1
        adapter_path = os.path.join(self.temp_dir, f"adapter_v{next_version}")
        os.makedirs(adapter_path, exist_ok=True)

        if hasattr(trainer_model, "save_pretrained") and hasattr(trainer_model, "peft_config"):
            try:
                trainer_model.save_pretrained(adapter_path)
                cfg_file = os.path.join(adapter_path, "adapter_config.json")
                if os.path.exists(cfg_file):
                    with open(cfg_file, "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                    if not cfg_data.get("base_model_name_or_path"):
                        cfg_data["base_model_name_or_path"] = self.base_model_name
                        with open(cfg_file, "w", encoding="utf-8") as f:
                            json.dump(cfg_data, f, indent=2)
                self.current_adapter_path = adapter_path
                self.policy_version = next_version
                return self.policy_version
            except Exception:
                # A PEFT export failure must not fall through to an incomplete manual adapter.
                raise

        lora_sd = self.extract_lora_state_dict(trainer_model)
        if not lora_sd:
            if vllm_engine is not None:
                raise ValueError("Cannot synchronize full-weight training as a LoRA adapter")
            # In-process fallback already shares the trainer's full parameters.
            self.policy_version = next_version
            return self.policy_version

        # Write standard PEFT adapter_config.json
        peft_config = {
            "base_model_name_or_path": self.base_model_name,
            "bias": "none",
            "lora_alpha": self.lora_alpha,
            "lora_dropout": 0.0,
            "r": self.lora_r,
            "target_modules": self.target_modules,
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
        }
        with open(os.path.join(adapter_path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(peft_config, f, indent=2)

        # Write adapter weights
        try:
            from safetensors.torch import save_file
            save_file(lora_sd, os.path.join(adapter_path, "adapter_model.safetensors"))
        except ImportError:
            torch.save(lora_sd, os.path.join(adapter_path, "adapter_model.bin"))

        self.current_adapter_path = adapter_path
        self.policy_version = next_version
        return self.policy_version

    def prune_old_adapters(self, max_kept: Optional[int] = None):
        """Deletes older adapter versions to prevent disk leaks."""
        limit = max_kept or self.max_kept_adapters
        if not os.path.exists(self.temp_dir):
            return
        for item in os.listdir(self.temp_dir):
            if item.startswith("adapter_v"):
                try:
                    ver = int(item[len("adapter_v"):])
                    if ver <= self.policy_version - limit:
                        shutil.rmtree(os.path.join(self.temp_dir, item), ignore_errors=True)
                except ValueError:
                    pass

    def cleanup(self):
        """Removes the entire temporary adapter directory on teardown."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def get_current_lora_request(self) -> Optional[Any]:
        """Creates a vLLM LoRARequest object for the current synchronized version."""
        if self.policy_version <= 0 or self.current_adapter_path is None:
            return None
        try:
            from vllm.lora.request import LoRARequest
            return LoRARequest(
                lora_name=f"adapter_v{self.policy_version}",
                lora_int_id=self.policy_version,
                lora_path=self.current_adapter_path,
            )
        except Exception as exc:
            raise RuntimeError("Cannot construct LoRA request for synchronized trainer policy") from exc


class DualCorrectionGRPOLoss(nn.Module):
    """
    Sequence-chunked GRPO loss with detached Truncated Importance Sampling (TIS),
    upper-bound truncation, differentiable entropy floors, and global token reduction.
    """
    def __init__(
        self,
        clip_eps: float = 0.2,
        tis_bound_c: float = 2.0,
        temperature: float = 0.7,
        temperature_scale: bool = False,
        use_score_centering: bool = False,
        use_calibrated_clipping: bool = False,
        entropy_floor: float = 0.01,
        entropy_coeff: float = 0.001,
        chunk_size: int = 512,
    ):
        super().__init__()
        if chunk_size < 1 or tis_bound_c < 1.0 or not math.isfinite(tis_bound_c):
            raise ValueError(
                "chunk_size must be positive and tis_bound_c must be finite and >= 1.0 "
                "(bound < 1.0 would clamp every importance weight below 1 and silently shrink the gradient)"
            )
        self.clip_eps = clip_eps
        self.tis_bound_c = tis_bound_c
        self.temperature = max(temperature, 1e-5)
        self.temperature_scale = temperature_scale
        self.use_score_centering = use_score_centering
        self.use_calibrated_clipping = use_calibrated_clipping
        self.entropy_floor = entropy_floor
        self.entropy_coeff = entropy_coeff
        self.chunk_size = chunk_size

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: List[int],
        advantages: torch.Tensor,
        rollout_logprobs: Optional[List[List[float]]] = None,
        global_valid_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Mask response tokens only
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lengths):
            response_mask[i, p_len:] = attention_mask[i, p_len:].bool()

        target_mask = response_mask[:, 1:]
        policy_logprobs, policy_entropy = token_statistics(
            model, input_ids, attention_mask, self.chunk_size,
            temperature=self.temperature if self.temperature_scale else 1.0,
            with_entropy=True,
        )

        # Baseline detached logprobs under current training policy
        policy_logprobs_ref = policy_logprobs.detach()

        # Build baseline old logprobs tensor from vLLM rollout
        if rollout_logprobs is not None:
            if len(rollout_logprobs) != batch_size:
                raise ValueError("rollout logprobs must contain one list per sequence")
            flat_rollout_lp = policy_logprobs_ref.clone()
            for i in range(batch_size):
                p_len = prompt_lengths[i]
                resp_start = max(0, p_len - 1)
                r_lps = rollout_logprobs[i]
                valid_resp_len = int(target_mask[i].sum().item())
                if len(r_lps) != valid_resp_len or not all(math.isfinite(lp) for lp in r_lps):
                    raise ValueError("rollout logprobs must be finite and match every response token")
                if valid_resp_len > 0:
                    flat_rollout_lp[i, resp_start : resp_start + valid_resp_len] = torch.tensor(
                        r_lps[:valid_resp_len], dtype=torch.float32, device=device
                    )
            vllm_old_logprobs = flat_rollout_lp
        else:
            vllm_old_logprobs = policy_logprobs_ref

        # On-policy surrogate ratio: starts at 1.0 at step 0 so clipping is not prematurely triggered
        log_ratio_on_policy = policy_logprobs - policy_logprobs_ref
        ratio_on_policy = torch.exp(log_ratio_on_policy)

        # Detached Truncated Importance Sampling (TIS): standard upper-bound truncation,
        # BUG M1 fix: self-normalization by E[min(rho, C)] is now applied
        # unconditionally so the loss scale is invariant to the rollout/policy
        # mismatch. Previously the normalization lived behind the
        # use_score_centering flag (default OFF), letting the effective
        # learning rate silently ride E[min(rho, C)] as the policies drifted.
        with torch.no_grad():
            log_is_weight = policy_logprobs_ref - vllm_old_logprobs
            is_weight = torch.exp(log_is_weight.clamp(max=math.log(self.tis_bound_c)))
            if target_mask.any():
                valid_mask_float = target_mask.float()
                mean_is_weight = (is_weight * valid_mask_float).sum() / valid_mask_float.sum().clamp(min=1.0)
                is_weight = is_weight / mean_is_weight.clamp(min=1e-8)

        # Calibrated clipping: adapt clip_eps based on rolling variance if enabled
        effective_clip_eps = self.clip_eps
        if self.use_calibrated_clipping and target_mask.any():
            with torch.no_grad():
                log_ratio_std = log_ratio_on_policy.masked_select(target_mask).std(unbiased=False).item()
                if not math.isnan(log_ratio_std):
                    effective_clip_eps = min(self.clip_eps, max(0.05, self.clip_eps / (1.0 + log_ratio_std)))

        adv = advantages.unsqueeze(-1).to(ratio_on_policy.device)
        surr1 = ratio_on_policy * adv
        surr2 = torch.clamp(ratio_on_policy, 1.0 - effective_clip_eps, 1.0 + effective_clip_eps) * adv
        token_policy_loss = -torch.min(surr1, surr2) * is_weight

        # Global or local token reduction
        valid_tokens = target_mask.float().sum().clamp(min=1.0)
        norm_factor = float(global_valid_tokens) if global_valid_tokens is not None else valid_tokens.item()
        policy_loss = (token_policy_loss * target_mask.float()).sum() / max(1.0, norm_factor)

        # Policy entropy regularization
        mean_entropy = (policy_entropy * target_mask.float()).sum() / valid_tokens
        entropy_penalty = torch.clamp(self.entropy_floor - policy_entropy, min=0.0)
        entropy_loss = (entropy_penalty * target_mask.float()).sum() / max(1.0, norm_factor)
        total_loss = policy_loss + self.entropy_coeff * entropy_loss

        metrics = {
            "grpo_loss": round(total_loss.item(), 5),
            "policy_loss": round(policy_loss.item(), 5),
            "mean_ratio": round(ratio_on_policy.masked_select(target_mask).mean().item() if target_mask.any() else 1.0, 4),
            "mean_is_weight": round(is_weight.masked_select(target_mask).mean().item() if target_mask.any() else 1.0, 4),
            "mean_entropy": round(mean_entropy.item(), 4),
            "effective_clip_eps": round(effective_clip_eps, 4),
            "mean_advantage": round(advantages.mean().item(), 4),
        }
        return total_loss, metrics


class FallbackRolloutEngine:
    """
    Lightweight, in-process PyTorch rollout engine for fallback when vLLM is unavailable
    or encounters runtime CUDA/multiprocessing exceptions.
    """
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_model_len: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        device = next(self.model.parameters()).device
        results = []
        eos_ids = getattr(self.tokenizer, "eos_token_id", None)
        eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
        for prompt in prompts:
            tokenization_args = {"return_tensors": "pt"}
            if max_model_len is not None:
                tokenization_args.update(truncation=True, max_length=max_model_len - max_new_tokens)
            raw_inputs = self.tokenizer(prompt, **tokenization_args)
            if hasattr(raw_inputs, "to"):
                inputs = raw_inputs.to(device)
            elif isinstance(raw_inputs, dict):
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in raw_inputs.items()}
            else:
                inputs = raw_inputs
            p_ids = inputs["input_ids"][0].tolist()

            gen_tokens = []
            gen_logprobs = []
            curr_input = inputs["input_ids"]

            for _ in range(max_new_tokens):
                out = self.model(input_ids=curr_input, use_cache=False)
                next_logits = out.logits[:, -1, :].float()
                probs = F.softmax(next_logits / max(temperature, 1e-5), dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    sorted_probs[sorted_indices_to_remove] = 0.0
                    probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)
                if top_p < 1.0:
                    next_token = sorted_indices.gather(-1, next_token)
                token_val = next_token.item()
                # Match vLLM raw_logprobs, before temperature and top-p processing.
                token_lp = F.log_softmax(next_logits, dim=-1)[0, token_val].item()

                gen_tokens.append(token_val)
                gen_logprobs.append(token_lp)

                if token_val in eos_ids:
                    break
                curr_input = torch.cat([curr_input, next_token], dim=1)

            resp_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            results.append({
                "prompt_token_ids": p_ids,
                "response_token_ids": gen_tokens,
                "response_text": resp_text,
                "rollout_logprobs": gen_logprobs,
                "is_truncated": len(gen_tokens) >= max_new_tokens and gen_tokens[-1] not in eos_ids,
            })
        return results


class GRPOBridge:
    """
    Industrial-grade GRPO Bridge between Unsloth trainer and vLLM generator.
    Handles the entire training-rollout lifecycle, weight sync, importance sampling,
    group integrity, and memory handover for dense models.
    """
    def __init__(self, config: Optional[GRPOBridgeConfig] = None):
        self.config = config or GRPOBridgeConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vram_mgr = VRAMLifecycleManager(
            device=self.device,
            enable_sleep=self.config.enable_vllm_sleep,
            offload_trainer=self.config.offload_trainer_during_rollout,
        )
        self.weight_sync = WeightSynchronizer(
            base_model_name=self.config.model_name,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            max_kept_adapters=self.config.max_kept_adapters,
            adapter_sync_dir=self.config.adapter_sync_dir,
        )
        self.loss_fn = DualCorrectionGRPOLoss(
            clip_eps=self.config.clip_eps,
            tis_bound_c=self.config.tis_bound_c,
            temperature=self.config.temperature,
            temperature_scale=self.config.temperature_scale,
            use_score_centering=self.config.use_score_centering,
            use_calibrated_clipping=self.config.use_calibrated_clipping,
            entropy_floor=self.config.entropy_floor,
            entropy_coeff=self.config.entropy_coeff,
            chunk_size=self.config.chunk_size,
        )
        self.trainer_model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.vllm_engine: Optional[Any] = None
        self.fallback_engine: Optional[FallbackRolloutEngine] = None
        self.global_step: int = 0
        self.mock_test = False
        self.vllm_initialization_error = None

    def initialize_trainer(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        mock_test: bool = False,
        **kwargs,
    ):
        """Initializes the Unsloth or PEFT trainer model with LoRA adapters and checkpointing."""
        self.mock_test = bool(mock_test or kwargs.get("mock_test", False))
        load_options = {
            "token": self.config.hf_token, "cache_dir": self.config.model_cache_dir,
            "revision": self.config.model_revision, "local_files_only": self.config.local_files_only,
        }
        if model is not None and tokenizer is not None:
            self.trainer_model = model
            self.tokenizer = tokenizer
        else:
            is_mock = self.mock_test or self.config.model_name in ("dummy", "test_model", "mock")
            self.mock_test = is_mock
            if HAS_UNSLOTH and "cuda" in self.device and not is_mock:
                try:
                    self.trainer_model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=self.config.model_name,
                        max_seq_length=self.config.max_model_len,
                        load_in_4bit=self.config.load_in_4bit,
                        load_in_8bit=self.config.load_in_8bit,
                        **load_options,
                    )
                    self.trainer_model = FastLanguageModel.get_peft_model(
                        self.trainer_model,
                        r=self.config.lora_r,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        target_modules=self.config.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
                    )
                except Exception:
                    self.trainer_model = None

            if self.trainer_model is None:
                if is_mock:
                    from transformers import AutoConfig
                    cfg = AutoConfig.for_model(
                        "qwen2",
                        vocab_size=500,
                        hidden_size=64,
                        intermediate_size=128,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        max_position_embeddings=256,
                    )
                    self.trainer_model = AutoModelForCausalLM.from_config(cfg)
                    class MockTokenizer:
                        vocab_size = 500
                        pad_token_id = 0
                        eos_token_id = 1
                        def __call__(self, text, return_tensors="pt", **_kw):
                            return {"input_ids": torch.tensor([[2, 10, 20, 30]], dtype=torch.long)}
                        def decode(self, token_ids, skip_special_tokens=True):
                            return f"mock output ({len(token_ids)} tokens)"
                    self.tokenizer = MockTokenizer()
                else:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.config.model_name, trust_remote_code=True, **load_options,
                        )
                        model_options = dict(load_options)
                        if self.config.load_in_4bit or self.config.load_in_8bit:
                            from transformers import BitsAndBytesConfig
                            model_options.update(
                                quantization_config=BitsAndBytesConfig(
                                    load_in_4bit=self.config.load_in_4bit,
                                    load_in_8bit=self.config.load_in_8bit,
                                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                ),
                                device_map={"": self.device},
                            )
                        self.trainer_model = AutoModelForCausalLM.from_pretrained(
                            self.config.model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                            **model_options,
                        )
                        if self.config.load_in_4bit or self.config.load_in_8bit:
                            from peft import prepare_model_for_kbit_training
                            self.trainer_model = prepare_model_for_kbit_training(self.trainer_model)
                    except Exception as e_load:
                        raise RuntimeError(f"Failed to load requested trainer model {self.config.model_name}") from e_load

                from peft import LoraConfig, get_peft_model
                adapter_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none", task_type="CAUSAL_LM",
                )
                self.trainer_model = get_peft_model(self.trainer_model, adapter_config)

        if (self.trainer_model is not None and hasattr(self.trainer_model, "to")
                and not getattr(self.trainer_model, "is_loaded_in_4bit", False)
                and not getattr(self.trainer_model, "is_loaded_in_8bit", False)):
            self.trainer_model = self.trainer_model.to(self.device)

        # Configure fast training mode and gradient checkpointing
        if HAS_UNSLOTH:
            try:
                FastLanguageModel.for_training(self.trainer_model)
            except Exception:
                pass

        if hasattr(self.trainer_model, "gradient_checkpointing_enable"):
            try:
                self.trainer_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception:
                try:
                    self.trainer_model.gradient_checkpointing_enable()
                except Exception:
                    pass

        if hasattr(self.trainer_model, "enable_input_require_grads"):
            try:
                self.trainer_model.enable_input_require_grads()
            except Exception:
                pass

        self.fallback_engine = FallbackRolloutEngine(self.trainer_model, self.tokenizer)

    def initialize_vllm(self) -> bool:
        """Spins up the vLLM engine with safe headless, multi-processing, sleep, and LoRA support."""
        if self.mock_test:
            return False
        try:
            has_adapter = any("lora_" in name for name, _ in self.trainer_model.named_parameters())
            if has_adapter and not self.config.enable_lora:
                raise ValueError("trainer has LoRA adapters but vLLM LoRA is disabled")
            if any(p.requires_grad and "lora_" not in name for name, p in self.trainer_model.named_parameters()):
                raise ValueError("full-weight training requires the in-process rollout engine")
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            os.environ["VLLM_USE_FLASHINFER"] = "0"

            hf_over = dict(self.config.hf_overrides or {})

            from vllm import LLM
            if self.config.offload_trainer_during_rollout:
                self.trainer_model.eval()
                self.trainer_model.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            llm_kwargs = {
                "model": self.config.model_name,
                "trust_remote_code": True,
                "max_model_len": self.config.max_model_len,
                "gpu_memory_utilization": self.config.vllm_gpu_utilization,
                "enforce_eager": self.config.enforce_eager,
                "enable_lora": self.config.enable_lora,
                "max_lora_rank": self.config.lora_r,
                "max_loras": 1,
                "hf_overrides": hf_over,
            }
            if self.config.hf_token:
                llm_kwargs["hf_token"] = self.config.hf_token
            if self.config.model_cache_dir:
                llm_kwargs["download_dir"] = self.config.model_cache_dir
            if self.config.model_revision:
                llm_kwargs["revision"] = self.config.model_revision
            if self.config.vllm_quantization:
                llm_kwargs["quantization"] = self.config.vllm_quantization
            if self.config.kv_cache_dtype != "auto":
                llm_kwargs["kv_cache_dtype"] = self.config.kv_cache_dtype
            if self.config.enable_vllm_sleep:
                llm_kwargs["enable_sleep_mode"] = True
            if self.config.logprobs_mode:
                llm_kwargs["logprobs_mode"] = self.config.logprobs_mode

            # Current vLLM config uses one structured speculative configuration.
            if self.config.speculative_ngram:
                llm_kwargs["speculative_config"] = {
                    "method": "ngram", "num_speculative_tokens": self.config.num_speculative_tokens,
                    "prompt_lookup_max": self.config.ngram_prompt_lookup_max,
                }

            try:
                self.vllm_engine = LLM(**llm_kwargs)
            except TypeError:
                # Tolerate versions without sleep_mode, logprobs_mode, or speculative params
                llm_kwargs.pop("speculative_model", None)
                llm_kwargs.pop("num_speculative_tokens", None)
                llm_kwargs.pop("ngram_prompt_lookup_max", None)
                llm_kwargs.pop("speculative_config", None)
                llm_kwargs.pop("enable_sleep_mode", None)
                llm_kwargs.pop("logprobs_mode", None)
                try:
                    self.vllm_engine = LLM(**llm_kwargs)
                except TypeError:
                    llm_kwargs.pop("hf_overrides", None)
                    self.vllm_engine = LLM(**llm_kwargs)

            # Export the initial (or resumed) adapter before first inference.
            if has_adapter and self.weight_sync.policy_version == 0:
                self.weight_sync.sync_to_vllm(self.trainer_model)
            return True
        except Exception as e:
            self.vllm_initialization_error = str(e)
            print(f"[GRPOBridge] Note: vLLM initialization deferred ({e}). Operating with fallback engine.")
            self.vllm_engine = None
            if self.config.offload_trainer_during_rollout:
                self.trainer_model.to(self.device)
            if self.config.require_vllm:
                raise RuntimeError(f"Target-environment vLLM check failed: {e}") from e
            return False

    def generate_rollouts(self, prompts: List[str]) -> RolloutBatch:
        """
        Generates concurrency_g rollouts per prompt using vLLM or the fallback engine.
        Extracts token log probabilities and uses versioned LoRARequest.
        """
        batch = RolloutBatch(
            policy_version=self.weight_sync.policy_version,
            concurrency_g=self.config.concurrency_g,
        )

        expanded_prompts = []
        prompt_group_map = []
        for g_idx, prompt in enumerate(prompts):
            for _ in range(self.config.concurrency_g):
                expanded_prompts.append(prompt)
                prompt_group_map.append(g_idx)

        use_fallback = self.vllm_engine is None
        if not use_fallback:
            try:
                self.vram_mgr.enter_rollout_phase(self.trainer_model, self.vllm_engine)
                from vllm import SamplingParams
                sp = SamplingParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_new_tokens,
                    logprobs=1,
                )
                lora_req = self.weight_sync.get_current_lora_request()
                if lora_req is not None:
                    outputs = self.vllm_engine.generate(expanded_prompts, sp, lora_request=lora_req)
                else:
                    outputs = self.vllm_engine.generate(expanded_prompts, sp)

                if len(outputs) != len(expanded_prompts):
                    raise ValueError("vLLM returned an incomplete rollout batch")

                for idx, out in enumerate(outputs):
                    p_text = expanded_prompts[idx]
                    p_ids = list(out.prompt_token_ids)
                    resp = out.outputs[0]
                    r_text = resp.text
                    r_ids = list(resp.token_ids)

                    r_logprobs = []
                    if resp.logprobs is not None and len(resp.logprobs) == len(r_ids):
                        for token_id, lp_dict in zip(r_ids, resp.logprobs):
                            if token_id in lp_dict:
                                r_logprobs.append(float(lp_dict[token_id].logprob))
                            else:
                                raise ValueError("vLLM omitted the sampled token logprob")
                    else:
                        raise ValueError("vLLM did not return complete token logprobs")
                    if not all(math.isfinite(lp) for lp in r_logprobs):
                        raise ValueError("vLLM returned nonfinite token logprobs")

                    batch.items.append(RolloutItem(
                        prompt_text=p_text,
                        prompt_token_ids=p_ids,
                        response_text=r_text,
                        response_token_ids=r_ids,
                        rollout_logprobs=r_logprobs,
                        group_id=prompt_group_map[idx],
                        policy_version=self.weight_sync.policy_version,
                        is_truncated=getattr(resp, "finish_reason", None) == "length",
                    ))
            except Exception as e:
                print(f"[GRPOBridge] Rollout generation note ({e}). Routing to fallback engine.")
                use_fallback = True
                batch.items.clear()

        if use_fallback:
            # Failed rollout may have left vLLM resident and trainer offloaded.
            self.vram_mgr.enter_train_phase(self.trainer_model, self.vllm_engine)
            self.trainer_model.eval()
            self.fallback_engine.model = self.trainer_model
            fb_results = self.fallback_engine.generate(
                expanded_prompts,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_model_len=self.config.max_model_len,
            )
            for idx, res in enumerate(fb_results):
                batch.items.append(RolloutItem(
                    prompt_text=expanded_prompts[idx],
                    prompt_token_ids=res["prompt_token_ids"],
                    response_text=res["response_text"],
                    response_token_ids=res["response_token_ids"],
                    rollout_logprobs=res["rollout_logprobs"],
                    group_id=prompt_group_map[idx],
                    policy_version=self.weight_sync.policy_version,
                    is_truncated=res.get("is_truncated", False),
                ))

        return batch

    def generate_single(self, prompt: str) -> str:
        """
        Generates a single model completion for prompt.
        Used for multi-turn harness interaction and conversational rollouts.
        """
        saved_g = self.config.concurrency_g
        self.config.concurrency_g = 1
        try:
            batch = self.generate_rollouts([prompt])
            if batch.items:
                return batch.items[0].response_text
            return ""
        finally:
            self.config.concurrency_g = saved_g

    def score_rollouts(
        self,
        batch: RolloutBatch,
        reward_fn: Callable[[str, str], float],
    ):
        """
        Evaluates rewards for all rollouts, validates and filters complete groups,
        and computes group-centered advantages.
        Eliminates group sampling degeneracy (E4).
        """
        if self.config.drop_incomplete_groups and not batch.validate_groups():
            batch.filter_valid_groups(self.config.concurrency_g)

        for item in batch.items:
            item.reward = float(reward_fn(item.prompt_text, item.response_text))
            if not math.isfinite(item.reward):
                raise ValueError("GRPO rewards must be finite")

        groups: Dict[int, List[RolloutItem]] = {}
        for item in batch.items:
            groups.setdefault(item.group_id, []).append(item)

        for g_id, group_items in groups.items():
            if len(group_items) != self.config.concurrency_g and self.config.drop_incomplete_groups:
                continue

            rewards = [it.reward for it in group_items]
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
            std_r = math.sqrt(var_r)

            for it in group_items:
                if std_r > 1e-6:
                    it.advantage = (it.reward - mean_r) / (std_r + 1e-6)
                else:
                    it.advantage = 0.0

    def train_step(
        self,
        batch: RolloutBatch,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        """
        Executes micro-batched GRPO training step with detached TIS, global token normalization,
        gradient accumulation, gradient norm clipping, and automated adapter directory pruning.
        """
        if len(batch.items) == 0:
            return {"loss": 0.0, "step": self.global_step}
        if batch.policy_version != self.weight_sync.policy_version or any(
            it.policy_version != batch.policy_version for it in batch.items
        ):
            raise ValueError("stale or mixed policy versions in rollout batch")
        if any(not it.prompt_token_ids for it in batch.items):
            raise ValueError("each rollout needs at least one prompt token for causal prediction")
        if self.config.drop_incomplete_groups:
            batch.filter_valid_groups(batch.concurrency_g)
            if not batch.items:
                return {"loss": 0.0, "step": self.global_step}
        self.vram_mgr.enter_train_phase(self.trainer_model, self.vllm_engine)

        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        micro_bs = max(1, self.config.micro_batch_size)
        num_items = len(batch.items)
        if self.config.use_score_centering and micro_bs < num_items:
            raise ValueError("importance-weight score centering requires a single microbatch")
        num_micro_batches = math.ceil(num_items / micro_bs)

        # Global response token count for exact token-level loss reduction
        global_valid_tokens = max(1, sum(len(it.response_token_ids) for it in batch.items))

        optimizer.zero_grad()
        accumulated_metrics: Dict[str, float] = {}

        for mb_idx in range(num_micro_batches):
            mb_items = batch.items[mb_idx * micro_bs : (mb_idx + 1) * micro_bs]
            if not mb_items:
                continue

            padded_inputs = []
            prompt_lens = []
            advantages = []
            rollout_lps = []

            for it in mb_items:
                seq = it.prompt_token_ids + it.response_token_ids
                padded_inputs.append(seq)
                prompt_lens.append(len(it.prompt_token_ids))
                advantages.append(it.advantage)
                rollout_lps.append(it.rollout_logprobs)

            max_seq_len = max(len(s) for s in padded_inputs)
            input_ids_tensor = torch.full((len(padded_inputs), max_seq_len), pad_id, dtype=torch.long, device=self.device)
            attention_mask_tensor = torch.zeros_like(input_ids_tensor)

            for i, seq in enumerate(padded_inputs):
                input_ids_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                attention_mask_tensor[i, : len(seq)] = 1

            adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)

            mb_loss, mb_metrics = self.loss_fn(
                model=self.trainer_model,
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                prompt_lengths=prompt_lens,
                advantages=adv_tensor,
                rollout_logprobs=rollout_lps,
                global_valid_tokens=global_valid_tokens,
            )
            if not torch.isfinite(mb_loss):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("Nonfinite GRPO loss; optimizer update cancelled")

            # mb_loss is already scaled by global_valid_tokens
            mb_loss.backward()

            for k, v in mb_metrics.items():
                weight = 1.0 if k in ("grpo_loss", "policy_loss") else sum(len(it.response_token_ids) for it in mb_items) / global_valid_tokens
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v * weight

        # Gradient norm clipping and gradient norm tracking
        grad_norm_val = 0.0
        trainable_params = [p for p in self.trainer_model.parameters() if p.requires_grad]
        if trainable_params:
            if self.config.max_grad_norm > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                grad_norm_val = float(total_norm.item() if hasattr(total_norm, "item") else total_norm)
            else:
                total_sq = sum(p.grad.data.norm(2).item() ** 2 for p in trainable_params if p.grad is not None)
                grad_norm_val = math.sqrt(total_sq)
                if not math.isfinite(grad_norm_val):
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError("Nonfinite GRPO gradient; optimizer update cancelled")

        # Adapter export is part of the update: a failed export must restore the
        # optimizer/policy rather than retry an already-applied update.
        new_version = safe_optimizer_step(
            optimizer, trainable_params,
            after_step=lambda: self.weight_sync.sync_to_vllm(self.trainer_model, self.vllm_engine),
        )
        # Do not carry gradient buffers into the next memory handoff.
        optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

        # Synchronize updated LoRA weights into standard PEFT layout and prune old directories
        self.weight_sync.prune_old_adapters(max_kept=self.config.max_kept_adapters)

        accumulated_metrics["step"] = self.global_step
        accumulated_metrics["policy_version"] = new_version
        accumulated_metrics["grad_norm"] = round(grad_norm_val, 5)
        accumulated_metrics["vram"] = self.vram_mgr.get_memory_info()
        return accumulated_metrics

    def step(
        self,
        prompts: List[str],
        reward_fn: Callable[[str, str], float],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end iteration of the GRPO bridge:
        1. Generates rollouts with token logprobs and versioned LoRA request
        2. Scores rollouts and normalizes advantages across verified groups
        3. Executes micro-batched training step with detached TIS and gradient clipping
        4. Synchronizes LoRA weights and prunes old adapters
        """
        batch = self.generate_rollouts(prompts)
        self.score_rollouts(batch, reward_fn)
        train_metrics = self.train_step(batch, optimizer)
        return {
            "batch_size": len(batch),
            "step": self.global_step,
            "metrics": train_metrics,
        }
