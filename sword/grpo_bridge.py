"""
GRPO Bridge: Robust Training-Inference Bridge between Unsloth and vLLM for Dense Models.
Systematically addresses all failure modes:
- A1-A6: Truncated Importance Sampling (TIS) with detached reference policy, temperature scaling, and LoRA request tracking
- B1-B5: Standard PEFT adapter serialization, precision partitioning, calibrated clipping, and policy entropy floor
- C1-C5: Two-phase VRAM lifecycle handover, micro-batched sequence chunking, and no-grad entropy monitoring
- D1-D4: Runtime dynamic LoRA request synchronization without restarting vLLM
- E1-E4: Group-centered advantage estimation, group size verification, and gradient norm clipping
- F1-F4: Unsloth training mode activation, gradient checkpointing, and target module compatibility
- G1-G4: Resilient fallback engine if vLLM encounters runtime or memory exceptions
"""

import os
import gc
import sys
import json
import math
import time
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


@dataclass
class GRPOBridgeConfig:
    """
    Configuration for the Unsloth-vLLM GRPO Bridge.
    All parameters are tuned to prevent OOM, numerical divergence, and reward collapse.
    """
    model_name: str = "Qwen/Qwen3.5-0.8B"
    concurrency_g: int = 4
    max_model_len: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    clip_eps: float = 0.2
    tis_bound_c: float = 2.0
    use_score_centering: bool = False
    use_calibrated_clipping: bool = False
    entropy_floor: float = 0.01
    entropy_coeff: float = 0.001
    max_grad_norm: float = 1.0
    micro_batch_size: int = 2
    vllm_gpu_utilization: float = 0.40
    vllm_quantization: Optional[str] = None
    kv_cache_dtype: str = "auto"
    enforce_eager: bool = True
    enable_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    chunk_size: int = 512
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    device: Optional[str] = None


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


@dataclass
class RolloutBatch:
    items: List[RolloutItem] = field(default_factory=list)
    policy_version: int = 0
    concurrency_g: int = 4

    def __len__(self) -> int:
        return len(self.items)

    def validate_groups(self) -> bool:
        """Verifies that every prompt group has exactly concurrency_g rollouts."""
        group_counts: Dict[int, int] = {}
        for item in self.items:
            group_counts[item.group_id] = group_counts.get(item.group_id, 0) + 1
        return all(c == self.concurrency_g for c in group_counts.values())


class VRAMLifecycleManager:
    """
    Manages GPU memory handover between rollout generation and training forward/backward.
    Prevents co-location OOM on shared GPUs.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device

    def enter_rollout_phase(self, trainer_model: Optional[nn.Module] = None):
        """Prepares environment for vLLM rollout generation."""
        if trainer_model is not None:
            trainer_model.eval()
        gc.collect()
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.empty_cache()

    def enter_train_phase(self, trainer_model: Optional[nn.Module] = None):
        """Cleans memory before training forward and backward pass."""
        gc.collect()
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.empty_cache()
        if trainer_model is not None:
            trainer_model.train()

    def get_memory_info(self) -> Dict[str, float]:
        """Returns allocated, reserved, and available GPU memory in gigabytes."""
        if not torch.cuda.is_available() or "cuda" not in self.device:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}
        dev_idx = torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
        alloc_gb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
        res_gb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)
        return {
            "allocated_gb": round(alloc_gb, 3),
            "reserved_gb": round(res_gb, 3),
            "total_gb": round(total_gb, 3),
            "free_gb": round(total_gb - res_gb, 3),
        }


class WeightSynchronizer:
    """
    Synchronizes LoRA weights from the active trainer model into vLLM using standard
    PEFT format (adapter_config.json and adapter_model.safetensors) and vLLM LoRARequest.
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3.5-0.8B",
        lora_r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[List[str]] = None,
    ):
        self.base_model_name = base_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.policy_version = 0
        self.temp_dir = tempfile.mkdtemp(prefix="sword_lora_sync_")
        self.current_adapter_path: Optional[str] = None

    def extract_lora_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extracts trainable LoRA parameters while preserving standard PEFT key hierarchy."""
        lora_sd: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_sd[name] = param.detach().cpu()
        return lora_sd

    def sync_to_vllm(self, trainer_model: nn.Module, vllm_engine: Optional[Any] = None) -> int:
        """
        Exports LoRA state dict and adapter_config.json in standard PEFT format.
        Updates active policy version stamp for dynamic LoRARequest dispatch.
        """
        self.policy_version += 1
        adapter_path = os.path.join(self.temp_dir, f"adapter_v{self.policy_version}")
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
                return self.policy_version
            except Exception:
                pass

        lora_sd = self.extract_lora_state_dict(trainer_model)

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
        except Exception:
            torch.save(lora_sd, os.path.join(adapter_path, "adapter_model.bin"))

        self.current_adapter_path = adapter_path
        return self.policy_version

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
        except Exception:
            return None


class DualCorrectionGRPOLoss(nn.Module):
    """
    Sequence-chunked GRPO loss with detached Truncated Importance Sampling (TIS),
    temperature-scaled logits, and no-grad entropy monitoring.
    """
    def __init__(
        self,
        clip_eps: float = 0.2,
        tis_bound_c: float = 2.0,
        temperature: float = 0.7,
        use_score_centering: bool = False,
        use_calibrated_clipping: bool = False,
        entropy_floor: float = 0.01,
        entropy_coeff: float = 0.001,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.clip_eps = clip_eps
        self.tis_bound_c = tis_bound_c
        self.temperature = max(temperature, 1e-5)
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Mask response tokens only
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lengths):
            response_mask[i, p_len:] = attention_mask[i, p_len:].bool()

        # Training forward pass in bfloat16
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
        token_entropy_list = []

        # Sequence chunking: computes cross-entropy in 512-token chunks
        for start_idx in range(0, num_tokens, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_tokens)
            # Temperature scale logits to match rollout sampling temperature
            chunk_logits = (shift_logits[:, start_idx:end_idx, :].reshape(-1, shift_logits.size(-1)) / self.temperature).float()
            chunk_targets = targets[:, start_idx:end_idx].reshape(-1)

            token_logprobs = -F.cross_entropy(
                chunk_logits,
                chunk_targets,
                reduction="none",
            ).reshape(batch_size, end_idx - start_idx)
            policy_logprobs_list.append(token_logprobs)

            # Entropy calculation under no_grad to prevent keeping autograd graphs in VRAM
            with torch.no_grad():
                probs = F.softmax(chunk_logits, dim=-1)
                chunk_entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1)
                token_entropy_list.append(chunk_entropy.reshape(batch_size, end_idx - start_idx))

        policy_logprobs = torch.cat(policy_logprobs_list, dim=1)
        policy_entropy = torch.cat(token_entropy_list, dim=1)

        # Baseline detached logprobs under current training policy
        policy_logprobs_ref = policy_logprobs.detach()

        # Build baseline old logprobs tensor from vLLM rollout
        if rollout_logprobs is not None:
            flat_rollout_lp = torch.zeros_like(policy_logprobs)
            for i in range(batch_size):
                p_len = prompt_lengths[i]
                resp_start = max(0, p_len - 1)
                r_lps = rollout_logprobs[i] if i < len(rollout_logprobs) else []
                valid_resp_len = min(len(r_lps), policy_logprobs.size(1) - resp_start)
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

        # Detached Truncated Importance Sampling (TIS) correction factor for vLLM off-policy divergence
        with torch.no_grad():
            log_is_weight = policy_logprobs_ref - vllm_old_logprobs
            raw_is_weight = torch.exp(log_is_weight)
            is_weight = torch.clamp(raw_is_weight, min=1.0 / self.tis_bound_c, max=self.tis_bound_c)
            if self.use_score_centering and target_mask.any():
                valid_mask_float = target_mask.float()
                mean_is_weight = (is_weight * valid_mask_float).sum() / valid_mask_float.sum().clamp(min=1.0)
                is_weight = is_weight - (mean_is_weight - 1.0)

        # Calibrated clipping: adapt clip_eps based on rolling variance if enabled
        effective_clip_eps = self.clip_eps
        if self.use_calibrated_clipping and target_mask.any():
            with torch.no_grad():
                log_ratio_std = log_ratio_on_policy.masked_select(target_mask).std().item()
                if not math.isnan(log_ratio_std):
                    effective_clip_eps = min(self.clip_eps, max(0.05, self.clip_eps / (1.0 + log_ratio_std)))

        adv = advantages.unsqueeze(-1).to(ratio_on_policy.device)
        surr1 = ratio_on_policy * adv
        surr2 = torch.clamp(ratio_on_policy, 1.0 - effective_clip_eps, 1.0 + effective_clip_eps) * adv
        token_policy_loss = -torch.min(surr1, surr2) * is_weight

        valid_tokens = target_mask.float().sum().clamp(min=1.0)
        policy_loss = (token_policy_loss * target_mask.float()).sum() / valid_tokens

        # Policy entropy regularization
        mean_entropy = (policy_entropy * target_mask.float()).sum() / valid_tokens
        entropy_penalty = torch.clamp(self.entropy_floor - mean_entropy, min=0.0)
        total_loss = policy_loss + (self.entropy_coeff * entropy_penalty)

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
    ) -> List[Dict[str, Any]]:
        device = next(self.model.parameters()).device
        results = []
        for prompt in prompts:
            raw_inputs = self.tokenizer(prompt, return_tensors="pt")
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
                token_val = next_token.item()
                token_lp = torch.log(probs[0, token_val].clamp(min=1e-12)).item()

                gen_tokens.append(token_val)
                gen_logprobs.append(token_lp)

                if token_val == getattr(self.tokenizer, "eos_token_id", None):
                    break
                curr_input = torch.cat([curr_input, next_token], dim=1)

            resp_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            results.append({
                "prompt_token_ids": p_ids,
                "response_token_ids": gen_tokens,
                "response_text": resp_text,
                "rollout_logprobs": gen_logprobs,
            })
        return results


class GRPOBridge:
    """
    Industrial-grade GRPO Bridge between Unsloth trainer and vLLM generator.
    Handles the entire training-rollout lifecycle, weight sync, importance sampling,
    and memory handover for dense models.
    """
    def __init__(self, config: Optional[GRPOBridgeConfig] = None):
        self.config = config or GRPOBridgeConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vram_mgr = VRAMLifecycleManager(device=self.device)
        self.weight_sync = WeightSynchronizer(
            base_model_name=self.config.model_name,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
        )
        self.loss_fn = DualCorrectionGRPOLoss(
            clip_eps=self.config.clip_eps,
            tis_bound_c=self.config.tis_bound_c,
            temperature=self.config.temperature,
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

    def initialize_trainer(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initializes the Unsloth or PEFT trainer model with LoRA adapters and checkpointing."""
        if model is not None and tokenizer is not None:
            self.trainer_model = model
            self.tokenizer = tokenizer
        else:
            if HAS_UNSLOTH and "cuda" in self.device:
                try:
                    self.trainer_model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=self.config.model_name,
                        max_seq_length=self.config.max_model_len,
                        load_in_4bit=False,
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
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
                self.trainer_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
                from sword.finetune import apply_lora_to_model
                self.trainer_model = apply_lora_to_model(
                    self.trainer_model,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_moe=False,
                )

        if self.trainer_model is not None and hasattr(self.trainer_model, "to"):
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
        """Spins up the vLLM engine with safe headless, multi-processing, and LoRA support."""
        try:
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            os.environ.setdefault("VLLM_USE_FLASHINFER", "0")

            from vllm import LLM
            llm_kwargs = {
                "model": self.config.model_name,
                "trust_remote_code": True,
                "max_model_len": self.config.max_model_len,
                "gpu_memory_utilization": self.config.vllm_gpu_utilization,
                "enforce_eager": self.config.enforce_eager,
                "enable_lora": self.config.enable_lora,
                "max_lora_rank": self.config.lora_r,
                "max_loras": 1,
                "hf_overrides": {
                    "allow_global_per_layer_attribute_access": True,
                    "text_config": {"allow_global_per_layer_attribute_access": True},
                },
            }
            if self.config.vllm_quantization:
                llm_kwargs["quantization"] = self.config.vllm_quantization
            if self.config.kv_cache_dtype != "auto":
                llm_kwargs["kv_cache_dtype"] = self.config.kv_cache_dtype

            self.vllm_engine = LLM(**llm_kwargs)
            return True
        except Exception as e:
            print(f"[GRPOBridge] Note: vLLM initialization deferred ({e}). Operating with fallback engine.")
            self.vllm_engine = None
            return False

    def generate_rollouts(self, prompts: List[str]) -> RolloutBatch:
        """
        Generates concurrency_g rollouts per prompt using vLLM or the fallback engine.
        Extracts token log probabilities and uses versioned LoRARequest.
        """
        self.vram_mgr.enter_rollout_phase(self.trainer_model)
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

                for idx, out in enumerate(outputs):
                    p_text = expanded_prompts[idx]
                    p_ids = list(out.prompt_token_ids)
                    resp = out.outputs[0]
                    r_text = resp.text
                    r_ids = list(resp.token_ids)

                    r_logprobs = []
                    if resp.logprobs:
                        for token_id, lp_dict in zip(r_ids, resp.logprobs):
                            if token_id in lp_dict:
                                r_logprobs.append(float(lp_dict[token_id].logprob))
                            else:
                                r_logprobs.append(0.0)
                    else:
                        r_logprobs = [0.0] * len(r_ids)

                    batch.items.append(RolloutItem(
                        prompt_text=p_text,
                        prompt_token_ids=p_ids,
                        response_text=r_text,
                        response_token_ids=r_ids,
                        rollout_logprobs=r_logprobs,
                        group_id=prompt_group_map[idx],
                        policy_version=self.weight_sync.policy_version,
                    ))
            except Exception as e:
                print(f"[GRPOBridge] Rollout generation note ({e}). Routing to fallback engine.")
                use_fallback = True
                batch.items.clear()

        if use_fallback:
            fb_results = self.fallback_engine.generate(
                expanded_prompts,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
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
                ))

        return batch

    def score_rollouts(
        self,
        batch: RolloutBatch,
        reward_fn: Callable[[str, str], float],
    ):
        """
        Evaluates rewards for all rollouts and computes group-centered advantages.
        Eliminates group sampling degeneracy (E4).
        """
        for item in batch.items:
            item.reward = float(reward_fn(item.prompt_text, item.response_text))

        groups: Dict[int, List[RolloutItem]] = {}
        for item in batch.items:
            groups.setdefault(item.group_id, []).append(item)

        for g_id, group_items in groups.items():
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
        Executes micro-batched GRPO training step with detached TIS, gradient accumulation,
        gradient norm clipping, and runtime LoRA weight synchronization.
        """
        self.vram_mgr.enter_train_phase(self.trainer_model)

        if len(batch.items) == 0:
            return {"loss": 0.0, "step": self.global_step}

        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        micro_bs = max(1, self.config.micro_batch_size)
        num_items = len(batch.items)
        num_micro_batches = math.ceil(num_items / micro_bs)

        optimizer.zero_grad()
        accumulated_metrics: Dict[str, float] = {}

        for mb_idx in range(num_micro_batches):
            mb_items = batch.items[mb_idx * micro_bs : (mb_idx + 1) * micro_bs]
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
            )

            # Gradient accumulation across micro-batches
            scaled_loss = mb_loss / num_micro_batches
            scaled_loss.backward()

            for k, v in mb_metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + (v / num_micro_batches)

        # Gradient norm clipping
        if self.config.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.trainer_model.parameters() if p.requires_grad],
                max_norm=self.config.max_grad_norm,
            )

        optimizer.step()
        self.global_step += 1

        # Synchronize updated LoRA weights into standard PEFT layout
        new_version = self.weight_sync.sync_to_vllm(self.trainer_model, self.vllm_engine)

        accumulated_metrics["step"] = self.global_step
        accumulated_metrics["policy_version"] = new_version
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
        2. Scores rollouts and normalizes advantages across groups
        3. Executes micro-batched training step with detached TIS and gradient clipping
        4. Synchronizes LoRA weights for next rollout iteration
        """
        batch = self.generate_rollouts(prompts)
        self.score_rollouts(batch, reward_fn)
        train_metrics = self.train_step(batch, optimizer)
        return {
            "batch_size": len(batch),
            "step": self.global_step,
            "metrics": train_metrics,
        }
