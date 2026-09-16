"""
Sword: Pure-PyTorch High-Throughput Attention & Generation Speed Engine.
"""

import sys

try:
    import unsloth
except Exception:
    pass


def _fix_torchvision_compatibility():
    """
    Prevents crashes caused by mismatched or broken torchvision C++ extension binaries
    (e.g., 'RuntimeError: operator torchvision::nms does not exist') when transformers
    imports image/multimodal utilities.
    """
    try:
        import torchvision
        _ = torchvision.ops.nms
    except Exception:
        sys.modules["torchvision"] = None
        sys.modules["torchvision.io"] = None
        sys.modules["torchvision.ops"] = None


_fix_torchvision_compatibility()


def _fix_transformers_fp8_quantizer_bug():
    """
    Hotfixes upstream Transformers bug in FineGrainedFP8HfQuantizer.update_tp_plan
    where layer_overrides is None, causing AttributeError: 'NoneType' object has no attribute 'get'
    when loading models like tencent/Hy-MT2-30B-A3B-FP8.
    """
    try:
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
        orig_update_tp_plan = getattr(FineGrainedFP8HfQuantizer, "update_tp_plan", None)
        if orig_update_tp_plan is not None and not getattr(orig_update_tp_plan, "_sword_patched", False):
            def patched_update_tp_plan(self, config, _orig=orig_update_tp_plan):
                try:
                    return _orig(self, config)
                except AttributeError as err:
                    if "'NoneType' object has no attribute 'get'" in str(err):
                        return config
                    raise
            patched_update_tp_plan._sword_patched = True
            FineGrainedFP8HfQuantizer.update_tp_plan = patched_update_tp_plan
    except Exception:
        pass

    try:
        from transformers.integrations import finegrained_fp8
        for fn_name in ("fp8_grouped_mm_experts_forward", "fp8_batched_mm_experts_forward"):
            if hasattr(finegrained_fp8, fn_name):
                orig_forward_fn = getattr(finegrained_fp8, fn_name)
                if not getattr(orig_forward_fn, "_sword_patched", False):
                    def make_safe_wrapper(target_fn):
                        def safe_wrapper(self, *args, **kwargs):
                            if getattr(self, "activation_scheme", None) == "static":
                                self.activation_scheme = "dynamic"
                            try:
                                return target_fn(self, *args, **kwargs)
                            except (ImportError, NotImplementedError):
                                fast_fwd = getattr(self, "_sword_fast_forward", None)
                                if fast_fwd is not None:
                                    return fast_fwd(*args, **kwargs)
                                orig_fwd = getattr(self, "_sword_original_forward", None)
                                if orig_fwd is not None:
                                    return orig_fwd(*args, **kwargs)
                                raise
                        return safe_wrapper
                    wrapped_fn = make_safe_wrapper(orig_forward_fn)
                    wrapped_fn._sword_patched = True
                    setattr(finegrained_fp8, fn_name, wrapped_fn)
    except Exception:
        pass


_fix_transformers_fp8_quantizer_bug()


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
        import torch
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


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
    """Fallback standard RoPE parameter computation for default rope_type."""
    import torch
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", 1536)
        num_heads = getattr(config, "num_attention_heads", 16)
        head_dim = hidden_size // num_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _fix_transformers_bailing_compatibility():
    """
    Hotfixes upstream Transformers (v4.46+) compatibility:
    1. `is_torch_fx_available` was removed from `transformers.utils.import_utils`, which causes
       `modeling_bailing_moe_v3.py` to fail with ImportError.
    2. In modern transformers, `rope_scaling` property returns a default dict without 'factor',
       which causes `BailingMoeV3MultiLatentAttention` to fail with KeyError: 'factor'.
    3. `ROPE_INIT_FUNCTIONS` in transformers does not contain 'default', which causes
       `BailingMoeV3RotaryEmbedding` to fail with KeyError: 'default'.
    """
    try:
        import transformers.utils.import_utils as import_utils
        if not hasattr(import_utils, "is_torch_fx_available"):
            def is_torch_fx_available():
                try:
                    import torch.fx
                    return True
                except Exception:
                    return False
            import_utils.is_torch_fx_available = is_torch_fx_available

        import transformers.utils as utils
        if not hasattr(utils, "is_torch_fx_available"):
            utils.is_torch_fx_available = import_utils.is_torch_fx_available

        import transformers
        if not hasattr(transformers, "is_torch_fx_available"):
            transformers.is_torch_fx_available = import_utils.is_torch_fx_available
    except Exception:
        pass

    try:
        # Register 'default' in ROPE_INIT_FUNCTIONS to prevent KeyError: 'default'
        import transformers.modeling_rope_utils as rope_utils
        if hasattr(rope_utils, "ROPE_INIT_FUNCTIONS") and "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
            rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    except Exception:
        pass

    try:
        # Patch dynamic module loading to intercept Bailing classes and sanitize config.rope_scaling
        import transformers.dynamic_module_utils as dyn_utils
        if not getattr(dyn_utils, "_sword_bailing_hooked", False):
            orig_get_class = dyn_utils.get_class_from_dynamic_module
            def patched_get_class(class_reference, pretrained_model_name_or_path, **kwargs):
                cls = orig_get_class(class_reference, pretrained_model_name_or_path, **kwargs)
                if cls is not None:
                    mod = sys.modules.get(cls.__module__)
                    if mod and hasattr(mod, "ROPE_INIT_FUNCTIONS") and "default" not in mod.ROPE_INIT_FUNCTIONS:
                        mod.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
                    if hasattr(cls, "__name__"):
                        if "BailingMoeV3" in cls.__name__ or "MultiLatentAttention" in cls.__name__:
                            if not getattr(cls, "_sword_rope_patched", False):
                                orig_init = cls.__init__
                                def safe_init(self, *args, **kw):
                                    cfg = args[0] if args else kw.get("config", getattr(self, "config", None))
                                    if cfg is not None and hasattr(cfg, "rope_scaling") and isinstance(cfg.rope_scaling, dict):
                                        if cfg.rope_scaling.get("rope_type") == "default":
                                            cfg.rope_scaling = None
                                        elif "factor" not in cfg.rope_scaling:
                                            cfg.rope_scaling["factor"] = 1.0
                                    return orig_init(self, *args, **kw)
                                cls.__init__ = safe_init
                                cls._sword_rope_patched = True
                return cls
            dyn_utils.get_class_from_dynamic_module = patched_get_class
            dyn_utils._sword_bailing_hooked = True
    except Exception:
        pass

    try:
        from transformers import AutoConfig
        if not getattr(AutoConfig, "_sword_bailing_hooked", False):
            orig_config_from_pretrained = AutoConfig.from_pretrained
            @classmethod
            def safe_config_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                cfg = orig_config_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
                if getattr(cfg, "rope_scaling", None) is not None:
                    if isinstance(cfg.rope_scaling, dict) and cfg.rope_scaling.get("rope_type") == "default":
                        cfg.rope_scaling = None
                    elif isinstance(cfg.rope_scaling, dict) and "factor" not in cfg.rope_scaling:
                        cfg.rope_scaling["factor"] = 1.0
                return cfg
            AutoConfig.from_pretrained = safe_config_from_pretrained
            AutoConfig._sword_bailing_hooked = True
    except Exception:
        pass


_fix_transformers_bailing_compatibility()

from .attention import PureFlashAttention, apply_rotary_pos_emb

from .kv_cache import StaticKVCache, SmartKVCache
from .model import FastTransformerModel, FastTransformerConfig
from .engine import SpeedEngine
from .patcher import (
    patch_model,
    patch_moe,
    patch_moe_experts,
    patch_qwen,
    patch_qwen3_moe,
    patch_ling,
    unpatch_model,
    unpatch_moe,
    unpatch_qwen,
    unpatch_qwen3_moe,
    unpatch_ling,
    set_attention_mode,
    convert_to_fp8,
    convert_moe_to_fp8,
)
from .loader import load_qwen_model, load_moe_model, load_qwen3_moe_model, load_ling_model
from .ling import FastLingServer, setup_fla_compatibility
from .server import FastServer, FastMoEServer, FastQwenServer, FastQwen3MoeServer
from .finetune import benchmark_finetune_8k, apply_lora_to_model
from .trainer import (
    setup_blackwell_environment,
    download_from_drive,
    download_hf_checkpoint,
    is_valid_checkpoint,
    load_offline_dataset,
    FullCheckpointCallback,
)
from .rl import (
    start_grpo,
    GRPOTrainer,
    ContinuousStreamingQueue,
    PrimaryScorer,
    ExternalVerifier,
    MoERouterMonitor,
    ChunkedGRPOLoss,
)
from .gym import (
    DockerCodingGym,
    GymInstance,
    GymExecutionResult,
    OpenSWEAdapter,
    ScaleSWEAdapter,
    load_swe_dataset,
    extract_unified_diff,
)

__version__ = "0.9.2"
print(f"[Sword] Version {__version__} loaded successfully.")


def run_standalone_math_grpo(*args, **kwargs):
    """Convenience entry point for running Math GRPO training directly from sword."""
    from sword.train_math_grpo import run_standalone_math_grpo as _run
    return _run(*args, **kwargs)


__all__ = [
    "run_standalone_math_grpo",
    "PureFlashAttention",
    "apply_rotary_pos_emb",
    "StaticKVCache",
    "SmartKVCache",
    "FastTransformerModel",
    "FastTransformerConfig",
    "SpeedEngine",
    "patch_model",
    "patch_moe",
    "patch_moe_experts",
    "patch_qwen",
    "patch_qwen3_moe",
    "patch_ling",
    "unpatch_model",
    "unpatch_moe",
    "unpatch_qwen",
    "unpatch_qwen3_moe",
    "unpatch_ling",
    "set_attention_mode",
    "convert_to_fp8",
    "convert_moe_to_fp8",
    "load_qwen_model",
    "load_moe_model",
    "load_qwen3_moe_model",
    "load_ling_model",
    "FastServer",
    "FastMoEServer",
    "FastQwenServer",
    "FastQwen3MoeServer",
    "FastLingServer",
    "setup_fla_compatibility",
    "benchmark_finetune_8k",
    "apply_lora_to_model",
    "setup_blackwell_environment",
    "download_from_drive",
    "download_hf_checkpoint",
    "is_valid_checkpoint",
    "load_offline_dataset",
    "FullCheckpointCallback",
    "start_grpo",
    "GRPOTrainer",
    "ContinuousStreamingQueue",
    "PrimaryScorer",
    "ExternalVerifier",
    "MoERouterMonitor",
    "ChunkedGRPOLoss",
    "DockerCodingGym",
    "GymInstance",
    "GymExecutionResult",
    "OpenSWEAdapter",
    "ScaleSWEAdapter",
    "load_swe_dataset",
    "extract_unified_diff",
    "__version__",
]


