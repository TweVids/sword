"""
Sword: Pure-PyTorch High-Throughput Attention & Generation Speed Engine.
"""

import sys


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

from .attention import PureFlashAttention, apply_rotary_pos_emb
from .kv_cache import StaticKVCache
from .model import FastTransformerModel, FastTransformerConfig
from .engine import SpeedEngine
from .patcher import (
    patch_model,
    patch_moe,
    patch_moe_experts,
    patch_qwen,
    unpatch_model,
    unpatch_moe,
    unpatch_qwen,
    set_attention_mode,
)
from .loader import load_qwen_model, load_moe_model
from .server import FastServer, FastMoEServer, FastQwenServer
from .finetune import benchmark_finetune_8k, apply_lora_to_model
from .trainer import (
    setup_blackwell_environment,
    download_from_drive,
    download_hf_checkpoint,
    is_valid_checkpoint,
    load_offline_dataset,
    FullCheckpointCallback,
)

__version__ = "0.5.0"
print(f"[Sword] Version {__version__} loaded successfully.")

__all__ = [
    "PureFlashAttention",
    "apply_rotary_pos_emb",
    "StaticKVCache",
    "FastTransformerModel",
    "FastTransformerConfig",
    "SpeedEngine",
    "patch_model",
    "patch_moe",
    "patch_moe_experts",
    "patch_qwen",
    "unpatch_model",
    "unpatch_moe",
    "unpatch_qwen",
    "set_attention_mode",
    "load_qwen_model",
    "load_moe_model",
    "FastServer",
    "FastMoEServer",
    "FastQwenServer",
    "benchmark_finetune_8k",
    "apply_lora_to_model",
    "setup_blackwell_environment",
    "download_from_drive",
    "download_hf_checkpoint",
    "is_valid_checkpoint",
    "load_offline_dataset",
    "FullCheckpointCallback",
    "__version__",
]
