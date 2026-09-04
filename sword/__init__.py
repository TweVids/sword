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

from .attention import PureFlashAttention, apply_rotary_pos_emb
from .kv_cache import StaticKVCache
from .model import FastTransformerModel, FastTransformerConfig
from .engine import SpeedEngine
from .patcher import patch_qwen, unpatch_qwen, set_attention_mode
from .loader import load_qwen_model
from .server import FastQwenServer
from .finetune import benchmark_finetune_8k, apply_lora_to_model
from .trainer import (
    setup_blackwell_environment,
    download_from_drive,
    download_hf_checkpoint,
    is_valid_checkpoint,
    load_offline_dataset,
    FullCheckpointCallback,
)

__version__ = "0.4.0"
print(f"[Sword] Version {__version__} loaded successfully.")

__all__ = [
    "PureFlashAttention",
    "apply_rotary_pos_emb",
    "StaticKVCache",
    "FastTransformerModel",
    "FastTransformerConfig",
    "SpeedEngine",
    "patch_qwen",
    "unpatch_qwen",
    "set_attention_mode",
    "load_qwen_model",
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
