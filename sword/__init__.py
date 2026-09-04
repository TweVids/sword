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
from .patcher import patch_qwen
from .loader import load_qwen_model
from .server import FastQwenServer

__version__ = "0.1.6"
print(f"[Sword] Version {__version__} loaded successfully.")

__all__ = [
    "PureFlashAttention",
    "apply_rotary_pos_emb",
    "StaticKVCache",
    "FastTransformerModel",
    "FastTransformerConfig",
    "SpeedEngine",
    "patch_qwen",
    "load_qwen_model",
    "FastQwenServer",
    "__version__",
]
