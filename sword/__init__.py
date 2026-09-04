"""
Sword: Pure-PyTorch High-Throughput Attention & Generation Engine.
Designed for high-speed batched generation (4+ concurrency) on modern GPUs (e.g. NVIDIA Blackwell)
without fragile external C++/CUDA dependencies like flash-attn or vLLM.
"""

from .attention import PureFlashAttention, apply_rotary_pos_emb
from .kv_cache import StaticKVCache
from .model import FastTransformerModel, FastTransformerConfig
from .engine import SpeedEngine

__all__ = [
    "PureFlashAttention",
    "apply_rotary_pos_emb",
    "StaticKVCache",
    "FastTransformerModel",
    "FastTransformerConfig",
    "SpeedEngine",
]
