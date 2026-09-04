from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PureFlashAttention
from .kv_cache import StaticKVCache


@dataclass
class FastTransformerConfig:
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA ratio 4:1
    head_dim: Optional[int] = 128
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dtype: torch.dtype = torch.bfloat16


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep fp32 accumulation for numerical stability
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class FastMLP(nn.Module):
    """SwiGLU MLP layer as used in LLaMA/Mistral/Qwen models."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FastDecoderLayer(nn.Module):
    def __init__(self, config: FastTransformerConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PureFlashAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FastMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[StaticKVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Residual + Attention
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed,
            cos=cos,
            sin=sin,
            start_pos=start_pos,
            kv_cache=kv_cache,
            attn_mask=attn_mask,
        )
        hidden_states = hidden_states + attn_out

        # Residual + MLP
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed)
        hidden_states = hidden_states + mlp_out

        return hidden_states


class FastTransformerModel(nn.Module):
    """
    Pure PyTorch Transformer model built from scratch for maximum throughput.
    Features:
    - Pre-computed RoPE frequencies
    - PureFlashAttention with zero-allocation Static KV caching
    - High compute-to-memory bandwidth efficiency
    """
    def __init__(self, config: FastTransformerConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            FastDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE table
        self.register_buffer(
            "cos_cached",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.empty(0),
            persistent=False,
        )
        self._init_rope()

    def _init_rope(self):
        dim = self.config.head_dim or (self.config.hidden_size // self.config.num_attention_heads)
        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(self.config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]  # [1, 1, max_seq_len, dim]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[StaticKVCache] = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for either prefill (seq_len > 1) or single-token decode (seq_len == 1).
        """
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Slice RoPE embeddings for current sequence range
        cos = self.cos_cached[:, :, start_pos : start_pos + seq_len, :]
        sin = self.sin_cached[:, :, start_pos : start_pos + seq_len, :]

        # Cast RoPE to model dtype
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                start_pos=start_pos,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)

        if return_logits:
            # During decode, we only need logits for the last token
            if seq_len == 1:
                return self.lm_head(hidden_states)
            else:
                return self.lm_head(hidden_states)
        return hidden_states

    @classmethod
    def from_huggingface_config(cls, hf_config):
        """Build FastTransformerConfig from a HuggingFace PretrainedConfig."""
        config = FastTransformerConfig(
            vocab_size=getattr(hf_config, "vocab_size", 32000),
            hidden_size=getattr(hf_config, "hidden_size", 2048),
            intermediate_size=getattr(hf_config, "intermediate_size", 5632),
            num_hidden_layers=getattr(hf_config, "num_hidden_layers", 16),
            num_attention_heads=getattr(hf_config, "num_attention_heads", 16),
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", 4),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 4096),
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        )
        return cls(config)
