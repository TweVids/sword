import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .kv_cache import StaticKVCache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) to Query and Key tensors.
    
    Compatible with HuggingFace/Llama/Qwen2/Qwen3.5 conventions.
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_kv_heads, seq_len, head_dim]
    cos, sin: [seq_len, dim], [batch_size, seq_len, dim], or [batch_size, 1, seq_len, dim]
    """
    if position_ids is not None:
        # Index cos/sin by position_ids if non-sequential
        cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
        sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)

    # Ensure cos and sin match 4D shape [batch, 1, seq_len, rotary_dim] for proper broadcasting across heads
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.ndim > 4:
        while cos.ndim > 4:
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
        while cos.ndim < 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    if rotary_dim < q.shape[-1]:
        # Partial RoPE (e.g. Qwen 3.5 with partial_rotary_factor=0.25)
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

        q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        # Full RoPE
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class PureFlashAttention(nn.Module):
    """
    Pure-PyTorch Attention layer using F.scaled_dot_product_attention (SDPA).
    
    Zero external C++ dependency (no flash-attn or vLLM package required).
    Dispatches directly to FlashAttention-2/3 kernels natively on modern GPUs (e.g. Blackwell).
    Supports Grouped-Query Attention (GQA) and static KV cache slicing.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: Optional[int] = None,
        dropout_p: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[StaticKVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for attention.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            cos, sin: RoPE embeddings
            start_pos: Starting position for KV cache insertion (0 during prefill)
            kv_cache: Optional pre-allocated StaticKVCache
            attn_mask: Optional attention mask
            is_causal: Whether to apply causal masking
        """
        bsz, seq_len, _ = hidden_states.shape

        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [bsz, num_heads, seq_len, head_dim]
        q = q.view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v, start_pos)
            # If we are in decode step (seq_len == 1) and we have historical tokens, causal mask is false
            # because we want Q to attend to all past tokens in K, V.
            if seq_len == 1 and start_pos > 0:
                is_causal = False

        # Grouped-Query Attention (GQA) KV head expansion — zero-copy expand+reshape
        if self.num_key_value_groups > 1:
            bsz_, num_kv, slen, hdim = k.shape
            k = k[:, :, None, :, :].expand(bsz_, num_kv, self.num_key_value_groups, slen, hdim).reshape(bsz_, num_kv * self.num_key_value_groups, slen, hdim)
            v = v[:, :, None, :, :].expand(bsz_, num_kv, self.num_key_value_groups, slen, hdim).reshape(bsz_, num_kv * self.num_key_value_groups, slen, hdim)

        # Fast Attention via PyTorch Native SDPA.
        # Let PyTorch's own dispatcher pick the best backend (FlashAttention, efficient, math).
        # Matches patcher.py behavior — no backend constraint needed; contiguous ensures FA2 eligibility.
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal and attn_mask is None,
        )

        # Reshape back: [bsz, seq_len, num_attention_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)
