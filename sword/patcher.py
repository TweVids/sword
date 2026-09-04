import types
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Optional, Tuple


def _fast_sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Pure-PyTorch Fast SDPA Attention kernel.
    Dispatches to FLASH_ATTENTION / EFFICIENT_ATTENTION on modern hardware (e.g. Blackwell).
    """
    if query.is_cuda:
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal and attention_mask is None,
                )
        except Exception:
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal and attention_mask is None,
            )
    else:
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal and attention_mask is None,
        )


def make_patched_qwen_attention_forward(original_forward):
    """
    Wraps Qwen attention forward to route through our pure-PyTorch FlashAttention SDPA
    and support static KV caching, preserving bitsandbytes 4-bit/8-bit linear projections.
    """
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        # If custom static cache is attached to self or passed in kwargs
        static_cache = getattr(self, "_sword_static_cache", None) or kwargs.get("sword_static_cache", None)
        layer_idx = getattr(self, "layer_idx", 0)

        bsz, q_len, _ = hidden_states.shape

        # Linear projections (works transparently with bitsandbytes Linear4bit / Linear8bitLt or standard Linear)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        cfg = getattr(self, "config", None)
        num_heads = getattr(self, "num_heads", getattr(self, "num_attention_heads", getattr(cfg, "num_attention_heads", 16)))
        num_kv_heads = getattr(self, "num_key_value_heads", getattr(cfg, "num_key_value_heads", num_heads))
        head_dim = getattr(self, "head_dim", getattr(cfg, "head_dim", query_states.shape[-1] // num_heads))

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE if position_embeddings provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            from .attention import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle static KV Cache if active
        is_causal = (q_len > 1)
        if static_cache is not None:
            start_pos = kwargs.get("start_pos", 0)
            key_states, value_states = static_cache.update(layer_idx, key_states, value_states, start_pos)
            if q_len == 1 and start_pos > 0:
                is_causal = False
        elif past_key_values is not None and hasattr(past_key_values, "update"):
            key_states, value_states = past_key_values.update(key_states, value_states, layer_idx, kwargs)

        # GQA expansion
        num_groups = num_heads // num_kv_heads
        if num_groups > 1:
            key_states = key_states.repeat_interleave(num_groups, dim=1)
            value_states = value_states.repeat_interleave(num_groups, dim=1)

        # Dispatch via our Pure FlashAttention SDPA
        attn_output = _fast_sdpa_attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            is_causal=is_causal and attention_mask is None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return (attn_output, None)

    return patched_forward


def patch_qwen(model):
    """
    Patches all attention layers of a Qwen (Qwen2, Qwen2.5, Qwen3, Qwen3.5) model
    with Sword's pure FlashAttention SDPA kernel.
    Preserves all weights and quantization (bitsandbytes 4-bit/8-bit).
    """
    patched_count = 0
    for name, module in model.named_modules():
        mod_type = module.__class__.__name__.lower()
        if "attention" in mod_type or "attn" in mod_type:
            if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                module.forward = types.MethodType(make_patched_qwen_attention_forward(module.forward), module)
                patched_count += 1
    print(f"[Sword] Patched {patched_count} attention modules with Pure FlashAttention SDPA.")
    return model
