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
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Pure-PyTorch Fast SDPA Attention kernel.
    Dispatches to FLASH_ATTENTION / EFFICIENT_ATTENTION on modern hardware (e.g. Blackwell).
    Ensures contiguous memory for all inputs and bias masks.
    """
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if attention_mask is not None:
        attention_mask = attention_mask.contiguous()

    if query.is_cuda:
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal and attention_mask is None,
                    scale=scale,
                )
        except Exception:
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal and attention_mask is None,
                scale=scale,
            )
    else:
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal and attention_mask is None,
            scale=scale,
        )


def make_patched_qwen_attention_forward(original_forward):
    """
    Wraps Qwen attention forward to route through our pure-PyTorch FlashAttention SDPA
    and support static KV caching, preserving bitsandbytes 4-bit/8-bit linear projections.
    Natively supports both standard Qwen (Qwen2/2.5) and Qwen3.5 (QK-norm + gated attention).
    """
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        static_cache = getattr(self, "_sword_static_cache", None) or kwargs.get("sword_static_cache", None)
        layer_idx = getattr(self, "layer_idx", 0)

        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        # -------------------------------------------------------------
        # 1. Dynamic Q/K/V Projections (handles standard Qwen and Qwen3.5)
        # -------------------------------------------------------------
        query_raw = self.q_proj(hidden_states)
        key_raw = self.k_proj(hidden_states)
        value_raw = self.v_proj(hidden_states)

        cfg = getattr(self, "config", None)
        t_cfg = getattr(cfg, "text_config", cfg)

        num_heads = getattr(self, "num_heads", getattr(self, "num_attention_heads", getattr(t_cfg, "num_attention_heads", 16)))
        num_kv_heads = getattr(self, "num_key_value_heads", getattr(t_cfg, "num_key_value_heads", num_heads))
        head_dim = getattr(self, "head_dim", getattr(t_cfg, "head_dim", None))

        if head_dim is None:
            head_dim = key_raw.shape[-1] // num_kv_heads

        hidden_shape = (*input_shape, -1, head_dim)
        gate = None

        # Qwen 3.5 dual projection [query, gate] where output features == 2 * num_heads * head_dim
        # Slicing must be done along the head dimension:
        # q_proj_out.view(*input_shape, -1, head_dim * 2).chunk(2, dim=-1)
        if query_raw.shape[-1] == num_heads * head_dim * 2:
            query_states, gate = torch.chunk(
                query_raw.view(*input_shape, -1, head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
            query_states = query_states.view(hidden_shape)
        else:
            query_states = query_raw.view(hidden_shape)

        # Apply QK normalization if present
        if hasattr(self, "q_norm") and self.q_norm is not None:
            query_states = self.q_norm(query_states)

        hidden_shape_k = (*input_shape, -1, head_dim)
        key_states = key_raw.view(hidden_shape_k)
        if hasattr(self, "k_norm") and self.k_norm is not None:
            key_states = self.k_norm(key_states)

        value_states = value_raw.view(hidden_shape_k)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # -------------------------------------------------------------
        # 2. Rotary Position Embeddings (RoPE)
        # -------------------------------------------------------------
        if position_embeddings is not None:
            cos, sin = position_embeddings
            from .attention import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        elif hasattr(self, "rotary_emb"):
            start_pos_val = kwargs.get("start_pos", getattr(static_cache, "current_pos", 0) if static_cache else 0)
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(start_pos_val, start_pos_val + q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            try:
                cos, sin = self.rotary_emb(value_states, pos_ids)
            except TypeError:
                cos, sin = self.rotary_emb(value_states, seq_len=start_pos_val + q_len)
            from .attention import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # -------------------------------------------------------------
        # 3. KV Cache update (Static zero-allocation or dynamic fallback)
        # -------------------------------------------------------------
        is_causal = (q_len > 1)
        if static_cache is not None:
            # Multi-level fallback to resolve exact start_pos
            start_pos = kwargs.get("start_pos", None)
            if start_pos is None and "cache_position" in kwargs and kwargs["cache_position"] is not None:
                start_pos = int(kwargs["cache_position"][0].item())
            if start_pos is None:
                start_pos = getattr(static_cache, "current_pos", 0)

            key_states, value_states = static_cache.update(layer_idx, key_states, value_states, start_pos)
            if q_len == 1:
                is_causal = False
                attention_mask = None
        elif past_key_values is not None and hasattr(past_key_values, "update"):
            key_states, value_states = past_key_values.update(key_states, value_states, layer_idx, kwargs)

        if q_len == 1:
            is_causal = False
            attention_mask = None

        # -------------------------------------------------------------
        # 4. Grouped-Query Attention (GQA) KV Expansion
        # -------------------------------------------------------------
        num_groups = num_heads // num_kv_heads
        if num_groups > 1:
            key_states = key_states.repeat_interleave(num_groups, dim=1)
            value_states = value_states.repeat_interleave(num_groups, dim=1)

        # -------------------------------------------------------------
        # 5. Pure FlashAttention SDPA Kernel
        # -------------------------------------------------------------
        attn_output = _fast_sdpa_attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            is_causal=is_causal and attention_mask is None,
            scale=getattr(self, "scaling", None),
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        # Apply Qwen 3.5 sigmoid gate if present
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return (attn_output, None)

    return patched_forward


def patch_qwen(model):
    """
    Patches all full attention layers of Qwen (Qwen2, Qwen2.5, Qwen3, Qwen3.5)
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
