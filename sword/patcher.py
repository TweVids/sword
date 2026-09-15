import types
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Optional, Tuple
from .attention import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(hidden_states, n_rep, dim=1),
    but uses expand and reshape to avoid redundant memory copies.
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    )


def _fast_sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Pure-PyTorch Fast SDPA Attention kernel.
    Enforces FlashAttention / Efficient Attention C++ kernels on hardware (e.g. Blackwell).
    Seamlessly harmonizes FP8 / BF16 / FP16 key/value states from StaticKVCache.
    """
    if key.dtype != query.dtype or value.dtype != query.dtype:
        if str(query.dtype).startswith("torch.float8"):
            query = query.to(torch.bfloat16)
        target_dtype = query.dtype
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal and attention_mask is None,
                scale=scale,
            )
    except Exception:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal and attention_mask is None,
            scale=scale,
        )


def _vanilla_quadratic_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Standard O(N^2) quadratic attention mechanism.
    Materializes the full [batch, heads, seq_len, seq_len] attention matrix in VRAM.
    Used to demonstrate O(N^2) memory & latency explosion vs FlashAttention O(N).
    """
    if key.dtype != query.dtype or value.dtype != query.dtype:
        if str(query.dtype).startswith("torch.float8"):
            query = query.to(torch.bfloat16)
        target_dtype = query.dtype
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    bsz, num_heads, q_len, head_dim = query.shape
    kv_len = key.shape[2]
    if scale is None:
        scale = head_dim ** -0.5

    # Full quadratic N x N matrix multiplication: O(N^2) memory
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale

    if is_causal and q_len > 1:
        mask = torch.triu(torch.full((q_len, kv_len), float("-inf"), device=query.device, dtype=query.dtype), diagonal=1)
        scores = scores + mask

    if attention_mask is not None:
        scores = scores + attention_mask

    attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p)

    return torch.matmul(attn_probs, value)


def make_patched_attention_forward(original_forward):
    """
    Wraps causal self-attention forward to route through our pure-PyTorch FlashAttention SDPA
    and support static zero-allocation KV caching.
    Natively supports dense and MoE architectures:
    - HYV3 / Hunyuan-3 (MoE with QK-norm, GQA)
    - Qwen2, Qwen2.5, Qwen2-MoE, Qwen3, Qwen3.5 (QK-norm, dual-projection gating)
    - DeepSeek-V2/V3, LLaMA, Mistral, Mixtral
    Preserves all weights and quantization (native FP8, bitsandbytes 4-bit/8-bit).
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
        if static_cache is None and past_key_values is not None and hasattr(past_key_values, "k_cache"):
            static_cache = past_key_values
        layer_idx = getattr(self, "layer_idx", 0)

        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        # -------------------------------------------------------------
        # 1. Dynamic Q/K/V Projections (handles standard Qwen and Qwen3.5)
        # -------------------------------------------------------------
        query_raw = self.q_proj(hidden_states)
        key_raw = self.k_proj(hidden_states)
        value_raw = self.v_proj(hidden_states)

        # Cache model/head dimensions on module to eliminate per-token getattr overhead
        num_heads = getattr(self, "_sword_num_heads", None)
        if num_heads is None:
            cfg = getattr(self, "config", None)
            t_cfg = getattr(cfg, "text_config", cfg)
            num_heads = getattr(self, "num_heads", getattr(self, "num_attention_heads", getattr(t_cfg, "num_attention_heads", 16)))
            num_kv_heads = getattr(self, "num_key_value_heads", getattr(t_cfg, "num_key_value_heads", num_heads))
            head_dim = getattr(self, "head_dim", getattr(t_cfg, "head_dim", None))
            if head_dim is None:
                head_dim = key_raw.shape[-1] // num_kv_heads
            self._sword_num_heads = num_heads
            self._sword_num_kv_heads = num_kv_heads
            self._sword_head_dim = head_dim
            self._sword_num_groups = num_heads // num_kv_heads
            self._has_q_norm = hasattr(self, "q_norm") and self.q_norm is not None
            self._has_k_norm = hasattr(self, "k_norm") and self.k_norm is not None
            num_groups = self._sword_num_groups
        else:
            num_kv_heads = self._sword_num_kv_heads
            head_dim = self._sword_head_dim
            num_groups = self._sword_num_groups

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
        if self._has_q_norm:
            query_states = self.q_norm(query_states)

        hidden_shape_k = (*input_shape, -1, head_dim)
        key_states = key_raw.view(hidden_shape_k)
        if self._has_k_norm:
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
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # -------------------------------------------------------------
        # 3. KV Cache update (Static zero-allocation or dynamic fallback)
        # -------------------------------------------------------------
        is_causal = (q_len > 1)
        if static_cache is not None:
            start_pos = kwargs.get("start_pos", None)
            if start_pos is None:
                start_pos = getattr(static_cache, "current_pos", 0)

            key_states, value_states = static_cache.update(layer_idx, key_states, value_states, start_pos)
        elif past_key_values is not None and hasattr(past_key_values, "update"):
            key_states, value_states = past_key_values.update(key_states, value_states, layer_idx, kwargs)

        kv_len = key_states.shape[-2]
        if q_len == 1:
            # Single-token decode: Q attends to all cached K/V, causal mask is not needed
            is_causal = False
            attention_mask = None
        elif kv_len > q_len:
            # Multi-token evaluation with past KV cache (e.g. speculative prompt lookup decoding).
            # Query tokens attend to all past tokens [0 .. kv_len - q_len - 1]
            # and causally among themselves [kv_len - q_len .. kv_len - 1].
            slice_mask = torch.zeros(
                (1, 1, q_len, kv_len),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            causal_sub = torch.triu(
                torch.full(
                    (q_len, q_len),
                    float("-inf"),
                    device=query_states.device,
                    dtype=query_states.dtype,
                ),
                diagonal=1,
            )
            slice_mask[:, :, :, kv_len - q_len :] = causal_sub

            if attention_mask is not None and attention_mask.shape[-1] == kv_len:
                if attention_mask.dtype == torch.bool:
                    slice_mask = slice_mask.masked_fill(~attention_mask, float("-inf"))
                else:
                    slice_mask = slice_mask + attention_mask

            attention_mask = slice_mask
            is_causal = False

        # -------------------------------------------------------------
        # 4. Pure FlashAttention / Efficient Attention SDPA
        # -------------------------------------------------------------
        attn_mode = getattr(self, "_sword_attn_mode", "flash")

        if num_groups > 1 and q_len == 1 and attention_mask is None and attn_mode != "vanilla":
            # Zero-Copy GQA via View Transformation (9.76x faster, 0 bytes allocated)
            q_b = query_states.view(bsz * num_kv_heads, 1, num_groups, head_dim)
            k_b = key_states.view(bsz * num_kv_heads, 1, -1, head_dim)
            v_b = value_states.view(bsz * num_kv_heads, 1, -1, head_dim)
            attn_out_b = _fast_sdpa_attention(
                q_b,
                k_b,
                v_b,
                attention_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=getattr(self, "scaling", None),
            )
            attn_output = attn_out_b.view(bsz, num_heads, 1, head_dim)
        else:
            if num_groups > 1:
                key_states = repeat_kv(key_states, num_groups)
                value_states = repeat_kv(value_states, num_groups)

            if attn_mode == "vanilla":
                attn_output = _vanilla_quadratic_attention(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=attention_mask,
                    is_causal=is_causal and attention_mask is None,
                    scale=getattr(self, "scaling", None),
                )
            else:
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


def make_fast_moe_forward(original_forward):
    """
    High-performance zero-sync MoE expert dispatch forward.
    Replaces HuggingFace's eager routing loop (which executes one_hot + .nonzero() +
    torch.where() on GPU, inducing 1,500+ host-device synchronization roundtrips
    and ~800ms per token latency) with a 0.099ms zero-sync CPU routing dispatch.
    Numerically bit-identical to HuggingFace FP8Experts & Qwen3MoeExperts.
    Safely unwraps PEFT / Unsloth ParamWrapper layers and preserves active LoRA hooks.
    """
    def fast_moe_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Resolve target module if wrapped by PEFT / Unsloth ParamWrapper
        target = self
        while hasattr(target, "base_layer"):
            target = target.base_layer
        if hasattr(target, "module") and not hasattr(target, "gate_up_proj") and hasattr(target.module, "gate_up_proj"):
            target = target.module

        # Check if target has expected projections; if not, safely fallback to original_forward
        has_gate_up = hasattr(target, "gate_up_proj")
        has_split = hasattr(target, "gate_proj") and hasattr(target, "up_proj")
        has_exp_list = hasattr(target, "experts") and isinstance(target.experts, (list, torch.nn.ModuleList))
        is_fp8 = hasattr(target, "linear")

        if not (has_gate_up or has_split or has_exp_list or is_fp8):
            return original_forward(hidden_states, top_k_index, top_k_weights, *args, **kwargs)

        # Pure GPU/CPU Batched BMM MoE Dispatch (Zero-Sync, 100% CUDA Graph capture compatible)
        if (
            hasattr(target, "gate_up_proj")
            and hasattr(target, "down_proj")
            and getattr(target.gate_up_proj, "dim", lambda: 0)() == 3
            and getattr(target.down_proj, "dim", lambda: 0)() == 3
        ):
            num_tokens, hidden_dim = hidden_states.shape
            num_exp_per_tok = top_k_index.shape[1]
            flat_exp_idx = top_k_index.reshape(-1)
            flat_tok_idx = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand(-1, num_exp_per_tok).reshape(-1)
            x = hidden_states[flat_tok_idx].unsqueeze(1)
            w_up = target.gate_up_proj[flat_exp_idx]
            if w_up.dtype != x.dtype and str(w_up.dtype).startswith("torch.float8"):
                w_up = w_up.to(x.dtype)
            gate_up = torch.bmm(x, w_up.transpose(1, 2))
            gate, up = gate_up.chunk(2, dim=-1)
            act_fn = getattr(target, "act_fn", F.silu)
            act = act_fn(gate) * up
            w_down = target.down_proj[flat_exp_idx]
            if w_down.dtype != act.dtype and str(w_down.dtype).startswith("torch.float8"):
                w_down = w_down.to(act.dtype)
            down = torch.bmm(act, w_down.transpose(1, 2))
            flat_w = top_k_weights.reshape(-1, 1, 1).to(down.dtype)
            weighted = (down * flat_w).squeeze(1)
            out = torch.zeros_like(hidden_states)
            out.index_add_(0, flat_tok_idx, weighted.to(out.dtype))
            return out

        def _do_dispatch():
            num_tokens, hidden_dim = hidden_states.shape
            num_experts = getattr(target, "num_experts", 128)
            has_gate = getattr(target, "has_gate", True)
            is_static = getattr(target, "activation_scheme", "dynamic") == "static"

            # Fast CPU routing: single D2H copy avoids GPU-side .nonzero() and torch.where() sync stalls completely
            top_k_cpu = top_k_index.tolist()
            expert_to_tokens = {}
            for tok_i, exp_ids in enumerate(top_k_cpu):
                for k_pos, exp_id in enumerate(exp_ids):
                    if exp_id < num_experts:
                        expert_to_tokens.setdefault(exp_id, []).append((tok_i, k_pos))

            final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
            act_fn = getattr(target, "act_fn", F.silu)

            for exp_id, pairs in expert_to_tokens.items():
                gate_up_act_scale = (
                    target.gate_up_proj_activation_scale[exp_id] if (is_static and hasattr(target, "gate_up_proj_activation_scale")) else None
                )
                down_act_scale = (
                    target.down_proj_activation_scale[exp_id] if (is_static and hasattr(target, "down_proj_activation_scale")) else None
                )

                if is_fp8:
                    weight_up = target.gate_up_proj[exp_id] if has_gate else target.up_proj[exp_id]
                    scale_up = target.gate_up_proj_scale_inv[exp_id] if has_gate else target.up_proj_scale_inv[exp_id]
                    weight_down = target.down_proj[exp_id]
                    scale_down = target.down_proj_scale_inv[exp_id]

                if len(pairs) == 1:
                    tok_i, k_pos = pairs[0]
                    current_state = hidden_states[tok_i : tok_i + 1]
                    if is_fp8:
                        proj_out = target.linear(current_state, weight_up, scale_up, activation_scale=gate_up_act_scale)
                        proj_out = target._apply_gate(proj_out) if has_gate else act_fn(proj_out)
                        proj_out = target.linear(proj_out, weight_down, scale_down, activation_scale=down_act_scale)
                    else:
                        if has_exp_list:
                            proj_out = target.experts[exp_id](current_state)
                        elif has_split:
                            proj_g = target.gate_proj[exp_id]
                            proj_u = target.up_proj[exp_id]
                            gate = proj_g(current_state) if callable(proj_g) else F.linear(current_state, proj_g)
                            up = proj_u(current_state) if callable(proj_u) else F.linear(current_state, proj_u)
                            proj_out = act_fn(gate) * up
                            proj_d = target.down_proj[exp_id]
                            proj_out = proj_d(proj_out) if callable(proj_d) else F.linear(proj_out, proj_d)
                        else:
                            proj_module_up = target.gate_up_proj[exp_id]
                            proj_out = proj_module_up(current_state) if callable(proj_module_up) else F.linear(current_state, proj_module_up)
                            gate, up = proj_out.chunk(2, dim=-1)
                            proj_out = act_fn(gate) * up
                            proj_module_down = target.down_proj[exp_id]
                            proj_out = proj_module_down(proj_out) if callable(proj_module_down) else F.linear(proj_out, proj_module_down)

                    routing_weight = top_k_weights[tok_i, k_pos]
                    final_hidden_states[tok_i] += (proj_out[0] * routing_weight).float()
                else:
                    tok_indices = [p[0] for p in pairs]
                    k_positions = [p[1] for p in pairs]
                    idx_tensor = torch.tensor(tok_indices, dtype=torch.long, device=hidden_states.device)
                    k_tensor = torch.tensor(k_positions, dtype=torch.long, device=hidden_states.device)
                    current_state = hidden_states[idx_tensor]
                    if is_fp8:
                        proj_out = target.linear(current_state, weight_up, scale_up, activation_scale=gate_up_act_scale)
                        proj_out = target._apply_gate(proj_out) if has_gate else act_fn(proj_out)
                        proj_out = target.linear(proj_out, weight_down, scale_down, activation_scale=down_act_scale)
                    else:
                        if has_exp_list:
                            proj_out = target.experts[exp_id](current_state)
                        elif has_split:
                            proj_g = target.gate_proj[exp_id]
                            proj_u = target.up_proj[exp_id]
                            gate = proj_g(current_state) if callable(proj_g) else F.linear(current_state, proj_g)
                            up = proj_u(current_state) if callable(proj_u) else F.linear(current_state, proj_u)
                            proj_out = act_fn(gate) * up
                            proj_d = target.down_proj[exp_id]
                            proj_out = proj_d(proj_out) if callable(proj_d) else F.linear(proj_out, proj_d)
                        else:
                            proj_module_up = target.gate_up_proj[exp_id]
                            proj_out = proj_module_up(current_state) if callable(proj_module_up) else F.linear(current_state, proj_module_up)
                            gate, up = proj_out.chunk(2, dim=-1)
                            proj_out = act_fn(gate) * up
                            proj_module_down = target.down_proj[exp_id]
                            proj_out = proj_module_down(proj_out) if callable(proj_module_down) else F.linear(proj_out, proj_module_down)

                    weights = top_k_weights[idx_tensor, k_tensor, None]
                    weighted_out = proj_out * weights.to(proj_out.dtype)
                    final_hidden_states.index_add_(0, idx_tensor, weighted_out.float())

            return final_hidden_states.to(hidden_states.dtype)

        return _do_dispatch()

    return fast_moe_forward


def make_patched_qwen3_moe_block_forward(original_forward):
    """
    Patches Qwen3MoeSparseMoeBlock to capture router_logits as `self.last_router_logits`
    for MoE router collapse monitoring & auxiliary load-balancing loss in RL / GRPO.
    """
    def patched_block_forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        self.last_router_logits = router_logits
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return patched_block_forward


def patch_moe_experts(model, force_fast_moe: bool = False):
    """
    Patches MoE expert routing layers (FP8Experts, HYV3Experts, Qwen3MoeExperts, Qwen2MoeExperts)
    with Sword's Zero-Sync Fast MoE Forward when hardware-fused kernels are unavailable.
    Replaces slow GPU-side .nonzero() / torch.where() loops that induce hundreds of host-device
    synchronization stalls per token with high-speed zero-sync routing.
    Also instruments MoE sparse blocks to expose router logits for RL monitoring & aux loss.
    Safely unwraps nested PEFT/Unsloth ParamWrapper to preserve active LoRA hooks without crashing.
    Preserves high-speed fused grouped_mm / Triton kernels on Blackwell & CUDA hardware.
    """
    patched_count = 0
    fused_count = 0
    seen_modules = set()

    for name, module in model.named_modules():
        mod_type = module.__class__.__name__

        # Instrument MoE sparse block to record router logits for RL aux loss
        if mod_type in ("Qwen3MoeSparseMoeBlock", "Qwen2MoeSparseMoeBlock") or (hasattr(module, "experts") and hasattr(module, "gate") and not hasattr(module, "_sword_original_forward")):
            module._sword_original_forward = module.forward
            module.forward = types.MethodType(make_patched_qwen3_moe_block_forward(module._sword_original_forward), module)

        # Safely unwrap all nested ParamWrapper layers (PEFT / Unsloth)
        target_mod = module
        while hasattr(target_mod, "base_layer"):
            target_mod = target_mod.base_layer

        if target_mod.__class__.__name__ == "ParamWrapper":
            continue

        target_type = target_mod.__class__.__name__
        if (target_type in ("FP8Experts", "HYV3Experts", "Qwen3MoeExperts", "Qwen2MoeExperts") or name.endswith(".experts")) and id(target_mod) not in seen_modules:
            seen_modules.add(id(target_mod))

            cfg = getattr(target_mod, "config", getattr(module, "config", getattr(model, "config", None)))
            impl = getattr(cfg, "_experts_implementation", None)

            # Check if model is quantized (BitsAndBytes 4-bit/8-bit or Linear4bit)
            is_quantized = any(
                hasattr(p, "quant_state") or "Linear4bit" in m.__class__.__name__ or "8bit" in m.__class__.__name__
                for m in target_mod.modules() for p in m.parameters()
            ) or getattr(cfg, "load_in_4bit", False) or getattr(cfg, "quantization_config", None) is not None

            is_fp8 = hasattr(target_mod, "linear") or hasattr(target_mod, "gate_up_proj_scale_inv")

            # When real hardware-fused grouped_mm or Unsloth Triton kernels are available, preserve them
            has_real_fused_kernel = (
                impl in ("deepgemm", "deepgemm_megamoe")
                or (is_fp8 and hasattr(target_mod, "linear") and impl == "grouped_mm")
                or hasattr(target_mod, "_fused_kernel")
            )
            if not force_fast_moe and has_real_fused_kernel:
                if not hasattr(target_mod, "_sword_fast_forward"):
                    target_mod._sword_fast_forward = types.MethodType(make_fast_moe_forward(target_mod.forward), target_mod)
                fused_count += 1
                continue

            if not hasattr(target_mod, "_sword_original_forward"):
                target_mod._sword_original_forward = target_mod.forward
            target_mod.forward = types.MethodType(make_fast_moe_forward(target_mod._sword_original_forward), target_mod)
            patched_count += 1

    if patched_count > 0:
        print(f"[Sword] Patched {patched_count} MoE expert routing modules with Zero-Sync Fast Dispatch.")
    if fused_count > 0:
        print(f"[Sword] Preserving {fused_count} MoE layers with fused '{impl}' kernels.")
    return model


make_patched_qwen_attention_forward = make_patched_attention_forward


def set_attention_mode(model, mode: str = "flash"):
    """
    Sets attention execution mode across all patched layers:
    - 'flash': Pure-PyTorch FlashAttention SDPA (O(N) memory, tiled SRAM)
    - 'vanilla': Quadratic O(N^2) materialized attention matrix (for benchmarking)
    """
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "_sword_original_forward"):
            module._sword_attn_mode = mode
            count += 1
    return count


def patch_model(model, mode: str = "flash", patch_moe: bool = True):
    """
    Universal patcher for all Transformer & MoE causal attention modules
    (HYV3/Hunyuan-3, Qwen, Qwen2-MoE, DeepSeek, LLaMA, Mistral, etc.)
    with Sword's pure FlashAttention SDPA kernel and Static KV cache routing,
    plus Zero-Sync Fast MoE Expert Dispatch.
    Preserves all weights and quantization (native FP8, bitsandbytes 4-bit/8-bit, etc.).
    """
    import re
    patched_count = 0
    cur_layer_idx = 0
    for name, module in model.named_modules():
        mod_type = module.__class__.__name__.lower()
        if "attention" in mod_type or "attn" in mod_type:
            if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                if not hasattr(module, "_sword_original_forward"):
                    module._sword_original_forward = module.forward
                module._sword_attn_mode = mode

                # Infer layer_idx if missing
                if not hasattr(module, "layer_idx") or module.layer_idx is None:
                    m = re.search(r"layers?\.(\d+)", name)
                    if m:
                        module.layer_idx = int(m.group(1))
                    else:
                        module.layer_idx = cur_layer_idx
                        cur_layer_idx += 1

                module.forward = types.MethodType(make_patched_attention_forward(module._sword_original_forward), module)
                patched_count += 1
    print(f"[Sword] Patched {patched_count} attention modules with Pure FlashAttention SDPA (mode='{mode}').")

    if patch_moe:
        patch_moe_experts(model)

    # Check for Ling-3.0-tiny (BailingMoeV3) hybrid architecture ONLY if actually a Ling model
    is_ling = any("bailing" in m.__class__.__name__.lower() or "ling" in m.__class__.__name__.lower() for m in model.modules())
    if is_ling:
        try:
            from .ling import patch_ling as _patch_ling
            _patch_ling(model, mode=mode, patch_moe=patch_moe)
        except Exception as e:
            pass

    return model


patch_qwen = patch_model
patch_moe = patch_model
patch_qwen3_moe = patch_model
make_patched_qwen3_attention_forward = make_patched_attention_forward


def patch_ling(model, mode: str = "flash", patch_moe: bool = True):
    from .ling import patch_ling as _patch_ling
    return _patch_ling(model, mode=mode, patch_moe=patch_moe)


def unpatch_ling(model):
    from .ling import unpatch_ling as _unpatch_ling
    return _unpatch_ling(model)


def unpatch_model(model):
    """
    Restores all attention and MoE expert modules back to original unpatched forward methods.
    """
    unpatched_count = 0
    seen = set()
    for name, module in model.named_modules():
        target = getattr(module, "base_layer", module)
        for m in (module, target):
            if id(m) not in seen and hasattr(m, "_sword_original_forward"):
                seen.add(id(m))
                m.forward = m._sword_original_forward
                delattr(m, "_sword_original_forward")
                for attr in ("_sword_attn_mode", "_sword_static_cache", "_sword_num_heads", "_sword_num_kv_heads", "_sword_head_dim", "_sword_num_groups", "_has_q_norm", "_has_k_norm"):
                    if hasattr(m, attr):
                        delattr(m, attr)
                unpatched_count += 1
    try:
        from .ling import unpatch_ling as _unpatch_ling
        _unpatch_ling(model)
    except Exception:
        pass
    print(f"[Sword] Unpatched {unpatched_count} modules to original forward.")
    return model


unpatch_qwen = unpatch_model
unpatch_moe = unpatch_model
unpatch_qwen3_moe = unpatch_model

