"""
Pre-allocated Static & Smart KV Cache Buffer for High-Throughput Attention.

Optimizations:
- Zero-allocation slice writes eliminating PyTorch tensor concatenations.
- Fast metadata reset (O(1) clear without multi-gigabyte HBM zeroing).
- Auto-clear mechanism triggered on new RL rollout arrivals.
- Prefix KV broadcast / duplication for G-trajectory rollout generation in GRPO.
- Compatible with torch.compile(dynamic=True) and Blackwell FP8/BF16/FP16 tensor cores.
"""

from typing import Optional, List, Tuple, Union, Any
import torch


try:
    from transformers.cache_utils import Cache, CacheLayerMixin, LinearAttentionCacheLayerMixin
except ImportError:
    class Cache:
        pass
    class CacheLayerMixin:
        pass
    class LinearAttentionCacheLayerMixin:
        pass


class StaticLinearAttentionLayer(LinearAttentionCacheLayerMixin):
    """
    High-Throughput Static Linear Attention Cache Buffer (Qwen 3.5 GatedDeltaNet).
    Pre-allocates static 3D conv_states and 4D recurrent_states with pinned static addresses.
    """
    def __init__(
        self,
        max_batch_size: int,
        conv_dim: int,
        conv_k: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.conv_states = torch.zeros((max_batch_size, conv_dim, conv_k), dtype=dtype, device=device)
        self.recurrent_states = torch.zeros((max_batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=dtype, device=device)
        self.has_previous_state = False
        self.is_conv_states_initialized = True
        self.is_recurrent_states_initialized = True
        self.conv_kernel_size = conv_k
        self.dtype = dtype
        self.device = device
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "mark_static_address"):
            try:
                torch._dynamo.mark_static_address(self.conv_states)
                torch._dynamo.mark_static_address(self.recurrent_states)
            except Exception:
                pass

    def lazy_initialization(self, conv_states: Optional[torch.Tensor] = None, recurrent_states: Optional[torch.Tensor] = None) -> None:
        pass

    def update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.has_previous_state:
            self.conv_states.copy_(conv_states)
            self.has_previous_state = True
        else:
            num_new_tokens = conv_states.shape[-1]
            if num_new_tokens >= self.conv_kernel_size:
                self.conv_states.copy_(conv_states[..., -self.conv_kernel_size :])
            else:
                self.conv_states.copy_(torch.roll(self.conv_states, shifts=-num_new_tokens, dims=-1))
                self.conv_states[..., -num_new_tokens:] = conv_states
        return self.conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
        self.recurrent_states.copy_(recurrent_states)
        return self.recurrent_states

    def reset(self):
        self.conv_states.zero_()
        self.recurrent_states.zero_()
        self.has_previous_state = False


class StaticAttentionLayer(CacheLayerMixin):
    """
    Lightweight layer view wrapping full attention static KV cache.
    """
    def __init__(self, parent_cache: "StaticKVCache", layer_idx: int):
        super().__init__()
        self.parent = parent_cache
        self.layer_idx = layer_idx
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.parent.update(key_states, value_states, self.layer_idx, **kwargs)

    def get_seq_length(self) -> int:
        return self.parent.current_pos

    def get_max_length(self) -> Optional[int]:
        return self.parent.max_seq_len

    def get_max_cache_shape(self) -> int:
        return self.parent.max_seq_len

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return query_length + self.parent.current_pos, self.parent.current_pos


class StaticKVCache(Cache):
    """
    High-Throughput Static Key-Value Cache Buffer for Dense and Hybrid models.
    Supports pure Full Attention (Qwen 2/2.5/3, LLaMA) and Hybrid architectures (Qwen 3.5 GatedDeltaNet + Attention).
    Inherits from transformers Cache for full compatibility with native HF models & PeftModel.
    Eliminates dynamic memory allocations, garbage collection stalls,
    and tensor concatenations during token decode.
    """
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        auto_clear: bool = True,
        config: Optional[Any] = None,
        layer_types: Optional[List[str]] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self._max_batch_size = max_batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.auto_clear = auto_clear

        # Pre-allocate contiguous static memory per layer
        # Shape: [max_batch_size, num_kv_heads, max_seq_len, head_dim]
        self.k_cache: List[torch.Tensor] = [
            torch.zeros(
                (max_batch_size, num_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=self.device,
            )
            for _ in range(num_layers)
        ]
        self.v_cache: List[torch.Tensor] = [
            torch.zeros(
                (max_batch_size, num_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=self.device,
            )
            for _ in range(num_layers)
        ]

        # Inspect hybrid layer types (e.g. Qwen 3.5 GatedDeltaNet)
        if layer_types is None and config is not None:
            layer_types = getattr(config, "layer_types", None)
            if layer_types is None and hasattr(config, "text_config"):
                layer_types = getattr(config.text_config, "layer_types", None)

        built_layers = []
        for i in range(num_layers):
            l_type = layer_types[i] if layer_types and i < len(layer_types) else "full_attention"
            if l_type == "linear_attention" and config is not None:
                t_cfg = getattr(config, "text_config", config)
                lin_k_heads = getattr(t_cfg, "linear_num_key_heads", 2)
                lin_v_heads = getattr(t_cfg, "linear_num_value_heads", 2)
                lin_k_dim = getattr(t_cfg, "linear_key_head_dim", 32)
                lin_v_dim = getattr(t_cfg, "linear_value_head_dim", 32)
                conv_k = getattr(t_cfg, "linear_conv_kernel_dim", 4)
                conv_dim = lin_k_dim * lin_k_heads * 2 + lin_v_dim * lin_v_heads
                built_layers.append(
                    StaticLinearAttentionLayer(
                        max_batch_size=max_batch_size,
                        conv_dim=conv_dim,
                        conv_k=conv_k,
                        num_v_heads=lin_v_heads,
                        head_k_dim=lin_k_dim,
                        head_v_dim=lin_v_dim,
                        dtype=dtype,
                        device=self.device,
                    )
                )
            else:
                built_layers.append(StaticAttentionLayer(self, i))

        if issubclass(self.__class__, Cache) and hasattr(Cache, "__init__"):
            try:
                super().__init__(layers=built_layers)
            except Exception:
                self.layers = built_layers
                self.offloading = False
        else:
            self.layers = built_layers
            self.offloading = False

        # Tracking state
        self.seq_lengths = torch.zeros((max_batch_size,), dtype=torch.long, device=self.device)
        self.current_pos = 0
        self.rollout_id = 0
        self._active_batch_size = 0

    def set_pos(self, pos: int):
        """Sets current write position across all layers."""
        self.current_pos = pos

    def get_pos(self) -> int:
        """Returns current write position."""
        return self.current_pos

    def new_rollout(self, rollout_id: Optional[int] = None, batch_size: Optional[int] = None):
        """
        Signals the arrival of a new rollout batch.
        Instantly clears metadata and positions in O(1) time without
        wasting GPU memory bandwidth re-zeroing multi-gigabyte tensors.
        """
        if rollout_id is not None:
            self.rollout_id = rollout_id
        else:
            self.rollout_id += 1

        if batch_size is not None:
            self._active_batch_size = batch_size

        self.fast_reset()

    def fast_reset(self, batch_indices: Optional[Union[List[int], range]] = None):
        """
        O(1) metadata reset.
        Subsequent slice updates naturally overwrite buffer contents up to end_pos,
        so zeroing gigabytes of memory is completely avoided.
        """
        if batch_indices is None:
            self.seq_lengths.zero_()
            self.current_pos = 0
            for l in self.layers:
                if hasattr(l, "has_previous_state"):
                    l.has_previous_state = False
        else:
            for b in batch_indices:
                if b < self.max_batch_size:
                    self.seq_lengths[b] = 0
            self.current_pos = 0

    def reset(self, batch_indices: Optional[List[int]] = None, fast: bool = True):
        """
        Resets sequence lengths and positions.
        By default fast=True skips expensive physical memory zeroing.
        """
        if fast:
            self.fast_reset(batch_indices)
            return

        # Hard reset: physically zero tensors
        if batch_indices is None:
            for k, v in zip(self.k_cache, self.v_cache):
                k.zero_()
                v.zero_()
            for l in self.layers:
                if hasattr(l, "reset"):
                    l.reset()
                elif hasattr(l, "has_previous_state"):
                    l.has_previous_state = False
            self.seq_lengths.zero_()
            self.current_pos = 0
        else:
            for b in batch_indices:
                if b < self._max_batch_size:
                    for k, v in zip(self.k_cache, self.v_cache):
                        k[b].zero_()
                        v[b].zero_()
                    self.seq_lengths[b] = 0
            self.current_pos = 0

    def has_previous_state(self, layer_idx: Optional[int] = None) -> bool:
        """
        HF Cache interface for hybrid architectures (e.g. Qwen 3.5).
        Returns whether the specified layer (or any linear attention layer) has previous state.
        """
        if layer_idx is not None:
            if layer_idx < len(self.layers) and hasattr(self.layers[layer_idx], "has_previous_state"):
                return bool(self.layers[layer_idx].has_previous_state)
            return self.current_pos > 0
        return any(getattr(l, "has_previous_state", False) for l in self.layers)

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        """Updates linear attention conv states."""
        if layer_idx < len(self.layers) and hasattr(self.layers[layer_idx], "update_conv_state"):
            return self.layers[layer_idx].update_conv_state(conv_states, **kwargs)
        return conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        """Updates linear attention recurrent states."""
        if layer_idx < len(self.layers) and hasattr(self.layers[layer_idx], "update_recurrent_state"):
            return self.layers[layer_idx].update_recurrent_state(recurrent_states, **kwargs)
        return recurrent_states

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @max_batch_size.setter
    def max_batch_size(self, value: int):
        self._max_batch_size = value

    @property
    def batch_size(self) -> int:
        return self._active_batch_size or self._max_batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._active_batch_size = value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the current sequence length for the cache."""
        return self.current_pos

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length supported by the cache."""
        return self.max_seq_len

    def get_usable_length(self, new_seq_len: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new tokens, returns the usable length of the cache."""
        return self.current_pos

    def get_mask_sizes(self, query_length: int, layer_idx: Optional[int] = None) -> Tuple[int, int]:
        """Returns (kv_length, kv_offset) for causal masking."""
        return query_length + self.current_pos, self.current_pos

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
        start_pos: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HF Cache update interface compatibility.
        Supports both positional and keyword argument orders.
        """
        # Allow either (layer_idx, k, v) or (k, v, layer_idx)
        if isinstance(key_states, int):
            # Old signature: update(layer_idx, k, v, start_pos=...)
            l_idx = key_states
            k = value_states
            v = layer_idx
            if start_pos is None:
                start_pos = cache_kwargs if isinstance(cache_kwargs, int) else self.current_pos
        else:
            l_idx = layer_idx
            k = key_states
            v = value_states
            if start_pos is None and cache_kwargs and isinstance(cache_kwargs, dict):
                start_pos = cache_kwargs.get("start_pos", None)
            if start_pos is None:
                start_pos = kwargs.get("start_pos", self.current_pos)

        bsz, _, seq_len, _ = k.shape
        end_pos = start_pos + seq_len

        # Bounds safety clamping
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"[Sword-KV] Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}. "
                "Increase max_seq_len when initializing cache or server."
            )

        # In-place slice copy (zero memory allocation)
        self.k_cache[l_idx][:bsz, :, start_pos:end_pos, :] = k
        self.v_cache[l_idx][:bsz, :, start_pos:end_pos, :] = v

        # Return view up to current end position
        return (
            self.k_cache[l_idx][:bsz, :, :end_pos, :],
            self.v_cache[l_idx][:bsz, :, :end_pos, :],
        )

    def duplicate_prefix_for_rollouts(
        self,
        num_prompts: int,
        group_size: int,
        prompt_lens: List[int],
    ):
        """
        Multi-Trajectory Prefix KV Broadcast (GRPO / PPO optimization).
        When generating G rollouts per prompt:
          Given prefilled prompt KV at slot (i * G), broadcasts the prompt's
          KV states to all G-1 sibling rollout slots [i*G + 1 ... i*G + G - 1]
          across all layers using in-place strided slice copying.
        
        Eliminates redundant prefill computation across rollout trajectories!
        """
        for i in range(num_prompts):
            src_slot = i * group_size
            dst_start = src_slot + 1
            dst_end = (i + 1) * group_size
            if dst_start >= dst_end:
                continue

            p_len = prompt_lens[i]
            for layer_idx in range(self.num_layers):
                if layer_idx < len(self.layers) and hasattr(self.layers[layer_idx], "conv_states"):
                    l_mod = self.layers[layer_idx]
                    l_mod.conv_states[dst_start:dst_end].copy_(l_mod.conv_states[src_slot : src_slot + 1])
                    l_mod.recurrent_states[dst_start:dst_end].copy_(l_mod.recurrent_states[src_slot : src_slot + 1])
                    l_mod.has_previous_state = True
                else:
                    src_k = self.k_cache[layer_idx][src_slot : src_slot + 1, :, :p_len, :]
                    src_v = self.v_cache[layer_idx][src_slot : src_slot + 1, :, :p_len, :]
                    self.k_cache[layer_idx][dst_start:dst_end, :, :p_len, :].copy_(src_k)
                    self.v_cache[layer_idx][dst_start:dst_end, :, :p_len, :].copy_(src_v)

    def get_layer_cache(
        self,
        layer_idx: int,
        batch_size: int,
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns sliced view of current cache for given layer."""
        return (
            self.k_cache[layer_idx][:batch_size, :, :total_len, :],
            self.v_cache[layer_idx][:batch_size, :, :total_len, :],
        )


# Smart alias
SmartKVCache = StaticKVCache
