"""
Pre-allocated Static & Smart KV Cache Buffer for High-Throughput Attention.

Optimizations:
- Zero-allocation slice writes eliminating PyTorch tensor concatenations.
- Fast metadata reset (O(1) clear without multi-gigabyte HBM zeroing).
- Auto-clear mechanism triggered on new RL rollout arrivals.
- Prefix KV broadcast / duplication for G-trajectory rollout generation in GRPO.
- Compatible with torch.compile(dynamic=True) and Blackwell FP8/BF16/FP16 tensor cores.
"""

from typing import Optional, List, Tuple, Union
import torch


try:
    from transformers.cache_utils import Cache
except ImportError:
    class Cache:
        pass


class StaticKVCache(Cache):
    """
    High-Throughput Static Key-Value Cache Buffer.
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
    ):
        if issubclass(self.__class__, Cache) and hasattr(Cache, "__init__"):
            try:
                super().__init__(layers=[])
            except Exception:
                self.layers = []
                self.offloading = False
        else:
            self.layers = []
            self.offloading = False

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
