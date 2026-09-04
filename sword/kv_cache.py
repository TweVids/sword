import torch
from typing import Optional, List, Tuple


class StaticKVCache:
    """
    Pre-allocated static Key-Value cache buffer.
    
    Eliminates dynamic memory allocations and tensor concatenations during token decode,
    which are the primary bottlenecks killing TPS on modern GPUs like Blackwell.
    Compatible with torch.compile and CUDA graph capture due to fixed tensor addresses.
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
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pre-allocate contiguous static memory for all layers
        # Shape: [num_layers, max_batch_size, num_kv_heads, max_seq_len, head_dim]
        self.k_cache = torch.zeros(
            (num_layers, max_batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_layers, max_batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )

        # Current length tracking per sequence in batch
        self.seq_lengths = torch.zeros((max_batch_size,), dtype=torch.long, device=device)
        self.current_pos = 0

    def set_pos(self, pos: int):
        """Sets the current write position across all layers."""
        self.current_pos = pos

    def get_pos(self) -> int:
        """Returns the current write position."""
        return self.current_pos

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cache for a single layer without reallocation.
        
        Args:
            layer_idx: index of transformer layer
            k: [batch_size, num_kv_heads, seq_len, head_dim]
            v: [batch_size, num_kv_heads, seq_len, head_dim]
            start_pos: write offset along the sequence dimension (defaults to self.current_pos)
            
        Returns:
            Tuple of views into k_cache and v_cache up to (start_pos + seq_len).
        """
        if start_pos is None:
            start_pos = self.current_pos
        bsz, _, seq_len, _ = k.shape
        end_pos = start_pos + seq_len

        # In-place slice copy (zero allocation)
        self.k_cache[layer_idx, :bsz, :, start_pos:end_pos, :] = k
        self.v_cache[layer_idx, :bsz, :, start_pos:end_pos, :] = v

        # Return view up to the current end position
        return (
            self.k_cache[layer_idx, :bsz, :, :end_pos, :],
            self.v_cache[layer_idx, :bsz, :, :end_pos, :],
        )

    def get_layer_cache(
        self,
        layer_idx: int,
        batch_size: int,
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns sliced view of current cache for given layer."""
        return (
            self.k_cache[layer_idx, :batch_size, :, :total_len, :],
            self.v_cache[layer_idx, :batch_size, :, :total_len, :],
        )

    def reset(self, batch_indices: Optional[List[int]] = None):
        """Reset sequence lengths and clear positions for specified batches."""
        if batch_indices is None:
            self.k_cache.zero_()
            self.v_cache.zero_()
            self.seq_lengths.zero_()
            self.current_pos = 0
        else:
            for b in batch_indices:
                self.k_cache[:, b].zero_()
                self.v_cache[:, b].zero_()
                self.seq_lengths[b] = 0
            self.current_pos = 0
