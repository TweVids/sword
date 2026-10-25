"""Pure PyTorch replacement for flash-attn rotary, including packed sequences."""


def apply_rotary_fallback(x, cos, sin, seqlen_offsets=0, cu_seqlens=None,
                         max_seqlen=None, interleaved=False, inplace=False, conjugate=False):
    import torch

    if x.ndim not in (3, 4) or cos.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("rotary expects [B,S,H,D] or [T,H,D] and matching [S,D/2] tables")
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError("rotary dimension exceeds head dimension")
    offsets = torch.as_tensor(seqlen_offsets, device=x.device, dtype=torch.long)
    if x.ndim == 4:
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens requires packed [T,H,D] input")
        batch, seq_len = x.shape[:2]
        if offsets.ndim > 1 or offsets.numel() not in (1, batch):
            raise ValueError("one sequence offset is required per batch row")
        positions = torch.arange(seq_len, device=x.device)[None, :] + offsets.reshape(-1, 1)
        positions = positions.expand(batch, -1)
    elif cu_seqlens is not None:
        starts = torch.as_tensor(cu_seqlens, device=x.device, dtype=torch.long)
        if starts.ndim != 1 or starts.numel() < 2:
            raise ValueError("cu_seqlens must delimit packed sequences")
        lengths = starts[1:] - starts[:-1]
        if starts[0].item() != 0 or starts[-1].item() != x.shape[0] or (lengths < 0).any():
            raise ValueError("invalid packed sequence boundaries")
        if offsets.ndim > 1 or offsets.numel() not in (1, lengths.numel()):
            raise ValueError("one offset is required per packed sequence")
        sequence_ids = torch.repeat_interleave(torch.arange(lengths.numel(), device=x.device), lengths)
        offsets = offsets.reshape(-1).expand(lengths.numel())
        positions = torch.arange(x.shape[0], device=x.device) - starts[sequence_ids] + offsets[sequence_ids]
    else:
        if offsets.numel() != 1:
            raise ValueError("unbatched rotary requires a scalar offset")
        positions = torch.arange(x.shape[0], device=x.device) + offsets.reshape(())
    if (positions < 0).any() or (positions >= cos.shape[0]).any():
        raise ValueError("rotary positions exceed cosine/sine table bounds")
    c = cos.to(device=x.device)[positions].unsqueeze(-2).float()
    s = sin.to(device=x.device)[positions].unsqueeze(-2).float()
    if conjugate:
        s = -s
    rotating = x[..., :rotary_dim].float()
    if interleaved:
        x1, x2 = rotating[..., 0::2], rotating[..., 1::2]
        rotated = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    else:
        x1, x2 = rotating.chunk(2, dim=-1)
        rotated = torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    out = torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), dim=-1)
    if inplace:
        x.copy_(out)
        return x
    return out


def homogeneous_head_dim(config):
    """Recover a global head dimension only when every layer agrees."""
    try:
        return getattr(config, "head_dim", 0)
    except AttributeError:
        pass
    # Transformers' heterogeneity exception is a ValueError in some releases.
    except ValueError:
        pass
    except RuntimeError as exc:
        if type(exc).__name__ != "AmbiguousGlobalPerLayerAttributeError":
            raise
    layers = getattr(config, "per_layer_config", None)
    if not layers:
        raise ValueError("Cannot recover head_dim without per-layer configuration")
    dimensions = {layer.get("head_dim", 0) if isinstance(layer, dict) else getattr(layer, "head_dim", 0)
                  for layer in layers}
    if len(dimensions) != 1:
        raise ValueError("Heterogeneous head dimensions require per-layer vLLM support")
    return dimensions.pop()
