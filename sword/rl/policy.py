"""Token statistics with checkpointed vocabulary projection for GRPO updates."""

from typing import Optional, Tuple
import copy
import gc

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def safe_optimizer_step(optimizer, parameters, after_step=None):
    """Restore weights and optimizer state if an update fails partway through.

    Snapshots live on CPU so an OOM retry cannot train on a partially updated
    policy. This costs host memory proportional to trainable weights and state.
    """
    def cpu_copy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu", copy=True)
        if isinstance(value, dict):
            return {key: cpu_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_copy(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_copy(item) for item in value)
        return copy.deepcopy(value)

    parameters = list(parameters)
    weight_snapshot = [param.detach().to("cpu", copy=True) for param in parameters]
    state_snapshot = cpu_copy(optimizer.state_dict())
    try:
        result = optimizer.step()
        return after_step() if after_step is not None else result
    except Exception as exc:
        # The failed optimizer frame can hold references to temporary CUDA tensors.
        exc.__traceback__ = None
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                for param, saved in zip(parameters, weight_snapshot):
                    param.copy_(saved)
            optimizer.load_state_dict(state_snapshot)
        except Exception as restore_error:
            raise RuntimeError("Optimizer rollback failed; automatic retry is unsafe") from restore_error
        raise exc


def token_statistics(
    model, input_ids, attention_mask, chunk_size: int,
    temperature: float = 1.0, with_entropy: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return next-token logprobs, optionally entropy, without saving chunk logits.

    Standard decoder-only HF models have an unmodified linear output head. For
    these models run the decoder directly so full-sequence logits and every
    layer's hidden states are never materialized. Other architectures retain
    their own causal-LM forward (which may transform logits).
    """
    if chunk_size <= 0 or temperature <= 0:
        raise ValueError("chunk_size and temperature must be positive")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = getattr(base, "config", None)
    model_type = getattr(config, "model_type", None)
    linear_head_types = {"qwen2", "qwen3", "qwen3_moe", "qwen3_5_text", "llama", "mistral"}
    decoder = getattr(base, "model", None)
    head = base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
    direct_decoder = (
        model_type in linear_head_types and decoder is not None and head is not None
        and not getattr(config, "final_logit_softcapping", None)
        and not getattr(config, "logit_scale", None)
    )
    if direct_decoder:
        outputs = decoder(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, output_hidden_states=False, return_dict=True)
        features = outputs.last_hidden_state[:, :-1, :]
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True)
        features = outputs.logits[:, :-1, :]

    targets = input_ids[:, 1:]

    def project_chunk(features_chunk, targets_chunk):
        logits = head(features_chunk) if direct_decoder else features_chunk
        logits = logits.float() / temperature
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs.gather(-1, targets_chunk.unsqueeze(-1)).squeeze(-1)
        if with_entropy:
            entropy = -(log_probs.exp() * log_probs).sum(-1)
            return selected, entropy
        return selected

    logprobs, entropies = [], []
    for start in range(0, targets.shape[1], chunk_size):
        feature_chunk = features[:, start:start + chunk_size, :]
        target_chunk = targets[:, start:start + chunk_size]
        if torch.is_grad_enabled():
            stats = checkpoint(project_chunk, feature_chunk, target_chunk, use_reentrant=False)
        else:
            stats = project_chunk(feature_chunk, target_chunk)
        if with_entropy:
            lp, entropy = stats
            entropies.append(entropy)
        else:
            lp = stats
        logprobs.append(lp)
    if not logprobs:
        zero = features.sum(dim=-1) * 0.0
        return zero, zero if with_entropy else None
    return torch.cat(logprobs, dim=1), torch.cat(entropies, dim=1) if with_entropy else None
