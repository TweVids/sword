"""
Memory-Efficient Chunked GRPO Policy Gradient Loss.
Implements Unsloth-style chunked token logprob computation for Group Relative Policy Optimization (GRPO):
- Group advantage relative normalization
- Clipped surrogate policy gradient (epsilon = 0.2)
- Unbiased KL divergence estimator (Schulman)
- Sequence-chunked logit calculation to avoid [Batch, SeqLen, Vocab] VRAM materialization
"""

import math
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedGRPOLoss(nn.Module):
    """
    Computes GRPO policy gradient loss with chunked cross-entropy / log-probabilities,
    preventing VRAM explosion during long-context RL updates.
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        kl_coeff: float = 0.04,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.chunk_size = chunk_size

    @staticmethod
    def compute_group_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
        """
        Computes standard GRPO group relative advantages:
          A_i = (R_i - mean(R)) / (std(R) + eps)
        """
        if not rewards:
            return []
        if len(rewards) == 1:
            return [0.0]

        mean_r = sum(rewards) / len(rewards)
        variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = math.sqrt(variance)

        if std_r < eps:
            return [0.0 for _ in rewards]

        return [(r - mean_r) / (std_r + eps) for r in rewards]

    def forward_chunked(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: List[int],
        advantages: torch.Tensor,
        old_logprobs: Optional[torch.Tensor] = None,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the clipped GRPO surrogate loss in chunks across sequence length.

        Args:
            model: Active policy model (e.g. patched Ling-3.0 with LoRA)
            input_ids: [Batch, SeqLen] (concatenated prompt + generated response)
            attention_mask: [Batch, SeqLen]
            prompt_lengths: List of integer lengths indicating where prompts end and responses begin
            advantages: [Batch] advantage scalars per rollout
            old_logprobs: Optional [Batch, SeqLen] pre-computed logprobs at rollout time
            ref_logprobs: Optional [Batch, SeqLen] reference model logprobs for KL penalty
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Forward pass through base backbone to obtain final hidden states
        # (LM Head is applied in chunks to conserve VRAM)
        lm_head = getattr(model, "lm_head", None)
        base_model = getattr(model, "model", model)

        # Mask response tokens only for policy loss calculation
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lengths):
            response_mask[i, p_len:] = attention_mask[i, p_len:].bool()

        # If model has custom forward or standard transformer forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            # Output already has logits
            logits = outputs.logits
            return self._compute_loss_from_logits(
                logits, input_ids, response_mask, advantages, old_logprobs, ref_logprobs
            )

        # 2. Chunked logprob extraction using lm_head
        # Target tokens are input_ids shifted by 1
        targets = input_ids[:, 1:]
        target_mask = response_mask[:, 1:]
        h_states = hidden_states[:, :-1, :]  # [B, L-1, H]
        num_tokens = h_states.size(1)

        policy_logprobs_list = []

        # Iterate over sequence in chunks
        for start_idx in range(0, num_tokens, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_tokens)
            chunk_h = h_states[:, start_idx:end_idx, :]  # [B, chunk_len, H]
            chunk_targets = targets[:, start_idx:end_idx]  # [B, chunk_len]

            # Compute logits only for this chunk
            if lm_head is not None:
                chunk_logits = lm_head(chunk_h)  # [B, chunk_len, V]
            else:
                chunk_logits = F.linear(chunk_h, model.get_output_embeddings().weight)

            # Compute log softmax for targets directly
            log_probs = F.log_softmax(chunk_logits.float(), dim=-1)
            token_logprobs = torch.gather(log_probs, dim=-1, index=chunk_targets.unsqueeze(-1)).squeeze(-1)
            policy_logprobs_list.append(token_logprobs)

        # Concatenate across chunks -> [B, L-1]
        policy_logprobs = torch.cat(policy_logprobs_list, dim=1)

        return self._compute_surrogate_loss(
            policy_logprobs, target_mask, advantages, old_logprobs, ref_logprobs
        )

    def _compute_loss_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: Optional[torch.Tensor] = None,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Fallback path if model already returned logits."""
        targets = input_ids[:, 1:]
        target_mask = response_mask[:, 1:]
        shift_logits = logits[:, :-1, :]

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        policy_logprobs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        return self._compute_surrogate_loss(
            policy_logprobs, target_mask, advantages, old_logprobs, ref_logprobs
        )

    def _compute_surrogate_loss(
        self,
        policy_logprobs: torch.Tensor,
        target_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: Optional[torch.Tensor] = None,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the clipped policy gradient surrogate loss:
          r_t = exp(log_pi - log_pi_old)
          L = - min(r_t * A, clip(r_t, 1-eps, 1+eps) * A)
        """
        if old_logprobs is None:
            # First pass: old_logprobs detached from policy
            old_logprobs = policy_logprobs.detach()

        # Align lengths if needed
        min_len = min(policy_logprobs.size(1), old_logprobs.size(1), target_mask.size(1))
        p_logp = policy_logprobs[:, :min_len]
        old_logp = old_logprobs[:, :min_len]
        mask = target_mask[:, :min_len].float()

        # Ratio: r = exp(log_pi - log_pi_old)
        log_ratio = p_logp - old_logp
        ratio = torch.exp(log_ratio)

        # Reshape advantages to broadcast with tokens: [Batch, 1]
        adv = advantages.unsqueeze(-1).to(ratio.device)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss_per_token = -torch.min(surr1, surr2)

        # Masked average over valid response tokens
        valid_tokens = mask.sum().clamp(min=1.0)
        policy_loss = (policy_loss_per_token * mask).sum() / valid_tokens

        # KL divergence penalty against reference model if available
        kl_div = torch.tensor(0.0, device=policy_loss.device)
        if ref_logprobs is not None:
            ref_logp = ref_logprobs[:, :min_len]
            # Schulman unbiased estimator: exp(log_ref - log_pi) - (log_ref - log_pi) - 1
            kl_per_token = torch.exp(ref_logp - p_logp) - (ref_logp - p_logp) - 1.0
            kl_div = (kl_per_token * mask).sum() / valid_tokens

        total_loss = policy_loss + (self.kl_coeff * kl_div)

        metrics = {
            "grpo_loss": round(total_loss.item(), 5),
            "policy_loss": round(policy_loss.item(), 5),
            "kl_div": round(kl_div.item(), 5),
            "mean_ratio": round(ratio.mean().item(), 4),
            "mean_advantage": round(advantages.mean().item(), 4),
        }

        return total_loss, metrics
