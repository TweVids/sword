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
from .policy import token_statistics


DEFAULT_COLUMN_MAPPING: Dict[str, str] = {
    # Accuracy / Task Success (w = 1.0)
    "ground_truth": "accuracy",
    "execution": "accuracy",
    "practical_steps": "accuracy",
    "midi_music": "accuracy",
    "verifier_delta": "accuracy",

    # Formatting & Structural Style (w = 0.3)
    "output_format": "formatting",
    "thinking_formatting": "formatting",
    "thinking_tags": "formatting",
    "code_diff": "formatting",

    # Efficiency, Pacing & Quality (w = 0.2)
    "token_budget": "efficiency",
    "reasoning_structure": "efficiency",
    "context_recall": "efficiency",
    "writing_tells": "efficiency",
    "ui_design": "efficiency",
    "multi_turn_coherence": "efficiency",

    # Safety & Guards (Hard gate / direct penalty)
    "safety": "safety",
    "destructive_edit": "safety",
    "conversation_safety": "safety",
}

DEFAULT_COLUMN_WEIGHTS: Dict[str, float] = {
    "accuracy": 1.0,
    "formatting": 0.3,
    "efficiency": 0.2,
}


class ChunkedGRPOLoss(nn.Module):
    """
    Computes GRPO policy gradient loss with chunked cross-entropy / log-probabilities,
    preventing VRAM explosion during long-context RL updates.
    Supports both standard GRPO advantage normalization and GDPO (Decoupled Multi-Reward Normalization).
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

    @classmethod
    def compute_group_advantages(
        cls,
        rewards: List[Any],
        use_gdpo: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ) -> List[float]:
        """
        Computes group relative advantages:
        - If use_gdpo=True or rewards contain ScoredTrajectory/dict, routes to compute_gdpo_advantages
        - Otherwise computes standard GRPO group relative advantages:
            A_i = (R_i - mean(R)) / (std(R) + eps)
        """
        if use_gdpo or (rewards and not isinstance(rewards[0], (int, float))):
            advs, _ = cls.compute_gdpo_advantages(rewards, eps=eps, **kwargs)
            return advs

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

    @staticmethod
    def compute_gdpo_advantages(
        group_items: List[Any],
        column_weights: Optional[Dict[str, float]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        eps: float = 1e-8,
    ) -> Tuple[List[float], Dict[str, List[float]]]:
        """
        Computes Group reward-Decoupled Normalization Policy Optimization (GDPO) advantages.
        Reference: arXiv:2601.05242 (NVIDIA, Jan 2026).

        Separates multi-reward signals into 4 decoupled columns:
        1. accuracy   (w=1.0): task correctness, execution test passes, ground truth
        2. formatting (w=0.3): structural tags, markdown rules, code diff syntax
        3. efficiency (w=0.2): token budget pacing, anti-looping, context recall
        4. safety     (hard gate): terminal override & safety penalties applied directly
                      without polluting the variance of safe sibling rollouts.

        Returns:
            (total_advantages, column_advantages_dict)
        """
        if not group_items:
            return [], {}

        G = len(group_items)
        mapping = column_mapping or DEFAULT_COLUMN_MAPPING
        weights = column_weights or DEFAULT_COLUMN_WEIGHTS

        # Backward compatibility fallback: list of raw floats
        if all(isinstance(x, (int, float)) for x in group_items):
            std_adv = ChunkedGRPOLoss.compute_group_advantages(group_items, eps=eps)
            return std_adv, {"accuracy": std_adv}

        # 1. Extract and aggregate each rollout's scores into the 4 columns
        rollout_columns: List[Dict[str, float]] = []
        for item in group_items:
            if hasattr(item, "component_scores") and item.component_scores:
                comps = item.component_scores
            elif hasattr(item, "column_scores") and item.column_scores:
                comps = item.column_scores
            elif isinstance(item, dict):
                comps = item.get("component_scores") or item.get("column_scores") or item
            else:
                total_r = getattr(item, "total_reward", float(item) if isinstance(item, (int, float)) else 0.0)
                comps = {"ground_truth": total_r}

            col_totals = {"accuracy": 0.0, "formatting": 0.0, "efficiency": 0.0, "safety": 0.0}
            for k, val in comps.items():
                target_col = k if k in col_totals else mapping.get(k, "accuracy")
                if target_col not in col_totals:
                    col_totals[target_col] = 0.0
                col_totals[target_col] += float(val)

            rollout_columns.append(col_totals)

        if G == 1:
            s_val = rollout_columns[0]["safety"]
            adv = [round(s_val, 4)] if s_val < 0 else [0.0]
            return adv, {"accuracy": [0.0], "formatting": [0.0], "efficiency": [0.0], "safety": [s_val]}

        # 2. Decoupled normalization per channel
        norm_advs: Dict[str, List[float]] = {}
        for col in ["accuracy", "formatting", "efficiency"]:
            vals = [rollout_columns[i][col] for i in range(G)]
            mean_v = sum(vals) / G
            var_v = sum((v - mean_v) ** 2 for v in vals) / G
            std_v = math.sqrt(var_v)

            if std_v < eps:
                norm_advs[col] = [0.0 for _ in range(G)]
            else:
                norm_advs[col] = [(v - mean_v) / (std_v + eps) for v in vals]

        # Safety channel is handled directly as a hard constraint (not variance-normalized)
        safety_vals = [rollout_columns[i]["safety"] for i in range(G)]
        norm_advs["safety"] = safety_vals

        # 3. Combine normalized advantages with weights and apply safety gate
        total_advantages: List[float] = []
        for i in range(G):
            base_adv = sum(
                weights.get(col, 1.0) * norm_advs[col][i]
                for col in ["accuracy", "formatting", "efficiency"]
            )
            s_pen = safety_vals[i]

            if s_pen < 0:
                # Terminal safety constraint: clamp to negative, suppress reward
                adv_i = min(base_adv + s_pen, s_pen)
            else:
                adv_i = base_adv

            total_advantages.append(round(adv_i, 4))

        return total_advantages, norm_advs

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

        # Mask response tokens only for policy loss calculation
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lengths):
            response_mask[i, p_len:] = attention_mask[i, p_len:].bool()

        target_mask = response_mask[:, 1:]
        policy_logprobs, _ = token_statistics(model, input_ids, attention_mask, self.chunk_size)

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
        bsz, s_len, vocab_size = shift_logits.shape

        policy_logprobs = -F.cross_entropy(
            shift_logits.reshape(-1, vocab_size).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape(bsz, s_len)

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

        if old_logprobs.shape != policy_logprobs.shape or target_mask.shape != policy_logprobs.shape:
            raise ValueError("old_logprobs and response mask must align with shifted next-token logprobs")
        if ref_logprobs is not None and ref_logprobs.shape != policy_logprobs.shape:
            raise ValueError("reference logprobs must align with shifted next-token logprobs")
        min_len = policy_logprobs.size(1)
        p_logp = policy_logprobs[:, :min_len]
        old_logp = old_logprobs[:, :min_len].detach()
        mask = target_mask[:, :min_len].float()

        # Ratio: r = exp(log_pi - log_pi_old)
        log_ratio = (p_logp - old_logp).masked_fill(~target_mask.bool(), 0.0)
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
            ref_logp = ref_logprobs[:, :min_len].detach()
            # Schulman unbiased estimator: exp(log_ref - log_pi) - (log_ref - log_pi) - 1
            log_diff = (ref_logp - p_logp).masked_fill(~target_mask.bool(), 0.0)
            kl_per_token = torch.expm1(log_diff) - log_diff
            kl_div = (kl_per_token * mask).sum() / valid_tokens

        total_loss = policy_loss + (self.kl_coeff * kl_div)

        metrics = {
            "grpo_loss": round(total_loss.item(), 5),
            "policy_loss": round(policy_loss.item(), 5),
            "kl_div": round(kl_div.item(), 5),
            "mean_ratio": round(ratio.masked_select(target_mask.bool()).mean().item() if target_mask.any() else 1.0, 4),
            "mean_advantage": round(advantages.mean().item(), 4),
        }

        return total_loss, metrics


# Module-level convenient aliases
compute_gdpo_advantages = ChunkedGRPOLoss.compute_gdpo_advantages
compute_group_advantages = ChunkedGRPOLoss.compute_group_advantages


def compute_reference_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int = 512,
) -> Optional[torch.Tensor]:
    """
    BUG H4 fix helper: frozen-reference logprobs for the KL anchor.

    Runs token_statistics with the trainable adapters disabled (PEFT /
    unsloth `disable_adapter` context), so the returned logprobs come from
    the base policy, not the currently-trained LoRA. Returns None when the
    model does not support adapter disabling (e.g. full fine-tune or a mock),
    in which case callers must keep the KL term off and the training drift
    unanchored (same as before this fix).
    Shape: [Batch, SeqLen-1], aligned with token_statistics' shifted columns.
    """
    disable = getattr(model, "disable_adapter", None)
    if disable is None:
        return None
    from .policy import token_statistics

    was_training = model.training
    model.eval()
    try:
        with disable():
            with torch.no_grad():
                ref, _ = token_statistics(model, input_ids, attention_mask, chunk_size)
    finally:
        if was_training:
            model.train()
    return ref
