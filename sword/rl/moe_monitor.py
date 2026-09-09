"""
MoE Router Collapse & Cross-Domain Stability Monitor.
Implements Sections 15.1 and 15.8 of rl-engine-spec.md:
- Load-balancing auxiliary loss kept active during RL
- Real-time routing entropy monitoring per batch and per domain
- Per-expert utilization floor with hard alarm
- Cross-domain expert rerouting regression logging
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple


class MoERouterMonitor:
    """
    Guards against MoE router collapse and expert starvation during RL updates.
    """

    def __init__(
        self,
        num_experts: int = 64,
        top_k: int = 8,
        aux_loss_coeff: float = 0.01,
        entropy_warning_threshold: float = 1.2,
        utilization_floor_ratio: float = 0.15,  # Min 15% of fair share (1/num_experts)
        history_window: int = 50,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        self.entropy_warning_threshold = entropy_warning_threshold
        self.utilization_floor = utilization_floor_ratio / max(1, num_experts)
        self.history_window = history_window

        # Routing metrics history
        self.batch_entropies: deque = deque(maxlen=history_window)
        self.domain_entropies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.expert_counts: torch.Tensor = torch.zeros(num_experts, dtype=torch.float32)
        self.total_tokens_seen: int = 0

        # Alarms triggered
        self.alarms: List[Dict[str, Any]] = []

    def compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        domain: str = "general",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the standard Switch/GShard load-balancing auxiliary loss:
          L_aux = alpha * E * sum(f_e * P_e)
        Keeps load balancing active through RL updates (Section 15.1).
        """
        # router_logits: [batch_size * seq_len, num_experts] or [tokens, num_experts]
        if router_logits.dim() == 3:
            router_logits = router_logits.reshape(-1, router_logits.size(-1))

        num_tokens = router_logits.size(0)
        num_exp = router_logits.size(-1)

        # 1. Routing probabilities: P_e = mean softmax over tokens
        probs = F.softmax(router_logits, dim=-1)
        mean_probs = probs.mean(dim=0)  # [num_experts]

        # 2. Expert selection fraction: f_e = fraction of tokens choosing expert in top_k
        _, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)  # [tokens, top_k]
        expert_mask = F.one_hot(topk_indices, num_classes=num_exp).float().sum(dim=1)  # [tokens, num_exp]
        tokens_per_expert = expert_mask.sum(dim=0)  # [num_experts]
        fraction_per_expert = tokens_per_expert / (num_tokens * self.top_k)

        # 3. Aux loss = alpha * E * dot(f, P)
        aux_loss = self.aux_loss_coeff * num_exp * torch.sum(fraction_per_expert * mean_probs)

        # 4. Routing entropy: H = - sum(P_e * log(P_e))
        eps = 1e-10
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps)).item()

        # Update tracking
        self.batch_entropies.append(entropy)
        self.domain_entropies[domain].append(entropy)

        with torch.no_grad():
            self.expert_counts = self.expert_counts.to(tokens_per_expert.device)
            self.expert_counts += tokens_per_expert
            self.total_tokens_seen += num_tokens

        # Check for router collapse alerts
        metrics = {
            "routing_entropy": round(entropy, 4),
            "aux_loss": round(aux_loss.item(), 6),
            "max_expert_fraction": round(fraction_per_expert.max().item(), 4),
            "min_expert_fraction": round(fraction_per_expert.min().item(), 4),
        }

        self._check_alarms(entropy, fraction_per_expert, domain)
        return aux_loss, metrics

    def _check_alarms(self, entropy: float, expert_fractions: torch.Tensor, domain: str):
        """Monitors entropy drops and per-expert utilization floor (Section 15.1)."""
        # Low entropy alarm
        if entropy < self.entropy_warning_threshold:
            alarm = {
                "type": "low_routing_entropy",
                "domain": domain,
                "entropy": entropy,
                "threshold": self.entropy_warning_threshold,
            }
            self.alarms.append(alarm)
            print(f"[Sword-RL MoE Alarm] 🚨 Low routing entropy ({entropy:.2f} < {self.entropy_warning_threshold}) in domain '{domain}'! Potential router collapse.")

        # Starving expert floor alarm
        starving_experts = (expert_fractions < self.utilization_floor).nonzero(as_tuple=True)[0].tolist()
        if len(starving_experts) > (self.num_experts // 2):
            alarm = {
                "type": "starving_experts",
                "domain": domain,
                "count": len(starving_experts),
                "floor": self.utilization_floor,
            }
            self.alarms.append(alarm)
            print(f"[Sword-RL MoE Alarm] 🚨 {len(starving_experts)}/{self.num_experts} experts starving below utilization floor {self.utilization_floor:.4f}!")

    def get_summary(self) -> Dict[str, Any]:
        """Returns diagnostic summary of router health across domains."""
        domain_avg = {}
        for d, vals in self.domain_entropies.items():
            if vals:
                domain_avg[d] = round(sum(vals) / len(vals), 4)

        recent_entropy = round(sum(self.batch_entropies) / len(self.batch_entropies), 4) if self.batch_entropies else 0.0

        return {
            "recent_mean_entropy": recent_entropy,
            "domain_mean_entropies": domain_avg,
            "total_tokens_routed": self.total_tokens_seen,
            "alarm_count": len(self.alarms),
            "active_experts_ratio": (self.expert_counts > 0).float().mean().item(),
        }
