"""
Sword Continual Reinforcement Learning (RL) Engine.
Implements the Bounty-Driven Continual RL Engine specification (rl-engine-spec.md).
"""

from .schema import (
    DatasetRow,
    DomainType,
    EffortTier,
    Difficulty,
    FailureReason,
    Trajectory,
    ScoredTrajectory,
    VerifierRequest,
    VerifierQuestion,
    VerifierAnswer,
)
from .queue import ContinuousStreamingQueue
from .scorer import PrimaryScorer
from .verifier import ExternalVerifier
from .moe_monitor import MoERouterMonitor
from .loss import ChunkedGRPOLoss
from .engine import GRPOTrainer, start_grpo

__all__ = [
    "start_grpo",
    "GRPOTrainer",
    "ContinuousStreamingQueue",
    "PrimaryScorer",
    "ExternalVerifier",
    "MoERouterMonitor",
    "ChunkedGRPOLoss",
    "DatasetRow",
    "DomainType",
    "EffortTier",
    "Difficulty",
    "FailureReason",
    "Trajectory",
    "ScoredTrajectory",
    "VerifierRequest",
    "VerifierQuestion",
    "VerifierAnswer",
]
