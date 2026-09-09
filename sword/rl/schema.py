"""
Bounty-Driven Continual RL Engine Data Schemas & Contracts.
Implements the core types, effort tiers, failure taxonomy, and dataset schemas
defined in rl-engine-spec.md.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union


class DomainType(str, Enum):
    CODE = "code"
    MATH = "math"
    SCIENCE = "science"
    WRITING = "writing"
    PRACTICAL = "practical"
    CONVERSATION = "conversation"
    TOOL_USE = "tool_use"
    MULTI_TURN = "multi_turn"
    MIDI_PIANO = "midi_piano"
    UI_DESIGN = "ui_design"
    GENERAL = "general"


class EffortTier(str, Enum):
    LOW = "low"         # Max 1,024 tokens, no recheck expected
    MEDIUM = "medium"   # Max 4,024 tokens, recheck optional
    HIGH = "high"       # Max 11,024 tokens, recheck expected
    XHIGH = "xhigh"     # Max 22,024 tokens, recheck expected
    ULTRA = "ultra"     # Max 32,024 tokens, recheck expected
    MAX = "max"         # Dynamic token budget

    @property
    def max_tokens(self) -> int:
        limits = {
            EffortTier.LOW: 1024,
            EffortTier.MEDIUM: 4024,
            EffortTier.HIGH: 11024,
            EffortTier.XHIGH: 22024,
            EffortTier.ULTRA: 32024,
            EffortTier.MAX: 65536,
        }
        return limits.get(self, 4024)

    @property
    def recheck_required(self) -> bool:
        return self in (EffortTier.HIGH, EffortTier.XHIGH, EffortTier.ULTRA, EffortTier.MAX)


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FailureReason(str, Enum):
    TIMEOUT = "timeout"
    FORMAT_VIOLATION = "format_violation"
    WRONG_ANSWER = "wrong_answer"
    AMBIGUOUS_PROMPT = "ambiguous_prompt"
    SAFETY_VIOLATION = "safety_violation"
    DESTRUCTIVE_EDIT = "destructive_edit"
    BUDGET_EXCEEDED = "budget_exceeded"
    DOMAIN_MISMATCH = "domain_mismatch"
    ROUTER_COLLAPSE = "router_collapse"


@dataclass
class DatasetRow:
    """
    Standard dataset row matching Section 20 of rl-engine-spec.md.
    """
    problem_id: str
    user_problem: str
    domain: DomainType = DomainType.GENERAL
    effort_tier: EffortTier = EffortTier.MEDIUM
    difficulty: Difficulty = Difficulty.MEDIUM
    explain_flag: bool = False
    recheck_required: Optional[bool] = None
    task_scope: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None
    reference_corpus_id: Optional[str] = None
    prior_failure_reason: Optional[FailureReason] = None
    music_constraints: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    cluster_id: Optional[str] = None
    attempt_count: int = 0
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.domain, str):
            try:
                self.domain = DomainType(self.domain.lower())
            except ValueError:
                self.domain = DomainType.GENERAL

        if isinstance(self.effort_tier, str):
            try:
                self.effort_tier = EffortTier(self.effort_tier.lower())
            except ValueError:
                self.effort_tier = EffortTier.MEDIUM

        if isinstance(self.difficulty, str):
            try:
                self.difficulty = Difficulty(self.difficulty.lower())
            except ValueError:
                self.difficulty = Difficulty.MEDIUM

        if isinstance(self.prior_failure_reason, str):
            try:
                self.prior_failure_reason = FailureReason(self.prior_failure_reason.lower())
            except ValueError:
                pass

        if self.recheck_required is None:
            self.recheck_required = self.effort_tier.recheck_required

        if self.cluster_id is None:
            self.cluster_id = f"{self.domain.value}_{self.difficulty.value}"


@dataclass
class Trajectory:
    """
    A generated model rollout trace.
    """
    prompt: str
    full_text: str
    reasoning_trace: str = ""
    final_answer: str = ""
    token_count: int = 0
    latency_sec: float = 0.0
    model_id: Optional[str] = None
    router_metrics: Optional[Dict[str, Any]] = None


@dataclass
class ScoredTrajectory:
    """
    A trajectory scored by System 1 (Primary Scorer) and System 2 (Verifier).
    """
    trajectory: Trajectory
    total_reward: float = 0.0
    advantage: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    failure_reason: Optional[FailureReason] = None
    verifier_answers: Optional[Dict[str, Any]] = None
    is_safe: bool = True
    audit_log: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierQuestion:
    id: str
    prompt: str
    answer_type: str  # "boolean", "categorical", "integer"
    expected_fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class VerifierRequest:
    row_id: str
    domain: str
    user_problem: str
    trajectory_trace: str
    final_answer: str
    questions: List[VerifierQuestion] = field(default_factory=list)


@dataclass
class VerifierAnswer:
    question_id: str
    answer: Union[bool, str, int]
    citation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
