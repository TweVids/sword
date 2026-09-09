"""
Continuous Streaming Queue with Failure Taxonomy & Bounded Variation.
Implements Sections 2, 2a, 15.5, and 20 of rl-engine-spec.md:
- Kafka/queue-style streaming without halting between batches
- Failure-driven taxonomy requeuing (timeout / format_violation / wrong_answer)
- Ambiguous prompt routing to review queue (no noisy repeating)
- Terminal safety isolation
- Bounded retry (attempt cap per literal question) with surface variation
- Cluster-level pass/fail trend tracking and persistent-failure escalation
- Hot-reloading of new dataset links for "infinity time scale up"
- Checkpoint state persistence and resumption
"""

import os
import json
import re
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Optional, List, Dict, Any, Generator, Tuple, Union

from .schema import DatasetRow, FailureReason, DomainType, EffortTier, Difficulty
from ..trainer import download_from_drive


class ContinuousStreamingQueue:
    """
    Continual, dynamic data queue supporting:
    - Real-time dynamic dataset ingestion (Google Drive or local json/jsonl)
    - Non-blocking streaming batches
    - Bounded retry with question variation
    - Failure taxonomy routing
    - Cluster-level pass/fail tracking
    - Checkpoint save / restore
    """

    def __init__(
        self,
        data_sources: Optional[List[str]] = None,
        retry_cap: int = 3,
        cluster_failure_threshold: int = 6,
        cache_dir: str = "./data_cache",
        seed: int = 42,
    ):
        self.data_sources = list(data_sources or [])
        self.retry_cap = retry_cap
        self.cluster_failure_threshold = cluster_failure_threshold
        self.cache_dir = cache_dir
        self.seed = seed
        random.seed(seed)

        os.makedirs(self.cache_dir, exist_ok=True)

        # Primary problem buffer
        self.queue: deque[DatasetRow] = deque()
        # Active problems indexed by id
        self.known_problems: Dict[str, DatasetRow] = {}

        # Tracking attempts per literal question
        self.attempt_counts: Dict[str, int] = defaultdict(int)

        # Cluster tracking: cluster_id -> deque of recent results (1 for pass, 0 for fail)
        self.cluster_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.cluster_varied_templates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Isolated review queues
        self.review_queue: List[Dict[str, Any]] = []        # For ambiguous prompts & broken clusters
        self.safety_audit_queue: List[Dict[str, Any]] = []  # For safety violations

        # Processed sources tracking
        self.processed_sources: set = set()
        self.total_streamed: int = 0
        self.total_completed: int = 0

        # Load initial sources if provided
        for src in self.data_sources:
            self._load_source(src)

    def add_data_source(self, source: str) -> int:
        """
        Dynamically append a new data source (Drive link or local file)
        during active training without stopping the engine ("infinity time").
        """
        if source in self.processed_sources:
            print(f"[Sword-RL] Data source already registered: {source}")
            return 0
        self.data_sources.append(source)
        return self._load_source(source)

    def _resolve_source_path(self, source: str) -> str:
        """Resolves Google Drive URLs or verifies local paths."""
        if "drive.google.com" in source or "/d/" in source or "id=" in source:
            dest_filename = f"drive_dataset_{abs(hash(source)) % 1000000}.jsonl"
            dest_path = os.path.join(self.cache_dir, dest_filename)
            return download_from_drive(source, dest_path)
        return source

    def _load_source(self, source: str) -> int:
        """Loads items from a resolved path or url into the queue."""
        local_path = self._resolve_source_path(source)
        if not os.path.exists(local_path):
            print(f"[Sword-RL] Warning: Dataset file not found: {local_path}")
            return 0

        loaded = 0
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        row = self._dict_to_dataset_row(data, fallback_id=f"src_{len(self.processed_sources)}_{line_idx}")
                        if row.problem_id not in self.known_problems:
                            self.queue.append(row)
                            self.known_problems[row.problem_id] = row
                            loaded += 1
                    except Exception as e:
                        print(f"[Sword-RL] Warning: failed to parse row in {local_path}:{line_idx}: {e}")
            self.processed_sources.add(source)
            print(f"[Sword-RL] [OK] Ingested {loaded:,} problems from {source}. Total active queue: {len(self.queue):,}")
        except Exception as e:
            print(f"[Sword-RL] Error reading {local_path}: {e}")

        return loaded

    def _dict_to_dataset_row(self, data: Dict[str, Any], fallback_id: str) -> DatasetRow:
        """Converts raw dict to standardized DatasetRow."""
        problem_id = str(data.get("problem_id") or data.get("id") or fallback_id)
        user_problem = data.get("user_problem") or data.get("prompt") or data.get("problem") or data.get("question") or ""
        domain = data.get("domain", "general")
        effort_tier = data.get("effort_tier", "medium")
        difficulty = data.get("difficulty", "medium")
        explain_flag = bool(data.get("explain_flag", False))
        recheck_required = data.get("recheck_required", None)
        task_scope = data.get("task_scope", [])
        reference_answer = data.get("reference_answer", None)
        reference_corpus_id = data.get("reference_corpus_id", None)
        prior_failure_reason = data.get("prior_failure_reason", None)
        music_constraints = data.get("music_constraints", None)
        conversation_history = data.get("conversation_history", None)
        cluster_id = data.get("cluster_id") or f"{domain}_{difficulty}"
        attempt_count = int(data.get("attempt_count", 0))

        return DatasetRow(
            problem_id=problem_id,
            user_problem=user_problem,
            domain=domain,
            effort_tier=effort_tier,
            difficulty=difficulty,
            explain_flag=explain_flag,
            recheck_required=recheck_required,
            task_scope=task_scope,
            reference_answer=reference_answer,
            reference_corpus_id=reference_corpus_id,
            prior_failure_reason=prior_failure_reason,
            music_constraints=music_constraints,
            conversation_history=conversation_history,
            cluster_id=cluster_id,
            attempt_count=attempt_count,
            extra_metadata={k: v for k, v in data.items() if k not in {
                "problem_id", "id", "user_problem", "prompt", "problem", "question",
                "domain", "effort_tier", "difficulty", "explain_flag", "recheck_required",
                "task_scope", "reference_answer", "reference_corpus_id", "prior_failure_reason",
                "music_constraints", "conversation_history", "cluster_id", "attempt_count"
            }}
        )

    def append_problem(self, row: Union[DatasetRow, Dict[str, Any]]):
        """Adds a single problem directly into the queue."""
        if isinstance(row, dict):
            row = self._dict_to_dataset_row(row, fallback_id=f"dyn_{len(self.known_problems)}_{int(time.time()*1000)}")
        self.queue.append(row)
        self.known_problems[row.problem_id] = row

    def get_batch(self, batch_size: int = 4) -> List[DatasetRow]:
        """
        Pulls a batch of problems for parallel rollout generation.
        Ensures domain-diversity (Section 15.1) across the batch when possible.
        """
        if not self.queue:
            return []

        batch: List[DatasetRow] = []
        domains_in_batch: set = set()
        temp_requeue: deque[DatasetRow] = deque()

        # Try to pull domain-diverse items first
        while self.queue and len(batch) < batch_size:
            row = self.queue.popleft()
            if row.domain not in domains_in_batch or len(self.queue) < batch_size:
                batch.append(row)
                domains_in_batch.add(row.domain)
            else:
                temp_requeue.append(row)

        # Restore temporary skipped items back to head of queue
        while temp_requeue:
            self.queue.appendleft(temp_requeue.pop())

        # If batch still needs items, pop remainder sequentially
        while self.queue and len(batch) < batch_size:
            batch.append(self.queue.popleft())

        self.total_streamed += len(batch)
        return batch

    def handle_feedback(
        self,
        problem: DatasetRow,
        success: bool,
        failure_reason: Optional[FailureReason] = None,
        audit_note: Optional[str] = None,
    ):
        """
        Feedback routing implementing Section 2, 2a, and 15.5:
        - Tracks attempts per literal question
        - Updates cluster-level pass/fail metrics
        - Routes ambiguous_prompt and safety_violation to isolation queues
        - Bounded retry: if attempt >= retry_cap, spawns surface variation
        """
        cluster_id = problem.cluster_id or f"{problem.domain.value}_{problem.difficulty.value}"

        if success:
            self.cluster_history[cluster_id].append(1)
            self.total_completed += 1
            return

        # Record failure in cluster history
        self.cluster_history[cluster_id].append(0)

        # 1. Ambiguous prompt path -> review queue (Section 2)
        if failure_reason == FailureReason.AMBIGUOUS_PROMPT:
            self.review_queue.append({
                "reason": "ambiguous_prompt",
                "problem": asdict(problem),
                "timestamp": time.time(),
                "audit_note": audit_note,
            })
            print(f"[Sword-RL] [!] Problem {problem.problem_id} marked AMBIGUOUS_PROMPT -> routed to Review Queue.")
            return

        # 2. Safety violation path -> safety audit queue, never requeued normally (Section 8)
        if failure_reason == FailureReason.SAFETY_VIOLATION:
            self.safety_audit_queue.append({
                "reason": "safety_violation",
                "problem": asdict(problem),
                "timestamp": time.time(),
                "audit_note": audit_note,
            })
            print(f"[Sword-RL] [ALERT] Problem {problem.problem_id} triggered SAFETY_VIOLATION -> quarantined.")
            return

        # 3. Check persistent failure on cluster level (Section 2a)
        recent_cluster = list(self.cluster_history[cluster_id])
        if len(recent_cluster) >= self.cluster_failure_threshold and sum(recent_cluster) == 0:
            # Entire cluster has failed repeatedly
            self.review_queue.append({
                "reason": "persistent_cluster_failure",
                "cluster_id": cluster_id,
                "problem": asdict(problem),
                "timestamp": time.time(),
                "note": f"Cluster {cluster_id} failed {len(recent_cluster)} consecutive times.",
            })
            print(f"[Sword-RL] [!] Cluster {cluster_id} persistently failing -> escalated to Review Queue.")
            return

        # 4. Bounded Retry Logic (Section 2a & 15.5)
        self.attempt_counts[problem.problem_id] += 1
        current_attempts = self.attempt_counts[problem.problem_id]

        if current_attempts < self.retry_cap:
            # Requeue literal question with incremented attempt
            problem.attempt_count = current_attempts
            problem.prior_failure_reason = failure_reason
            self.queue.append(problem)
        else:
            # Reached literal retry cap -> Bounded Variation
            varied_problem = self._generate_cluster_variation(problem)
            self.queue.append(varied_problem)
            print(
                f"[Sword-RL] [VARIATION] Literal retry cap reached for {problem.problem_id} (attempts={current_attempts}). "
                f"Spawned varied question: {varied_problem.problem_id} in cluster '{cluster_id}'"
            )

    def _generate_cluster_variation(self, original: DatasetRow) -> DatasetRow:
        """
        Generates a surface-varied question from the same skill/difficulty/domain cluster (Section 2a).
        Substitutes numbers, variable names, or framing while preserving underlying reasoning structure.
        """
        orig_text = original.user_problem

        # Rule-based surface variation heuristics
        # 1. Perturb integers if found
        def perturb_number(m):
            num = int(m.group(0))
            offset = random.choice([-3, -2, -1, 1, 2, 3, 5])
            new_num = max(1, num + offset)
            return str(new_num)

        varied_text = re.sub(r"\b\d{1,4}\b", perturb_number, orig_text)

        # 2. Swap common variable names (e.g. x -> y, arr -> nums)
        var_swaps = {
            r"\barr\b": "nums",
            r"\bnums\b": "arr",
            r"\bx\b": "a",
            r"\by\b": "b",
            r"\bn\b": "k",
            r"\bfoo\b": "bar",
        }
        for pat, repl in var_swaps.items():
            if re.search(pat, varied_text):
                varied_text = re.sub(pat, repl, varied_text, count=2)
                break

        if varied_text == orig_text:
            varied_text = f"Consider a variant: {orig_text}"

        varied_id = f"{original.problem_id}_var{int(time.time()) % 10000}"

        return DatasetRow(
            problem_id=varied_id,
            user_problem=varied_text,
            domain=original.domain,
            effort_tier=original.effort_tier,
            difficulty=original.difficulty,
            explain_flag=original.explain_flag,
            recheck_required=original.recheck_required,
            task_scope=list(original.task_scope),
            reference_answer=None,  # Reset reference answer as values changed
            reference_corpus_id=original.reference_corpus_id,
            prior_failure_reason=None,
            music_constraints=original.music_constraints,
            conversation_history=original.conversation_history,
            cluster_id=original.cluster_id,
            attempt_count=0,
            extra_metadata={"derived_from": original.problem_id}
        )

    def get_cluster_pass_rate(self, cluster_id: str) -> float:
        """Returns the recent pass rate for a cluster."""
        history = self.cluster_history.get(cluster_id)
        if not history:
            return 1.0
        return sum(history) / len(history)

    def save_state(self, state_path: str):
        """Saves full streaming queue and curriculum state for checkpoint resumption."""
        state = {
            "data_sources": self.data_sources,
            "processed_sources": list(self.processed_sources),
            "total_streamed": self.total_streamed,
            "total_completed": self.total_completed,
            "attempt_counts": dict(self.attempt_counts),
            "cluster_history": {k: list(v) for k, v in self.cluster_history.items()},
            "review_queue_len": len(self.review_queue),
            "safety_audit_queue_len": len(self.safety_audit_queue),
            "remaining_queue_items": [asdict(p) for p in self.queue],
        }
        os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"[Sword-RL] [SAVED] Queue curriculum state saved -> {state_path}")

    def load_state(self, state_path: str) -> bool:
        """Restores queue state from checkpoint."""
        if not os.path.exists(state_path):
            print(f"[Sword-RL] No queue state found at {state_path}, starting fresh.")
            return False

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.data_sources = state.get("data_sources", [])
            self.processed_sources = set(state.get("processed_sources", []))
            self.total_streamed = state.get("total_streamed", 0)
            self.total_completed = state.get("total_completed", 0)
            self.attempt_counts = defaultdict(int, state.get("attempt_counts", {}))

            self.cluster_history = defaultdict(lambda: deque(maxlen=20))
            for k, v in state.get("cluster_history", {}).items():
                self.cluster_history[k] = deque(v, maxlen=20)

            self.queue.clear()
            for item in state.get("remaining_queue_items", []):
                row = self._dict_to_dataset_row(item, fallback_id=f"resumed_{len(self.queue)}")
                self.queue.append(row)
                self.known_problems[row.problem_id] = row

            print(f"[Sword-RL] [OK] Resumed queue state from {state_path}: {len(self.queue):,} remaining items.")
            return True
        except Exception as e:
            print(f"[Sword-RL] Error loading queue state from {state_path}: {e}")
            return False
