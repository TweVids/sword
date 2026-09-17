"""
Primary Deterministic Scorer (System 1 - Authoritative).
Implements the deterministic, countable, and structural reward rules from rl-engine-spec.md:
- Section 3: Soft-overage token budget penalties & complexity cross-matching
- Section 4: Self-correction / recheck anti-gaming rules
- Section 5: Context recall, density, and paraphrase overlap
- Section 6: Output formatting & structural list detection (no keyword gaming)
- Section 7 & 7a: Code editing diff parser, destructive-edit guard, scope check
- Section 8: Terminal safety override (clamped to -1.0 .. -2.0)
- Section 9: UI / Front-end "design slop" tell-detection
- Section 10: Writing, practical, conversational, multi-turn, tool-use, science checks
- Section 11a: MIDI / piano note parser (playability, duration math, voice-leading)
- Section 15.4: Component contribution capping & audit logging
"""

import math
import re
import json
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple, Set

from .schema import (
    DatasetRow,
    Trajectory,
    ScoredTrajectory,
    DomainType,
    EffortTier,
    FailureReason,
)
from .loss import DEFAULT_COLUMN_MAPPING


class PrimaryScorer:
    """
    Authoritative deterministic scoring engine.
    Never uses fuzzy keyword counts where semantic judgment is needed.
    Only evaluates verifiable, countable, structural, and parser-checkable metrics.
    """

    def __init__(
        self,
        component_cap: float = 0.35,
        safety_penalty_clamp: float = -1.5,
        gym: Optional[Any] = None,
    ):
        self.component_cap = component_cap
        self.safety_penalty_clamp = safety_penalty_clamp
        self.gym = gym

        # Known AI cliché hex colors (Section 9 palette fingerprinting)
        self.cliche_colors = {
            "#faedcd", "#fefae0", "#e2725b", "#fffdd0",
            "#0d1117", "#161b22", "#f5ebe0", "#d4a373"
        }

        # Cliché prose phrases to penalize in writing (Section 10)
        self.writing_cliches = [
            r"\bdelve into\b",
            r"\btapestry of\b",
            r"\ba testament to\b",
            r"\bin conclusion,\s*it is clear\b",
            r"\bbustling\b",
            r"\bhaven\b",
        ]

        # Template chrome tells for UI (Section 9)
        self.template_chrome_patterns = [
            r"\b[A-Z]\s+[A-Z]\s+[A-Z]\s+[A-Z]+\b",  # Tracked-out all-caps: F E A T U R E S
            r"\b[A-Z]{3,}\s*—\s*\w+",               # Em-dash labels: WORD — fragment
            r"(?:Button|Start|Learn More|Explore)\s*(?:->|→|&rarr;)",  # Trailing arrow obsession
        ]

    def score_trajectory(
        self,
        problem: DatasetRow,
        traj: Trajectory,
        estimated_token_count: Optional[int] = None,
    ) -> ScoredTrajectory:
        """
        Executes all deterministic rules against the trajectory.
        Produces a ScoredTrajectory with detailed component breakdown.
        """
        token_count = estimated_token_count or traj.token_count or max(1, len(traj.full_text) // 4)

        components: Dict[str, float] = {}
        audit: Dict[str, Any] = {}
        failure_reason: Optional[FailureReason] = None
        is_safe: bool = True

        # =========================================================
        # 1. Section 8: Terminal Safety Override (Highest Priority)
        # =========================================================
        safety_score, safety_audit = self._check_safety(traj.full_text)
        components["safety"] = safety_score
        audit["safety"] = safety_audit
        if safety_score < 0:
            is_safe = False
            failure_reason = FailureReason.SAFETY_VIOLATION
            # Terminal override clamp (-1.0 to -2.0)
            total_reward = max(self.safety_penalty_clamp, safety_score)
            col_scores = {"accuracy": 0.0, "formatting": 0.0, "efficiency": 0.0, "safety": total_reward}
            return ScoredTrajectory(
                trajectory=traj,
                total_reward=total_reward,
                component_scores=components,
                column_scores=col_scores,
                failure_reason=failure_reason,
                is_safe=False,
                audit_log=audit,
            )

        # =========================================================
        # 2. Section 3: Effort Tiers & Soft-Overage Penalties
        # =========================================================
        budget_score, budget_audit = self._score_token_budget(problem.effort_tier, token_count, problem.difficulty)
        components["token_budget"] = budget_score
        audit["token_budget"] = budget_audit
        if budget_score <= -1.0:
            failure_reason = FailureReason.BUDGET_EXCEEDED

        # =========================================================
        # 3. Section 5: Context Recall, Structure & Density
        # =========================================================
        restatement_score, restatement_audit = self._score_context_recall(problem.user_problem, traj.full_text)
        components["context_recall"] = restatement_score
        audit["context_recall"] = restatement_audit

        # =========================================================
        # 4. Section 6: Output Formatting (Structural checks)
        # =========================================================
        format_score, format_audit = self._score_formatting(problem.explain_flag, traj.final_answer)
        components["output_format"] = format_score
        audit["output_format"] = format_audit
        if format_score < -0.15 and not failure_reason:
            failure_reason = FailureReason.FORMAT_VIOLATION

        # =========================================================
        # 4a. Reasoning Structure: 3-5 Sentence Cap & Anti-Spam Loop
        # =========================================================
        reasoning_score, reasoning_audit = self._score_reasoning_structure(traj.reasoning_trace)
        components["reasoning_structure"] = reasoning_score
        audit["reasoning_structure"] = reasoning_audit
        if reasoning_audit.get("repetitive_loop_detected"):
            failure_reason = FailureReason.REASONING_LOOP

        # =========================================================
        # 4b. Thinking Formatting: Prohibit Markdown Bold (**text**, **Step 1**)
        # =========================================================
        thinking_fmt_score, thinking_fmt_audit = self._score_thinking_formatting(traj.reasoning_trace)
        components["thinking_formatting"] = thinking_fmt_score
        audit["thinking_formatting"] = thinking_fmt_audit

        # =========================================================
        # 4c. Thinking Tags: Opening <think> and Closing </think> Verification
        # =========================================================
        tag_score, tag_audit = self._score_thinking_tags(problem, traj)
        components["thinking_tags"] = tag_score
        audit["thinking_tags"] = tag_audit
        if tag_score < -0.15 and not failure_reason:
            failure_reason = FailureReason.FORMAT_VIOLATION

        # =========================================================
        # 5. Domain-Specific Structural Rules
        # =========================================================
        domain = problem.domain
        if domain == DomainType.CODE:
            code_score, code_audit = self._score_code_editing(problem, traj)
            components["code_diff"] = code_score
            audit["code"] = code_audit
            if "gym_execution" in code_audit:
                gym_exec = code_audit["gym_execution"]
                components["execution"] = gym_exec.get("reward", 1.0 if gym_exec.get("success") else -0.2)
            if code_audit.get("destructive_violation"):
                failure_reason = FailureReason.DESTRUCTIVE_EDIT

        elif domain == DomainType.UI_DESIGN:
            ui_score, ui_audit = self._score_ui_design(traj.full_text)
            components["ui_design"] = ui_score
            audit["ui_design"] = ui_audit

        elif domain == DomainType.MIDI_PIANO:
            midi_score, midi_audit = self._score_midi(problem.music_constraints, traj.full_text)
            components["midi_music"] = midi_score
            audit["midi"] = midi_audit

        elif domain == DomainType.WRITING:
            writing_score, writing_audit = self._score_writing(traj.full_text)
            components["writing_tells"] = writing_score
            audit["writing"] = writing_audit

        elif domain == DomainType.PRACTICAL:
            practical_score, practical_audit = self._score_practical(traj.final_answer)
            components["practical_steps"] = practical_score
            audit["practical"] = practical_audit

        elif domain == DomainType.CONVERSATION:
            conv_score, conv_audit = self._score_conversation(traj.full_text)
            components["conversation_safety"] = conv_score
            audit["conversation"] = conv_audit

        elif domain == DomainType.MULTI_TURN and problem.conversation_history:
            mt_score, mt_audit = self._score_multi_turn(problem.conversation_history, traj.full_text)
            components["multi_turn_coherence"] = mt_score
            audit["multi_turn"] = mt_audit

        elif domain in (DomainType.MATH, DomainType.SCIENCE) and problem.reference_answer:
            ref_score, ref_audit = self._score_ground_truth(problem.reference_answer, traj.final_answer)
            components["ground_truth"] = ref_score
            audit["ground_truth"] = ref_audit
            if ref_score < 0 and not failure_reason:
                failure_reason = FailureReason.WRONG_ANSWER

        # =========================================================
        # 6. Section 15.4: Component Capping & Composite Sum
        # =========================================================
        # Prevent any single positive metric from cross-subsidizing hard failures
        capped_components: Dict[str, float] = {}
        for k, v in components.items():
            if k == "execution":
                capped_components[k] = max(-1.0, min(1.0, v))
            elif v > 0:
                capped_components[k] = min(v, self.component_cap)
            else:
                capped_components[k] = v

        total_reward = sum(capped_components.values())
        column_scores = self.group_components_to_columns(capped_components)

        # Determine overall failure reason if negative reward
        if total_reward < 0 and not failure_reason:
            failure_reason = FailureReason.WRONG_ANSWER

        return ScoredTrajectory(
            trajectory=traj,
            total_reward=round(total_reward, 4),
            component_scores=capped_components,
            column_scores=column_scores,
            failure_reason=failure_reason,
            is_safe=is_safe,
            audit_log=audit,
        )

    # ------------------------------------------------------------------
    # Section 8: Terminal Safety Check
    # ------------------------------------------------------------------
    def _check_safety(self, text: str) -> Tuple[float, Dict[str, Any]]:
        text_lower = text.lower()

        # Hard attack / dangerous keywords
        harm_patterns = [
            r"\b(cbrn|sarin|ricin|vx nerve agent|botulinum toxin)\b",
            r"\b(exploit payload|metasploit script|zero-day weaponization)\b",
            r"\b(bypassing security filters|jailbreak instructions)\b",
        ]
        for pat in harm_patterns:
            if re.search(pat, text_lower):
                return -1.5, {"violation": True, "pattern": pat}

        return 0.0, {"violation": False}

    # ------------------------------------------------------------------
    # Section 3: Token Budget & Soft Overage
    # ------------------------------------------------------------------
    def _score_token_budget(
        self,
        tier: EffortTier,
        token_count: int,
        difficulty: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        max_budget = tier.max_tokens
        audit = {"token_count": token_count, "max_budget": max_budget, "tier": tier.value}

        # Check overage & effort correctness
        if token_count <= max_budget:
            # Low effort: model minimized tokens properly (< 1024)
            if tier == EffortTier.LOW:
                if token_count <= 800:
                    return 0.10, {**audit, "rapid_low_effort_rewarded": True}
                elif token_count > 1500:
                    return -0.20, {**audit, "complexity_cross_match_penalty": -0.20}
                return 0.0, audit

            # High / Ultra / Max effort: verify substantive reasoning took place
            if tier in (EffortTier.HIGH, EffortTier.XHIGH, EffortTier.ULTRA, EffortTier.MAX):
                if token_count >= 200:
                    return 0.10, {**audit, "deep_effort_rewarded": True}
                else:
                    # Effort was set to high/ultra, but output was superficial without verification
                    return -0.15, {**audit, "insufficient_effort_penalty": -0.15}

            return 0.0, audit

        overage_ratio = (token_count - max_budget) / max_budget
        audit["overage_ratio"] = round(overage_ratio, 3)

        # 0-10% over budget: no penalty
        if overage_ratio <= 0.10:
            return 0.0, audit

        # 10-50% over: linear penalty from 0 to -0.5
        if overage_ratio <= 0.50:
            penalty = -0.5 * ((overage_ratio - 0.10) / 0.40)
            return round(penalty, 3), audit

        # 50%+ over: -1.0
        return -1.0, audit

    # ------------------------------------------------------------------
    # Section 5: Context Recall & Restatement Check
    # ------------------------------------------------------------------
    def _score_context_recall(self, prompt: str, text: str) -> Tuple[float, Dict[str, Any]]:
        words_prompt = set(re.findall(r"\b\w{4,}\b", prompt.lower()))
        if not words_prompt:
            return 0.0, {}

        # Look at the first ~100 words of the trace
        first_100_words = re.findall(r"\b\w{4,}\b", text[:600].lower())
        if not first_100_words:
            return 0.0, {}

        set_first = set(first_100_words)
        overlap = len(words_prompt.intersection(set_first)) / len(words_prompt)

        # Anti-gaming: Exact verbatim repetition (> 0.90) gets 0 credit;
        # healthy paraphrase restatement (0.35 to 0.85) gets +0.15
        if 0.35 <= overlap <= 0.85:
            return 0.15, {"overlap": round(overlap, 3), "recalled": True}
        return 0.0, {"overlap": round(overlap, 3), "recalled": False}

    # ------------------------------------------------------------------
    # Section 6: Output Formatting (Numbered/Bulleted List Check)
    # ------------------------------------------------------------------
    def _score_formatting(self, explain_flag: bool, answer: str) -> Tuple[float, Dict[str, Any]]:
        if not explain_flag:
            return 0.0, {"checked": False}

        # Check for structured markdown list
        has_numbered = bool(re.search(r"^\s*\d+\.\s+", answer, re.MULTILINE))
        has_bullets = bool(re.search(r"^\s*[-*•]\s+", answer, re.MULTILINE))

        if has_numbered or has_bullets:
            return 0.10, {"structured_list_found": True}
        else:
            return -0.20, {"structured_list_found": False, "penalty": -0.20}

    # ------------------------------------------------------------------
    # Reasoning Structure: 3-5 Sentences Block Cap & Loop Detection
    # ------------------------------------------------------------------
    def _score_reasoning_structure(self, trace: str) -> Tuple[float, Dict[str, Any]]:
        if not trace or len(trace.strip()) < 20:
            return 0.0, {"checked": False, "reason": "empty_or_short_trace"}

        audit: Dict[str, Any] = {}

        # 1. Anti-Reward Hacking / Spam / Looping Detection
        # Check if model actively repeats or loops sentences in reasoning
        raw_sentences = [s.strip() for s in re.split(r"[.!?\n]+", trace) if len(s.strip().split()) >= 4]
        sentence_counts: Dict[str, int] = defaultdict(int)
        repeated_sentence = None
        for s in raw_sentences:
            norm = re.sub(r"\s+", " ", s.lower())
            sentence_counts[norm] += 1
            if sentence_counts[norm] >= 2:
                repeated_sentence = s
                break

        if repeated_sentence:
            audit["repetitive_loop_detected"] = True
            audit["repeated_quote"] = repeated_sentence[:100]
            audit["penalty"] = -0.5
            return -0.5, audit

        # 2. Block/Paragraph Analysis (\n separated wall-of-text check)
        blocks = [b.strip() for b in re.split(r"\n+", trace) if b.strip()]
        if not blocks:
            return 0.0, {"checked": False}

        substantive_blocks = 0
        has_wall_of_text = False
        max_sentences_in_block = 0
        block_sentence_counts = []

        for block in blocks:
            # Anti-hack: Check for split numbers or trivial fragments (no point gain)
            # E.g. "1.", "Step 1", "42", or lines with < 3 alphabetical words
            words = re.findall(r"\b[a-zA-Z]{2,}\b", block)
            if len(words) < 3 or re.fullmatch(r"^(?:step\s*)?\d+[.:\)]?\s*$", block, re.IGNORECASE):
                # Trivial number line or split fragment -> no point gain, ignore from substantive blocks
                continue

            # Count full sentences in substantive block
            sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", block) if len(s.strip().split()) >= 3]
            s_count = max(1, len(sentences))
            block_sentence_counts.append(s_count)
            substantive_blocks += 1
            if s_count > max_sentences_in_block:
                max_sentences_in_block = s_count

            # Cap is 3 to 5 sentences per block. Larger will be penalized -0.2
            if s_count > 5:
                has_wall_of_text = True

        audit["substantive_blocks"] = substantive_blocks
        audit["block_sentence_counts"] = block_sentence_counts
        audit["max_sentences_in_block"] = max_sentences_in_block

        # If any block exceeded 5 sentences -> -0.2 wall-of-text penalty
        if has_wall_of_text:
            audit["wall_of_text_violation"] = True
            audit["penalty"] = -0.2
            return -0.2, audit

        # If substantive blocks exist and all are correctly paced within 3 to 5 sentences -> +0.1 reward
        # (Note: split numbers alone without 3-5 sentence substantive blocks yield 0.0 - no point gain)
        if substantive_blocks > 0 and all(3 <= sc <= 5 for sc in block_sentence_counts):
            audit["well_structured_reasoning"] = True
            audit["reward"] = 0.1
            return 0.1, audit

        # Substantive blocks exist but under 3 sentences (e.g. 1-2 sentence fragments) -> no point gain (0.0)
        return 0.0, audit

    # ------------------------------------------------------------------
    # Thinking Formatting: No Markdown Bold, No Early "step-by-step", No Rapid Short Lists (<15 words)
    # ------------------------------------------------------------------
    def _score_thinking_formatting(self, trace: str) -> Tuple[float, Dict[str, Any]]:
        if not trace:
            return 0.0, {"checked": False}

        score = 0.0
        audit: Dict[str, Any] = {}

        # 1. Prohibited bold formatting like **text** or **Step 1**
        bold_patterns = re.findall(r"\*\*(?!\s)[^*\n]+?(?<!\s)\*\*|__(?!\s)[^_\n]+?(?<!\s)__", trace)
        if bold_patterns:
            score -= 0.2
            audit["prohibited_bold_found"] = True
            audit["bold_matches"] = bold_patterns[:5]

        # Extract early text preserving newlines for structural list detection
        words = trace.split()
        if len(words) <= 200:
            first_200_text = trace
        else:
            match_200 = re.match(r"(?:\s*\S+){1,200}", trace)
            first_200_text = match_200.group(0) if match_200 else trace[:1200]
        first_200_words = " ".join(words[:200])

        # 2. "Step-by-step" / "step to step" prohibition in the first 200 words
        step_pattern = r"\b(?:step[\s\-_]*(?:by|to)[\s\-_]*step)\b"
        if re.search(step_pattern, first_200_words, re.IGNORECASE):
            score -= 0.2
            audit["early_step_by_step_penalty"] = -0.2
            audit["early_step_by_step_detected"] = True

        # 3. Rapid short numbered list check in early thinking (e.g. 1. ... \n 2. ...)
        # If consecutive list items appear and any has < 15 words -> -0.2 penalty
        # Substantive items with >= 15 words avoid false positives and receive no penalty
        list_items = re.findall(r"(?:^|\n)\s*\d+[.:\)]\s*(.*?)(?=(?:\n\s*\d+[.:\)]|\Z))", first_200_text, re.DOTALL)
        if len(list_items) >= 2:
            short_items = [item.strip() for item in list_items if len(item.split()) < 15]
            if short_items:
                score -= 0.2
                audit["rapid_short_list_penalty"] = -0.2
                audit["short_list_items"] = [s[:60] for s in short_items[:3]]
            else:
                audit["substantive_list_allowed"] = True

        return max(-0.4, score), audit

    @staticmethod
    def group_components_to_columns(
        components: Dict[str, float],
        mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Groups fine-grained component scores into the 4 GDPO columns:
        accuracy, formatting, efficiency, safety.
        """
        mapping = mapping or DEFAULT_COLUMN_MAPPING
        columns = {"accuracy": 0.0, "formatting": 0.0, "efficiency": 0.0, "safety": 0.0}
        for comp_name, score in components.items():
            target_col = mapping.get(comp_name, "accuracy")
            if target_col not in columns:
                columns[target_col] = 0.0
            columns[target_col] += float(score)
        return {k: round(v, 4) for k, v in columns.items()}

    @staticmethod
    def is_thinking_required(problem: DatasetRow) -> bool:
        """
        Determines whether the thinking block (<think> ... </think>) is required or expected.
        Triggered when:
        1. Explicit metadata / flag overrides (e.g. require_thinking=True)
        2. High/Ultra effort tiers (recheck_required is True, effort_tier >= HIGH)
        3. Reasoning domains (Math, Science, Code)
        4. Problem prompt explicitly cues reasoning/step-by-step thinking
        5. Explain flag is True
        """
        if problem.extra_metadata.get("require_thinking") is not None:
            return bool(problem.extra_metadata["require_thinking"])
        if problem.recheck_required:
            return True
        if problem.effort_tier in (EffortTier.HIGH, EffortTier.XHIGH, EffortTier.ULTRA, EffortTier.MAX):
            return True
        if problem.explain_flag:
            return True
        if problem.domain in (DomainType.MATH, DomainType.SCIENCE, DomainType.CODE):
            return True

        # Check prompt for thinking/reasoning trigger keywords
        prompt_text = problem.user_problem.lower()
        triggers = [
            r"\bthink\b", r"\bthought\b", r"\breason\b", r"\breasoning\b",
            r"\bstep[-\s]by[-\s]step\b", r"\bshow\s+your\s+work\b",
            r"\bderive\b", r"\banalyze\b", r"\bexplain\b",
        ]
        return any(re.search(pat, prompt_text) for pat in triggers)

    # ------------------------------------------------------------------
    # Section 6a: Thinking Tags (<think> ... </think>) Structural Check
    # ------------------------------------------------------------------
    def _score_thinking_tags(self, problem: DatasetRow, traj: Trajectory) -> Tuple[float, Dict[str, Any]]:
        """
        Validates proper <think> ... </think> structural tag adherence.
        Checks:
        - Opening tag <think> and closing tag </think> balance and ordering
        - Non-empty reasoning content inside the thinking block
        - Presence of final answer following the closing </think> tag
        - Absence of leaked or unclosed tags in the final answer
        - Condition-based requirement (rewards proper tags when conditioned,
          penalizes missing tags when thinking was requested, and avoids false
          penalties on simple direct-response prompts).
        """
        full_text = traj.full_text
        audit: Dict[str, Any] = {}
        condition_met = self.is_thinking_required(problem)
        audit["thinking_conditioned"] = condition_met

        open_tag = "<think>"
        close_tag = "</think>"

        open_count = full_text.count(open_tag)
        close_count = full_text.count(close_tag)
        audit["open_tag_count"] = open_count
        audit["close_tag_count"] = close_count

        prompt_opened = open_tag in traj.prompt and traj.prompt.rstrip().endswith(open_tag)

        # 1. Malformed duplicate/multiple tags (hallucinated chat loop)
        if open_count > 1 or close_count > 1:
            audit["error"] = "duplicate_or_nested_tags"
            audit["penalty"] = -0.25
            return -0.25, audit

        # 2. Case: Opened <think> but never closed </think>
        if open_count == 1 and close_count == 0:
            audit["error"] = "unclosed_think_tag"
            audit["penalty"] = -0.25
            return -0.25, audit

        # 3. Case: Orphaned </think> without opening <think>
        if open_count == 0 and close_count == 1:
            if prompt_opened:
                # Prompt prefilled <think>, so model emitting single </think> is valid!
                pos_close = full_text.find(close_tag)
                thought = full_text[:pos_close].strip()
                answer = full_text[pos_close + len(close_tag):].strip()
                if len(thought) >= 5 and len(answer) > 0:
                    audit["valid_prefilled_thinking"] = True
                    audit["reward"] = 0.10
                    return 0.10, audit
                elif len(thought) < 5:
                    audit["error"] = "empty_prefilled_thought"
                    audit["penalty"] = -0.20
                    return -0.20, audit
                else:
                    audit["error"] = "missing_final_answer"
                    audit["penalty"] = -0.20
                    return -0.20, audit
            else:
                audit["error"] = "orphaned_close_tag"
                audit["penalty"] = -0.20
                return -0.20, audit

        # 4. Case: Standard pair (open_count == 1 and close_count == 1)
        if open_count == 1 and close_count == 1:
            pos_open = full_text.find(open_tag)
            pos_close = full_text.find(close_tag)

            if pos_open > pos_close:
                audit["error"] = "inverted_tags_close_before_open"
                audit["penalty"] = -0.25
                return -0.25, audit

            thought = full_text[pos_open + len(open_tag):pos_close].strip()
            answer = full_text[pos_close + len(close_tag):].strip()

            if len(thought) < 5:
                audit["error"] = "empty_thinking_block"
                audit["penalty"] = -0.20
                return -0.20, audit

            if len(answer) == 0:
                audit["error"] = "missing_final_answer"
                audit["penalty"] = -0.20
                return -0.20, audit

            if open_tag in answer or close_tag in answer:
                audit["error"] = "leaked_tags_in_final_answer"
                audit["penalty"] = -0.20
                return -0.20, audit

            audit["valid_thinking_tags"] = True
            reward = 0.10 if condition_met else 0.05
            audit["reward"] = reward
            return reward, audit

        # 5. Case: Neither tag present (open_count == 0 and close_count == 0)
        if condition_met:
            # Thinking was expected by condition, but tags were omitted
            audit["error"] = "missing_required_thinking_tags"
            audit["penalty"] = -0.20
            return -0.20, audit
        else:
            # Simple direct response, no tags expected -> neutral 0.0
            audit["tags_omitted_allowed"] = True
            return 0.0, audit

    # ------------------------------------------------------------------
    # Section 7 & 7a: Code Editing & Destructive Edit Guard
    # ------------------------------------------------------------------
    def _score_code_editing(self, problem: DatasetRow, traj: Trajectory) -> Tuple[float, Dict[str, Any]]:
        text = traj.full_text
        audit: Dict[str, Any] = {}

        # 1. Unified diff format check
        has_diff_markers = "<<<<<<<" in text and "=======" in text and ">>>>>>>" in text
        has_git_diff = "diff --git" in text or "```diff" in text

        if not (has_diff_markers or has_git_diff):
            audit["diff_format"] = False
            return -0.3, audit

        audit["diff_format"] = True
        score = 0.15

        # 2. Scope statement requirement
        # Model must state in its trace which files/functions it intends to change
        stated_scope = []
        scope_match = re.search(r"(?:modify|edit|change|fix)\s+([a-zA-Z0-9_\./\-]+\.(?:py|js|ts|cpp|rs|html|css))", traj.reasoning_trace, re.IGNORECASE)
        if scope_match:
            stated_scope.append(scope_match.group(1))

        # Check declared task_scope against touched files in diff
        touched_files = re.findall(r"(?:---|\+\+\+)\s+[ab]/([a-zA-Z0-9_\./\-]+)", text)
        if problem.task_scope and touched_files:
            out_of_scope = [f for f in touched_files if f not in problem.task_scope]
            if out_of_scope:
                penalty = -0.3
                audit["out_of_scope_files"] = out_of_scope
                score += penalty

        # 3. Destructive-edit deletion ratio check (Section 7a)
        deleted_lines = len(re.findall(r"^-\s.*", text, re.MULTILINE))
        added_lines = len(re.findall(r"^\+\s.*", text, re.MULTILINE))
        total_touched = max(1, deleted_lines + added_lines)

        audit["deleted_lines"] = deleted_lines
        audit["added_lines"] = added_lines

        # Deletion ratio > 60% without explicit delete instruction is heavily penalized
        if deleted_lines > 15 and (deleted_lines / total_touched) > 0.60:
            score -= 0.5
            audit["destructive_violation"] = True
            audit["deletion_ratio_penalty"] = -0.5

        # 4. Verbalized surgical intent correlated with small diff
        verbalized_intent = bool(re.search(r"(?:edit\s+instead\s+of\s+rewrit|surgical\s+edit|minimal\s+change)", traj.reasoning_trace, re.IGNORECASE))
        if verbalized_intent:
            if total_touched <= 25:
                score += 0.10  # Validated surgical intent
                audit["surgical_intent_rewarded"] = True
            else:
                audit["surgical_intent_gaming_blocked"] = True  # Said the phrase but wrote huge diff!

        # 5. Docker Coding Gym ground-truth test execution (if gym attached)
        if self.gym and "gym_instance" in problem.extra_metadata:
            try:
                from sword.gym.schema import GymInstance
                gym_inst = GymInstance.from_dict(problem.extra_metadata["gym_instance"])
                gym_res = self.gym.evaluate_patch(gym_inst, traj.full_text)
                audit["gym_execution"] = gym_res.to_dict()
                if gym_res.success:
                    score += 0.50
                elif not gym_res.patch_applied:
                    score -= 0.30
                else:
                    score += gym_res.reward * 0.30
            except Exception as e:
                audit["gym_execution_error"] = str(e)

        return max(-1.0, min(1.0 if self.gym else 0.35, score)), audit

    # ------------------------------------------------------------------
    # Section 9: UI / Front-end Design Slop Tell-Detection
    # ------------------------------------------------------------------
    def _score_ui_design(self, text: str) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        audit: Dict[str, Any] = {}

        # 1. DOM Nesting depth check
        div_nesting = len(re.findall(r"<div[^>]*>", text, re.IGNORECASE))
        if div_nesting > 12:
            score -= 0.15
            audit["excessive_dom_nesting"] = div_nesting

        # 2. Cliche palette fingerprinting
        found_cliches = [c for c in self.cliche_colors if c in text.lower()]
        if len(found_cliches) >= 2:
            score -= 0.15
            audit["cliche_palette_tells"] = found_cliches

        # 3. Template chrome regex checks
        tells = []
        for pat in self.template_chrome_patterns:
            if re.search(pat, text):
                tells.append(pat)
        if tells:
            score -= 0.15
            audit["template_chrome_tells"] = len(tells)

        # 4. prefers-reduced-motion compliance bonus
        if "prefers-reduced-motion" in text:
            score += 0.15
            audit["prefers_reduced_motion"] = True

        return score, audit

    # ------------------------------------------------------------------
    # Section 10: Multi-Domain Specialized Checks
    # ------------------------------------------------------------------
    def _score_writing(self, text: str) -> Tuple[float, Dict[str, Any]]:
        found_cliches = [p for p in self.writing_cliches if re.search(p, text, re.IGNORECASE)]
        if len(found_cliches) >= 2:
            return -0.25, {"cliches_found": found_cliches}
        return 0.05, {"cliches_found": []}

    def _score_practical(self, answer: str) -> Tuple[float, Dict[str, Any]]:
        # Concrete ordered actionability check
        steps = re.findall(r"^\s*(?:Step\s*\d+|\d+\.)\s+", answer, re.MULTILINE | re.IGNORECASE)
        if len(steps) >= 3:
            return 0.15, {"actionable_steps": len(steps)}
        return -0.10, {"actionable_steps": len(steps)}

    def _score_conversation(self, text: str) -> Tuple[float, Dict[str, Any]]:
        # Not-doing checks: no unsolicited diagnostic claims
        diag_patterns = [
            r"\byou (?:have|suffer from|are diagnosed with) (?:depression|bipolar|adhd|autism|ocd)\b",
            r"\byou should stop taking your medication\b",
        ]
        for pat in diag_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return -1.0, {"unsolicited_diagnosis_violation": True}
        return 0.0, {"clean": True}

    def _score_multi_turn(
        self,
        history: List[Dict[str, str]],
        current_text: str,
    ) -> Tuple[float, Dict[str, Any]]:
        # Constraint persistence check: verify constraints mentioned in early turns aren't contradicted
        constraints: Set[str] = set()
        for turn in history:
            if turn.get("role") == "user":
                found = re.findall(r"(?:do not|don't|must|never)\s+([a-zA-Z0-9_\s]{3,25})", turn.get("content", ""), re.IGNORECASE)
                constraints.update(found)

        violations = []
        for c in constraints:
            negation = f"I am {c}"
            if negation.lower() in current_text.lower():
                violations.append(c)
            elif any(prefix in c.lower() for prefix in ["use", "include", "mention", "have", "add"]):
                term = re.sub(r"^(?:use|include|mention|have|add)\s+", "", c.strip(), flags=re.IGNORECASE)
                sub_terms = re.split(r"\s+(?:or|and)\s+", term)
                for st in sub_terms:
                    st_clean = re.sub(r"\s+(?:in|on|at|for|with)\s+.*$", "", st.strip().lower())
                    if st_clean and len(st_clean) >= 3 and st_clean in current_text.lower():
                        violations.append(st_clean)

        if violations:
            return -0.3, {"constraint_violations": violations}
        return 0.10, {"constraints_preserved": len(constraints)}

    def _score_ground_truth(self, reference: str, answer: str) -> Tuple[float, Dict[str, Any]]:
        ref_clean = reference.strip().lower()
        ans_clean = answer.strip().lower()

        # Direct exact or substring match
        if ref_clean in ans_clean:
            return 0.35, {"matched": True, "exact": True}

        # Numerical comparison if answer is numbers
        ref_nums = re.findall(r"[-+]?\d*\.\d+|\d+", ref_clean)
        ans_nums = re.findall(r"[-+]?\d*\.\d+|\d+", ans_clean)
        if ref_nums and ans_nums and ref_nums[-1] == ans_nums[-1]:
            return 0.35, {"matched": True, "numerical_match": ref_nums[-1]}

        return -0.40, {"matched": False}

    # ------------------------------------------------------------------
    # Section 11a: Music (MIDI / Piano Note Generation)
    # ------------------------------------------------------------------
    def _score_midi(self, constraints: Optional[Dict[str, Any]], text: str) -> Tuple[float, Dict[str, Any]]:
        audit: Dict[str, Any] = {}
        score = 0.0

        # Parse JSON note events or REMI tokens
        json_match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if not json_match:
            return -0.3, {"parse_failed": True}

        try:
            notes = json.loads(json_match.group(0))
        except Exception:
            return -0.3, {"invalid_json_notes": True}

        if not isinstance(notes, list) or len(notes) < 4:
            return -0.3, {"insufficient_notes": len(notes) if isinstance(notes, list) else 0}

        # 1. Playability: hand-span (max 14 semitones between simultaneous notes)
        simultaneous = defaultdict(list)
        for n in notes:
            start = n.get("start", 0)
            pitch = n.get("pitch", 60)
            simultaneous[start].append(pitch)

        unplayable_chords = 0
        for start, chord_pitches in simultaneous.items():
            if len(chord_pitches) > 10:  # Max 10 fingers
                unplayable_chords += 1
            elif chord_pitches:
                span = max(chord_pitches) - min(chord_pitches)
                if span > 16:  # Exceeds physical hand reach without pedal
                    unplayable_chords += 1

        if unplayable_chords == 0:
            score += 0.15
            audit["playability_passed"] = True
        else:
            score -= 0.20
            audit["unplayable_chords"] = unplayable_chords

        # 2. Duration math against time signature (e.g. 4/4 = 4 beats)
        if constraints and "time_signature" in constraints:
            ts = constraints["time_signature"]  # e.g. "4/4"
            try:
                num, den = map(int, ts.split("/"))
                measure_len = num * (4 / den)
                total_duration = max(n.get("start", 0) + n.get("duration", 0) for n in notes)
                if math.isclose(total_duration % measure_len, 0, abs_tol=0.25):
                    score += 0.15
                    audit["duration_math_passed"] = True
                else:
                    audit["duration_math_mismatch"] = total_duration % measure_len
            except Exception:
                pass

        return score, audit
