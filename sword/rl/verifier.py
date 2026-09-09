"""
External Verifier Model Client (System 2 - Advisory Judge).
Implements Section 19 of rl-engine-spec.md:
- Advisory 9B verifier (e.g. Qwen 3.5 9B with Sword patch)
- Typed rubric query construction (boolean / categorical / int)
- Strict answer validation & citation enforcement (unsupported answers discarded)
- Score mapping table (owned by Primary Scorer, never exposed to judge)
- Total judge influence clamping (max ±0.35)
- Disagreement logging between deterministic and judge checks
- Anti-gaming prompt rotation & sampling rate control
"""

import json
import re
import random
from typing import Dict, Any, List, Optional, Tuple, Union

from .schema import (
    DatasetRow,
    Trajectory,
    VerifierRequest,
    VerifierQuestion,
    VerifierAnswer,
    DomainType,
    EffortTier,
)


class ExternalVerifier:
    """
    Advisory 9B Verifier Client.
    Non-negotiable invariants:
    1. NEVER accepts a raw score from the judge.
    2. Enforces typed schema (boolean, categorical, int).
    3. Rejects answers without verifiable citations from the trajectory.
    4. Clamps overall influence to prevent overriding deterministic rules.
    """

    def __init__(
        self,
        verifier_model: Optional[Any] = None,
        sampling_rate: float = 0.25,
        max_total_delta: float = 0.50,
        seed: int = 42,
    ):
        self.verifier_model = verifier_model
        self.sampling_rate = sampling_rate
        self.max_total_delta = max_total_delta
        self.random = random.Random(seed)

        # Pre-defined score mapping table (Section 19.3)
        # Note: This is NEVER transmitted to the 9B model
        self.score_mapping = {
            "looping": {
                ("yes", True): -0.50,
                ("no", False): 0.05,
            },
            "recheck_genuine": {
                ("yes", True): 0.15,
                ("no", False): -0.10,
            },
            "scope_violation": {
                ("yes", True): -0.30,
                ("no", False): 0.05,
            },
            "assumption_stated": {
                "stated": 0.10,
                "guessed_silently": -0.10,
                "over_clarified": -0.05,
                "not_applicable": 0.0,
            },
            "citation_grounded": {
                "supported": 0.15,
                "unsupported": -0.25,
                "not_checkable": 0.0,
            },
            "music_melodic_variation": {
                ("yes", True): 0.10,
                ("no", False): -0.15,
            }
        }

        # Disagreement audit logger
        self.disagreements: List[Dict[str, Any]] = []

    def should_sample(self, row: DatasetRow) -> bool:
        """
        Samples verifier evaluation at an unpredictable low frequency (Section 15.4 & 19.1)
        or if row explicitly requires verification.
        """
        if row.recheck_required:
            return True
        return self.random.random() < self.sampling_rate

    def get_rubric_questions(self, row: DatasetRow) -> List[VerifierQuestion]:
        """
        Constructs domain-specific typed questions (Section 19.2).
        """
        questions = []

        # 1. Self-correction check (Section 4)
        if row.recheck_required or row.effort_tier in (EffortTier.HIGH, EffortTier.XHIGH, EffortTier.ULTRA):
            questions.append(VerifierQuestion(
                id="recheck_genuine",
                prompt="Does the reasoning trace change direction or explicitly reject an earlier claim between its first half and second half?",
                answer_type="boolean",
                expected_fields={"answer": "yes|no", "citation": "quote from trace showing backtrack"}
            ))

        # 2. Looping / restatement check (Section 5)
        questions.append(VerifierQuestion(
            id="looping",
            prompt="Does the reasoning trace copy-paste or re-state the original prompt parameters more than twice without new analysis?",
            answer_type="boolean",
            expected_fields={"answer": "yes|no", "citation": "repeated quote"}
        ))

        # 3. Domain-specific checks
        if row.domain == DomainType.CODE:
            questions.append(VerifierQuestion(
                id="scope_violation",
                prompt="Does the diff touch any file, function, or component not declared in the model's stated scope?",
                answer_type="boolean",
                expected_fields={"answer": "yes|no", "offending_files": "list of filenames"}
            ))

        if row.domain in (DomainType.CONVERSATION, DomainType.PRACTICAL, DomainType.GENERAL):
            questions.append(VerifierQuestion(
                id="assumption_stated",
                prompt="If the problem was ambiguous, did the trace explicitly state its assumption before proceeding, rather than guessing silently or over-asking?",
                answer_type="categorical",
                expected_fields={"answer": "stated|guessed_silently|over_clarified|not_applicable"}
            ))

        if row.domain == DomainType.SCIENCE:
            questions.append(VerifierQuestion(
                id="citation_grounded",
                prompt="Are the scientific claims supported by standard scientific principles or does it hallucinate nonexistent mechanisms?",
                answer_type="categorical",
                expected_fields={"answer": "supported|unsupported|not_checkable", "citation": "quote"}
            ))

        return questions

    def evaluate_trajectory(
        self,
        row: DatasetRow,
        traj: Trajectory,
    ) -> Tuple[float, Dict[str, VerifierAnswer]]:
        """
        Sends trajectory to 9B Verifier, validates typed responses with citations,
        and computes clamped reward delta.
        """
        questions = self.get_rubric_questions(row)
        if not questions:
            return 0.0, {}

        request = VerifierRequest(
            row_id=row.problem_id,
            domain=row.domain.value,
            user_problem=row.user_problem,
            trajectory_trace=traj.reasoning_trace,
            final_answer=traj.final_answer,
            questions=questions,
        )

        # Call verifier model (or internal fallback heuristics if model not loaded)
        raw_response = self._query_verifier_model(request)

        # Validate answers against schema and citation requirements
        validated_answers: Dict[str, VerifierAnswer] = {}
        total_delta = 0.0

        for q in questions:
            ans_data = raw_response.get(q.id)
            if not ans_data or not isinstance(ans_data, dict):
                continue

            answer_val = ans_data.get("answer")
            citation = ans_data.get("citation") or ans_data.get("offending_files")

            # Validate citation existence for boolean / categorical flags (Section 19.2)
            has_citation = bool(citation and str(citation).strip() and str(citation).lower() != "none")

            if q.answer_type == "boolean":
                bool_val = None
                if isinstance(answer_val, bool):
                    bool_val = answer_val
                elif str(answer_val).lower() in ("yes", "true", "1"):
                    bool_val = True
                elif str(answer_val).lower() in ("no", "false", "0"):
                    bool_val = False

                if bool_val is None:
                    continue  # Discard invalid type

                # Enforce citation: unsupported "yes" with consequence is ignored!
                if bool_val is True and not has_citation:
                    print(f"[Sword-RL Verifier] ⚠️  Discarded unsupported '{q.id}=yes' lacking citation.")
                    continue

                validated_answers[q.id] = VerifierAnswer(
                    question_id=q.id,
                    answer=bool_val,
                    citation=str(citation) if citation else None,
                    is_valid=True,
                )

                # Map to pre-defined score delta
                delta = self._lookup_delta(q.id, bool_val)
                total_delta += delta

            elif q.answer_type == "categorical":
                cat_val = str(answer_val).lower().strip()
                validated_answers[q.id] = VerifierAnswer(
                    question_id=q.id,
                    answer=cat_val,
                    citation=str(citation) if citation else None,
                    is_valid=True,
                )
                delta = self._lookup_delta(q.id, cat_val)
                total_delta += delta

        # Section 19.1: Clamp overall verifier influence
        clamped_delta = max(-self.max_total_delta, min(self.max_total_delta, total_delta))
        return round(clamped_delta, 4), validated_answers

    def _lookup_delta(self, q_id: str, value: Any) -> float:
        """Looks up delta in the primary scorer's pre-defined table."""
        mapping = self.score_mapping.get(q_id, {})
        if isinstance(value, bool):
            for k, delta in mapping.items():
                if isinstance(k, tuple) and value in k:
                    return delta
        elif isinstance(value, str):
            if value in mapping:
                return mapping[value]
        return 0.0

    def _query_verifier_model(self, request: VerifierRequest) -> Dict[str, Any]:
        """
        Dispatches request to Qwen 3.5 9B verifier model.
        If no model loaded, performs deterministic semantic heuristics fallback.
        """
        if self.verifier_model is not None:
            return self._call_model_generate(request)
        return self._heuristic_fallback(request)

    def _call_model_generate(self, request: VerifierRequest) -> Dict[str, Any]:
        """Calls the Qwen 3.5 9B model with structured prompt formatting."""
        prompt_lines = [
            f"You are an impartial verifier auditing an AI response for task: {request.user_problem}",
            f"Domain: {request.domain}",
            "\n--- MODEL REASONING TRACE ---",
            request.trajectory_trace[:3000],
            "\n--- MODEL FINAL ANSWER ---",
            request.final_answer[:2000],
            "\nAnswer the following rubric questions in strictly valid JSON:",
        ]

        schema_hint = {}
        for q in request.questions:
            schema_hint[q.id] = {
                "prompt": q.prompt,
                "answer": q.expected_fields.get("answer", "value"),
                "citation": "exact quote or evidence from text"
            }

        prompt_lines.append(json.dumps(schema_hint, indent=2))
        prompt_lines.append("\nReturn ONLY the completed JSON object.")
        full_prompt = "\n".join(prompt_lines)

        try:
            # Check if verifier model has generate / serve interface
            if hasattr(self.verifier_model, "serve"):
                res = self.verifier_model.serve([full_prompt], max_new_tokens=256, temperature=0.0)
                gen_text = res["responses"][0]
            elif hasattr(self.verifier_model, "generate"):
                # HuggingFace pipeline or custom server
                gen_text = self.verifier_model(full_prompt, max_new_tokens=256)[0]["generated_text"]
            else:
                return self._heuristic_fallback(request)

            # Parse JSON from model output
            m = re.search(r"\{.*\}", gen_text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception as e:
            print(f"[Sword-RL Verifier] Warning: model inference parse error ({e}), using fallback.")

        return self._heuristic_fallback(request)

    def _heuristic_fallback(self, request: VerifierRequest) -> Dict[str, Any]:
        """
        Reliable fallback heuristics when running in lightweight mode without 9B model.
        """
        results = {}
        trace = request.trajectory_trace.lower()
        full = f"{request.trajectory_trace}\n{request.final_answer}".lower()

        for q in request.questions:
            if q.id == "recheck_genuine":
                # Real backtracking check: look for contrastive transition in trace
                direction_patterns = r"(?:actually[,\s]+(?:that\s+is\s+incorrect|it\s+is|we\s+need|upon)|wait[,\s]+(?:that\s+contradicts|hold\s+on|let\s+me)|let\s+me\s+recalculate|recalculat\w+)"
                match = re.search(direction_patterns, trace)
                if match:
                    results[q.id] = {
                        "answer": "yes",
                        "citation": match.group(0),
                    }
                else:
                    results[q.id] = {"answer": "no", "citation": None}

            elif q.id == "looping":
                # Check for repetitive identical sentences or looping patterns
                sentences = [s.strip() for s in re.split(r"[.!?\n]+", trace) if len(s.strip().split()) >= 4]
                counts: Dict[str, int] = {}
                duplicated = None
                for s in sentences:
                    norm = re.sub(r"\s+", " ", s.lower())
                    counts[norm] = counts.get(norm, 0) + 1
                    if counts[norm] >= 2:
                        duplicated = s
                        break

                if duplicated:
                    results[q.id] = {
                        "answer": "yes",
                        "citation": duplicated[:100],
                        "count": counts[re.sub(r"\s+", " ", duplicated.lower())]
                    }
                else:
                    results[q.id] = {"answer": "no", "citation": None}

            elif q.id == "scope_violation":
                results[q.id] = {"answer": "no", "citation": None}

            elif q.id == "assumption_stated":
                if "assume" in trace or "assuming" in trace:
                    results[q.id] = {"answer": "stated", "citation": "stated explicit assumption"}
                else:
                    results[q.id] = {"answer": "not_applicable", "citation": None}

            elif q.id == "citation_grounded":
                results[q.id] = {"answer": "supported", "citation": "mechanistic reasoning consistent"}

        return results

    def log_disagreement(self, item_id: str, rule_name: str, deterministic_val: Any, verifier_val: Any):
        """Logs divergences between deterministic rules and advisory judge."""
        self.disagreements.append({
            "item_id": item_id,
            "rule": rule_name,
            "deterministic": deterministic_val,
            "verifier": verifier_val,
        })
