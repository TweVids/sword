"""Live Model B scores must remain authoritative for math tasks."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sword.b_model import ModelBJudge, ModelBJudgeConfig
from sword.reward import MathVerifier, MultiMetricRewardScorer


@pytest.mark.parametrize("candidate,reference,truth,reasoning,answer", [
    ("Final Answer: twenty-five", "25", 1.0, 1.0, 1.0),
    ("Final Answer: 4 kilograms", "4", 0.2, 0.5, 0.2),
    ("Final Answer: 4", "4", 1.0, 0.25, 1.0),
])
def test_math_reward_uses_actual_http_judge_scores_and_preserves_reasoning(candidate, reference, truth, reasoning, answer):
    scores = {"truth": truth, "formatted": 1.0, "rule_reasoning_follow": reasoning,
              "rule_answer_follow": answer, "tools_success": 0.0, "cheating": False}
    history = ["<think>Compute the result once.\n" + "Repeat the same calculation.\n" * 100 + "</think>"]
    requests_seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests_seen.append(payload)
            body = json.dumps({"message": {"content": json.dumps(scores)}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    judge = ModelBJudge(ModelBJudgeConfig(
        model_name="controlled-math-judge", port=server.server_port, reasoning=False,
        auto_setup=False, auto_unload=False, force_gpu=False, fallback_model_name=None,
    ))
    judge.ensure_ready = lambda: True
    try:
        scorer = MultiMetricRewardScorer(judge)
        reward = scorer.score(
            "Solve this math problem.", reference, candidate,
            reasoning_rule="Show only necessary calculation; avoid repeated verification.",
            answer_rule="Give the correct value and requested units.",
            category="math", harness_name="none", assistant_history=history,
        )
        assert len(requests_seen) == 1
        assert reward.truth == truth and reward.rule_reasoning_follow == reasoning
        assert reward.rule_answer_follow == answer
        assert reward.total_score == pytest.approx((truth + 1.0 + reasoning + answer) / 4)
        assert reward.judge_source == "controlled-math-judge" and reward.judge_scores == scores
        evidence = requests_seen[0]["messages"][1]["content"]
        assert candidate in evidence and json.dumps(history) in evidence
        assert "avoid repeated verification" in evidence
        # These cases deliberately disagree with exact-number/string authority.
        exact_truth = 0.5 if MathVerifier.verify_answer(candidate, reference) else 0.0
        assert reward.truth != exact_truth
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("enabled", [False, True])
def test_math_uses_exact_fallback_when_judge_is_disabled_or_offline(enabled):
    judge = ModelBJudge(ModelBJudgeConfig(auto_setup=False, log_model_outputs=False)) if enabled else None
    if judge:
        judge._memory_ready = False
    scorer = MultiMetricRewardScorer(judge)
    reward = scorer.score("2 + 2?", "4", "Final Answer: 4", category="math")
    assert reward.truth == 1.0
    assert reward.judge_source == ("deterministic" if enabled else "disabled")


def test_math_judge_rubric_rewards_rule_compliance_without_rewarding_token_volume():
    guide = ModelBJudge.SYSTEM_PROMPT
    assert "Correctness is independent of reasoning length" in guide
    assert "repeated calculations earn no extra credit" in guide
    assert "When the rule requires concision" in guide
    assert "detailed derivation explicitly requested by the rule" in guide
