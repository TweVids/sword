"""Task scheduling, category accounting, and approval replay without model inference."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sword.task_queue import TaskRetryQueue


def problem_rows(count=1):
    return [SimpleNamespace(row_hash=hashlib.sha256(str(index).encode()).hexdigest()[:16],
                            problem_id=f"task-{index}", problem=f"Problem {index}",
                            category="math" if index % 2 == 0 else "code") for index in range(count)]


def outcome(queue, solved=False, step=None, solved_rollouts=None):
    key = queue.next_key()
    return queue.record_result(key, solved_rollouts=int(solved) if solved_rollouts is None else solved_rollouts,
                               rollout_count=2, best_truth=float(solved),
                               step=queue.dispatch_count + 1 if step is None else step)


class TestTaskRetryQueue(unittest.TestCase):
    def test_three_failures_move_to_n_plus_100_then_waitlist(self):
        rows = problem_rows(110)
        queue = TaskRetryQueue(rows)
        for _ in range(3):
            self.assertEqual(queue.next_key(), rows[0].row_hash)
            outcome(queue)
        current = queue.monitor_snapshot(rows[0].row_hash)["current_task"]
        self.assertEqual(current["status"], "deferred")
        self.assertEqual(current["next_position"], 103)
        for index in range(1, 100):
            self.assertEqual(queue.next_key(), rows[index].row_hash)
            outcome(queue, solved=True)
        self.assertEqual(queue.next_key(), rows[0].row_hash)
        final = outcome(queue)
        self.assertEqual(final["last_step"], 103)
        self.assertEqual(final["status"], "waiting_approval")
        self.assertEqual(final["failures"], 4)
        self.assertNotIn(rows[0].row_hash, queue.export_state()["queue"])

    def test_short_queue_defers_to_tail_without_idle_steps(self):
        rows = problem_rows(2)
        queue = TaskRetryQueue(rows)
        for _ in range(3):
            outcome(queue)
        self.assertEqual(queue.next_key(), rows[1].row_hash)
        outcome(queue, solved=True)
        self.assertEqual(queue.next_key(), rows[0].row_hash)
        self.assertEqual(outcome(queue)["status"], "waiting_approval")
        self.assertEqual(queue.dispatch_count, 5)

    def test_one_task_waits_after_four_failures(self):
        queue = TaskRetryQueue(problem_rows())
        for _ in range(4):
            outcome(queue)
        self.assertIsNone(queue.next_key())
        self.assertEqual(queue.monitor_snapshot()["summary"]["waiting_approval"], 1)

    def test_approval_starts_a_new_cycle_and_preserves_lifetime_failures(self):
        rows = problem_rows()
        queue = TaskRetryQueue(rows)
        for _ in range(4):
            outcome(queue)
        self.assertTrue(queue.decide(rows[0].row_hash, "approve"))
        task = queue.export_state()["tasks"][rows[0].row_hash]
        self.assertEqual(task["failures"], 4)
        self.assertEqual(task["consecutive_failures"], 0)
        self.assertFalse(task["solved"])
        self.assertTrue(queue.decision_event.is_set())
        self.assertFalse(queue.decide(rows[0].row_hash, "approve"))
        for _ in range(4):
            outcome(queue)
        task = queue.export_state()["tasks"][rows[0].row_hash]
        self.assertEqual(task["failures"], 8)
        self.assertEqual(task["deferrals"], 2)
        self.assertTrue(queue.decide(rows[0].row_hash, "reject"))
        self.assertEqual(queue.monitor_snapshot()["summary"]["removed"], 1)
        self.assertIsNone(queue.next_key())

    def test_remove_policy_never_creates_a_waitlist(self):
        queue = TaskRetryQueue(problem_rows(), failure_action="remove")
        for _ in range(4):
            outcome(queue)
        summary = queue.monitor_snapshot()["summary"]
        self.assertEqual(summary["removed"], 1)
        self.assertEqual(summary["waiting_approval"], 0)
        self.assertEqual(summary["unsolved"], 1)

    def test_category_totals_count_unique_tasks_and_all_failed_rollouts(self):
        queue = TaskRetryQueue(problem_rows(3))
        outcome(queue, solved=True, solved_rollouts=1)
        outcome(queue)
        snapshot = queue.monitor_snapshot()
        self.assertEqual(snapshot["summary"]["total"], 3)
        self.assertEqual(snapshot["summary"]["solved"], 1)
        self.assertEqual(snapshot["summary"]["unsolved"], 2)
        self.assertEqual(snapshot["summary"]["failed_attempts"], 1)
        self.assertEqual(snapshot["summary"]["failed_rollouts"], 3)
        categories = {row["category"]: row for row in snapshot["categories"]}
        self.assertEqual(categories["math"]["total"], 2)
        self.assertEqual(categories["math"]["solve_rate"], 50)
        self.assertEqual(categories["code"]["failed_attempts"], 1)
        outcome(queue, solved=True)
        self.assertEqual(queue.monitor_snapshot()["summary"]["solved"], 2)

    def test_solve_counts_reflect_latest_outcome(self):
        queue = TaskRetryQueue(problem_rows())
        outcome(queue, solved=True)
        self.assertEqual(queue.monitor_snapshot()["summary"]["solved"], 1)
        outcome(queue)
        self.assertEqual(queue.monitor_snapshot()["summary"]["solved"], 0)
        self.assertEqual(queue.monitor_snapshot()["summary"]["unsolved"], 1)

    def test_resume_preserves_queue_order_failures_and_categories(self):
        rows = problem_rows(5)
        queue = TaskRetryQueue(rows, retry_delay=4)
        for _ in range(3):
            outcome(queue)
        outcome(queue, solved=True)
        state = json.loads(json.dumps(queue.export_state()))
        restored = TaskRetryQueue(rows, state=state)
        self.assertEqual(restored.export_state(), state)
        self.assertEqual(restored.monitor_snapshot(), queue.monitor_snapshot())
        self.assertEqual(restored.config["retry_delay"], 4)

    def test_decisions_after_checkpoint_replay_once_and_ignore_future_attempts(self):
        rows = problem_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "decisions.jsonl")
            queue = TaskRetryQueue(rows, decisions_path=path)
            for _ in range(3):
                outcome(queue)
            older = queue.export_state()
            outcome(queue)
            saved = queue.export_state()
            queue.decide(rows[0].row_hash, "approve")
            restored = TaskRetryQueue(rows, state=saved, decisions_path=path)
            self.assertEqual(restored.next_key(), rows[0].row_hash)
            self.assertEqual(restored.decision_revision, 1)
            repeated = TaskRetryQueue(rows, state=restored.export_state(), decisions_path=path)
            self.assertEqual(repeated.decision_revision, 1)
            self.assertEqual(len(repeated.export_state()["queue"]), 1)
            rollback = TaskRetryQueue(rows, state=older, decisions_path=path)
            self.assertEqual(rollback.decision_revision, 0)
            self.assertEqual(rollback.export_state()["tasks"][rows[0].row_hash]["status"], "deferred")

    def test_duplicate_display_ids_are_independent_tasks(self):
        rows = problem_rows(2)
        rows[1].problem_id = rows[0].problem_id
        queue = TaskRetryQueue(rows, retry_delay=1)
        for _ in range(4):
            outcome(queue)
        self.assertTrue(queue.decide(rows[0].row_hash, "reject"))
        self.assertEqual(queue.next_key(), rows[1].row_hash)
        self.assertEqual(queue.monitor_snapshot()["summary"]["total"], 2)

    def test_another_runs_approval_does_not_apply_to_same_task_and_step(self):
        rows = problem_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "decisions.jsonl")
            old = TaskRetryQueue(rows, decisions_path=path)
            for _ in range(4):
                outcome(old)
            old.decide(rows[0].row_hash, "approve")
            fresh = TaskRetryQueue(rows, decisions_path=path)
            for _ in range(4):
                outcome(fresh)
            restored = TaskRetryQueue(rows, state=fresh.export_state(), decisions_path=path)
            self.assertIsNone(restored.next_key())
            self.assertEqual(restored.decision_revision, 0)

    def test_display_limit_does_not_limit_category_totals(self):
        queue = TaskRetryQueue(problem_rows(2500))
        snapshot = queue.monitor_snapshot(limit=3)
        self.assertEqual(len(snapshot["tasks"]), 3)
        self.assertEqual(snapshot["summary"]["total"], 2500)
        self.assertEqual(sum(row["total"] for row in snapshot["categories"]), 2500)
        self.assertEqual([task["next_position"] for task in snapshot["tasks"]], [1, 2, 3])

    def test_previously_completed_tasks_stay_in_category_totals(self):
        rows = problem_rows(2)
        queue = TaskRetryQueue(rows, completed_hashes=[rows[0].row_hash])
        self.assertEqual(queue.next_key(), rows[1].row_hash)
        self.assertEqual(queue.monitor_snapshot()["summary"]["solved"], 1)

    def test_disabled_retry_scheduler_cycles_in_dataset_order(self):
        rows = problem_rows(2)
        queue = TaskRetryQueue(rows, enabled=False)
        for index in range(10):
            self.assertEqual(queue.next_key(), rows[index % 2].row_hash)
            outcome(queue)
        self.assertEqual(queue.monitor_snapshot()["summary"]["waiting_approval"], 0)

    def test_invalid_restored_queue_and_configuration_are_rejected(self):
        rows = problem_rows(2)
        for options in ({"retry_limit": 0}, {"retry_delay": -1}, {"failure_action": "unknown"}):
            with self.assertRaises(ValueError):
                TaskRetryQueue(rows, **options)
        state = TaskRetryQueue(rows).export_state()
        for change in (lambda value: value["queue"].append(value["queue"][0]),
                       lambda value: value["queue"].pop(),
                       lambda value: value["tasks"].pop(rows[0].row_hash)):
            invalid = copy.deepcopy(state)
            change(invalid)
            with self.assertRaises(ValueError):
                TaskRetryQueue(rows, state=invalid)


if __name__ == "__main__":
    unittest.main()
