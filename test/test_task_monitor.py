"""Monitor counts and Gradio review callbacks, without launching a web server."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sword.task_queue import TaskRetryQueue
from sword.training_logger import TrainingLogger
from test.test_task_queue import outcome, problem_rows


class TestTaskMonitor(unittest.TestCase):
    def queue_and_logger(self):
        rows = problem_rows()
        queue = TaskRetryQueue(rows)
        logger = TrainingLogger(max_rollouts=2)
        logger.attach_progress(queue, SimpleNamespace(upload_info=lambda: {
            "enabled": True, "last_saved_step": 4, "last_uploaded_step": 3, "upload_error": "offline"}), save_steps=25)
        logger._apply("task", {"step": 4, "problem_id": rows[0].problem_id, "row_hash": rows[0].row_hash,
                               "problem": rows[0].problem, "category": rows[0].category, "rollout_count": 8})
        return queue, logger, rows[0].row_hash

    def test_counts_survive_dropped_generation_events(self):
        queue, logger, key = self.queue_and_logger()
        for _ in range(4):
            outcome(queue)
        for _ in range(500):
            logger.generation(0, "preview")
        state = logger.snapshot()
        self.assertEqual(state["task_progress"]["current_task"]["failures"], 4)
        self.assertEqual(state["task_progress"]["summary"]["failed_rollouts"], 8)
        self.assertEqual(state["checkpoint_upload"]["save_every_steps"], 25)

    def test_resumed_score_totals_include_subsequent_rollouts(self):
        queue, logger, _ = self.queue_and_logger()
        logger.attach_progress(queue, logger.checkpoint_manager, save_steps=10, score_sum=10.0, score_count=20)
        logger.publish("step", step=6, scores=[1.0, 0.0])
        event, data = logger.events.get_nowait()
        logger._apply(event, data)
        self.assertEqual(logger.snapshot()["scored_rollouts"], 22)
        self.assertAlmostEqual(logger.snapshot()["overall_mean"], 0.5)

    def test_dashboard_displays_categories_upload_failure_and_review_controls(self):
        queue, logger, key = self.queue_and_logger()
        for _ in range(4):
            outcome(queue)
        callbacks, clicks, launch_options = [], {}, {}
        code_execution = {"source": "print(input())", "cases": [{"input": "a", "output": "a", "stdout": "a\n", "success": True}]}
        logger._apply("score", {"rollout": 0, "scores": {"truth": 1}, "code_execution": code_execution})

        class Component:
            def __init__(self, *args, **kwargs):
                self.label = args[0] if args else kwargs.get("label")
                self.options = kwargs
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def tick(self, function, **kwargs):
                callbacks.append(function)
            load = change = tick
            def click(self, function, **kwargs):
                clicks[self.label] = function
            def launch(self, **kwargs):
                launch_options.update(kwargs)
                return None, "http://localhost:7860", "https://example.gradio.live"

        gradio = SimpleNamespace(**{name: Component for name in (
            "Blocks", "Markdown", "Textbox", "JSON", "Tab", "Accordion", "Dataframe", "LinePlot", "Dropdown", "Timer", "Button")})
        pandas = SimpleNamespace(DataFrame=lambda rows, **kwargs: list(rows))
        with patch.dict(sys.modules, {"gradio": gradio, "pandas": pandas}):
            logger._launch()
            values = callbacks[0](0, key)
            self.assertEqual(values[3], "4")
            self.assertEqual(values[5]["last_uploaded_step"], 3)
            self.assertEqual(values[5]["upload_error"], "offline")
            self.assertEqual(values[6][0][:5], ["math", 1, 0, 1, 0.0])
            self.assertEqual(values[8][0][2], 4)
            self.assertEqual(values[9].options["value"], key)
            self.assertEqual(values[-1], code_execution)
            self.assertIn("Retry approved", clicks["Approve retry"](key))
            self.assertFalse(queue.export_state()["tasks"][key]["solved"])
            self.assertEqual(callbacks[0](0, key)[8], [])
        self.assertTrue(launch_options["share"])
        self.assertIsNone(logger.error)

    def test_rejection_removes_only_selected_task(self):
        queue, logger, key = self.queue_and_logger()
        for _ in range(4):
            outcome(queue)
        self.assertIn("removed", logger.review_task(key, "reject"))
        self.assertIsNone(queue.next_key())
        self.assertEqual(logger.snapshot()["task_progress"]["summary"]["removed"], 1)
        self.assertIn("no longer", logger.review_task(key, "approve"))

    def test_approval_after_training_finishes_explains_resume(self):
        queue, logger, key = self.queue_and_logger()
        for _ in range(4):
            outcome(queue)
        logger.finish("Waiting for task approval")
        self.assertIn("resume_from_checkpoint='latest'", logger.review_task(key, "approve"))

    def test_no_selection_and_failed_decision_write_do_not_change_queue(self):
        queue, logger, key = self.queue_and_logger()
        for _ in range(4):
            outcome(queue)
        self.assertIn("Select a task", logger.review_task(None, "approve"))
        queue.decisions_path = "missing-directory/decisions.jsonl"
        with patch("sword.task_queue.open", side_effect=OSError("disk full"), create=True):
            self.assertIn("Could not save", logger.review_task(key, "approve"))
        self.assertIsNone(queue.next_key())

    def test_missing_gradio_finishes_launch_without_stopping_monitor_worker(self):
        logger = TrainingLogger()
        with patch.dict(sys.modules, {"gradio": None}):
            logger.start()
            self.assertTrue(logger._launch_complete.wait(2))
        self.assertTrue(logger.error)
        logger.finish("Completed")
        logger._worker.join(2)
        self.assertEqual(logger.snapshot()["status"], "Completed")


if __name__ == "__main__":
    unittest.main()
