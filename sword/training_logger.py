"""Optional Gradio monitoring and task review with bounded producer updates."""
import copy
import queue
import threading
import time
from collections import deque
from contextlib import contextmanager


class TrainingLogger:
    def __init__(self, *, max_rollouts=8, refresh_interval=1.0, server_port=None):
        self.max_rollouts = max(1, min(32, int(max_rollouts)))
        self.refresh_interval = max(0.5, float(refresh_interval))
        self.server_port = server_port
        self.events = queue.Queue(maxsize=128)
        self.lock = threading.Lock()
        self.state = {"status": "Starting", "step": 0, "problem_id": "", "problem": "", "row_hash": "", "category": "",
                      "rollout_count": 0, "rollouts": {}, "scores": {}, "mean_score": None,
                      "task_total": None, "overall_mean": None, "scored_rollouts": 0,
                      "score_sum": 0.0, "history": deque(maxlen=1000)}
        self.local_url = self.share_url = None
        self.error = None
        self.app = None
        self._finished = threading.Event()
        self._worker = None
        self._score_sum = 0.0
        self._score_count = 0
        self.task_queue = self.checkpoint_manager = None
        self.save_steps = None
        self._launch_complete = threading.Event()

    def start(self):
        self._worker = threading.Thread(target=self._consume, name="sword-logger-events", daemon=True)
        self._worker.start()
        def launch():
            try:
                self._launch()
            finally:
                self._launch_complete.set()
        threading.Thread(target=launch, name="sword-gradio", daemon=True).start()
        return self

    def attach_progress(self, task_queue, checkpoint_manager, *, save_steps, score_sum=0.0, score_count=0):
        self.task_queue, self.checkpoint_manager, self.save_steps = task_queue, checkpoint_manager, save_steps
        self._score_sum, self._score_count = score_sum, score_count
        with self.lock:
            self.state.update(score_sum=score_sum, scored_rollouts=score_count,
                              overall_mean=score_sum / score_count if score_count else None)

    def review_task(self, key, action):
        if not key:
            return "Select a task from the waitlist."
        if self.task_queue is None:
            return "Task review is unavailable."
        try:
            changed = self.task_queue.decide(key, action)
        except Exception as exc:
            return f"Could not save the task decision: {exc}"
        if not changed:
            return "This task is no longer waiting for approval."
        if action == "reject":
            return "Task removed from the active queue."
        if self._finished.is_set():
            return "Retry approved. Resume training with resume_from_checkpoint='latest' to continue."
        return "Retry approved; training will pick up this task."

    def publish(self, event, **data):
        """Only small CPU snapshots cross this queue; tensors never reach the UI."""
        if event in ("generation", "score") and data.get("rollout", 0) >= self.max_rollouts:
            return
        if event == "step":
            self._score_sum += sum(data["scores"])
            self._score_count += len(data["scores"])
            data.update(score_sum=self._score_sum, score_count=self._score_count)
        try:
            self.events.put_nowait((event, data))
        except queue.Full:
            # Drop stale UI updates instead of waiting on a browser or worker.
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait((event, data))
            except queue.Full:
                pass

    def generation(self, rollout, text, *, finished=False, turn=0):
        self.publish("generation", rollout=rollout, text=text[-32768:], finished=finished, turn=turn)

    def _consume(self):
        while not self._finished.is_set() or not self.events.empty():
            try:
                event, data = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            self._apply(event, data)

    def _apply(self, event, data):
        with self.lock:
            if event == "task":
                self.state.update(status="Generating", step=data["step"], problem_id=data["problem_id"],
                                  problem=data["problem"][:16000], rollout_count=data["rollout_count"],
                                  row_hash=data.get("row_hash", ""), category=data.get("category", ""),
                                  rollouts={}, scores={}, mean_score=None, task_total=None)
            elif event == "generation":
                current = self.state["rollouts"].setdefault(data["rollout"], {"turns": {}})
                current["turns"][data["turn"]] = data["text"]
                current.update(finished=data["finished"], turn=data["turn"])
                current["text"] = "\n\n".join(f"Turn {turn + 1}\n{text}" for turn, text in sorted(current["turns"].items()))[-32768:]
                while len(current["turns"]) > 8:
                    del current["turns"][min(current["turns"])]
            elif event == "score":
                self.state["status"] = "Scoring"
                self.state["scores"][data["rollout"]] = dict(data)
            elif event == "step":
                scores = data["scores"]
                total = sum(scores)
                mean = total / max(1, len(scores))
                self.state["score_sum"] = data["score_sum"]
                self.state["scored_rollouts"] = data["score_count"]
                self.state.update(status="Training", task_total=total, mean_score=mean,
                                  overall_mean=self.state["score_sum"] / max(1, self.state["scored_rollouts"]))
                self.state["history"].append({"step": data["step"], "problem_id": self.state["problem_id"],
                                              "mean_score": mean, "total_score": total})
            elif event == "status":
                self.state["status"] = data["status"]

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.state)
        state["history"] = list(state["history"])
        if self.task_queue is not None:
            state["task_progress"] = self.task_queue.monitor_snapshot(state["row_hash"])
        if self.checkpoint_manager is not None:
            state["checkpoint_upload"] = {"save_every_steps": self.save_steps, **self.checkpoint_manager.upload_info()}
        return state

    def finish(self, status):
        self.publish("status", status=status)
        self._finished.set()

    def info(self):
        return {"enabled": True, "local_url": self.local_url, "public_url": self.share_url,
                "error": self.error, "max_rollouts": self.max_rollouts}

    def _launch(self):
        try:
            import gradio as gr
            import pandas as pd

            with gr.Blocks(title="Sword training", css=".gradio-container {max-width: 1400px !important;}") as app:
                gr.Markdown("# Sword training")
                status = gr.Textbox(label="Training status", interactive=False)
                task = gr.Textbox(label="Current task", lines=4, interactive=False)
                summary = gr.JSON(label="Task and overall scores")
                failures = gr.Textbox(label="Current task: total failed attempts", interactive=False)
                retry_status = gr.Textbox(label="Task retry status and next position", interactive=False)
                checkpoint = gr.JSON(label="Checkpoint saves and Hugging Face uploads")
                with gr.Tab("Task outcomes"):
                    gr.Markdown("Solved counts use correctness checks. Unsolved includes tasks awaiting their first attempt. "
                                "Tables show up to 1,000 tasks; category totals include the entire dataset.")
                    categories = gr.Dataframe(headers=["Category", "Total problems", "Solved", "Unsolved", "Solved %",
                                                      "Attempted", "Failed attempts", "Waiting approval", "Removed"], interactive=False)
                    tasks = gr.Dataframe(headers=["Task", "Category", "Status", "Attempts", "Failed attempts", "Failed rollouts",
                                                 "Next position", "Last step", "Failure reason"], interactive=False)
                with gr.Tab("Task waitlist"):
                    gr.Markdown("Approve a task to start another retry cycle, or reject it to remove it from the active queue.")
                    waitlist = gr.Dataframe(headers=["Task", "Category", "Failed attempts", "Last truth", "Last step", "Problem"], interactive=False)
                    review_selected = gr.Dropdown(choices=[], value=None, label="Task to review", interactive=True)
                    approve = gr.Button("Approve retry")
                    reject = gr.Button("Reject / remove")
                    review_result = gr.Textbox(label="Review decision", interactive=False)
                with gr.Tab("Rollouts"):
                    outputs = []
                    for index in range(self.max_rollouts):
                        with gr.Accordion(f"Rollout {index + 1}", open=index < 2):
                            outputs.append(gr.Textbox(label="Model generation", lines=12, interactive=False))
                    scores = gr.Dataframe(headers=["Rollout", "Total", "Truth", "Formatting", "Reasoning rule",
                                                   "Answer rule", "Tools", "Cheating", "Judge"], interactive=False)
                with gr.Tab("Score history"):
                    plot = gr.LinePlot(x="step", y="mean_score", x_title="Step", y_title="Mean task score",
                                       title="Task score history")
                    history = gr.Dataframe(headers=["Step", "Task", "Mean score", "Total score"], interactive=False)
                with gr.Tab("Judge details"):
                    selected = gr.Dropdown(choices=[(str(i + 1), i) for i in range(self.max_rollouts)], value=0,
                                           label="Rollout")
                    quick = gr.JSON(label="Parsed judge scores")
                    raw = gr.Textbox(label="Raw judge response", lines=12, interactive=False)
                    evidence = gr.JSON(label="Measured execution evidence")
                    code_results = gr.JSON(label="Code test inputs and results")

                def refresh(index, review_key=None):
                    state = self.snapshot()
                    score_rows = []
                    for rollout, entry in sorted(state["scores"].items()):
                        value = entry["scores"]
                        score_rows.append([rollout + 1, value.get("total_score"), value.get("truth"), value.get("formatted"),
                                           value.get("rule_reasoning_follow"), value.get("rule_answer_follow"),
                                           value.get("tools_success"), value.get("cheating"), entry.get("judge_source")])
                    detail = state["scores"].get(int(index or 0), {})
                    generation = [state["rollouts"].get(i, {}).get("text", "") for i in range(self.max_rollouts)]
                    frame = pd.DataFrame(state["history"], columns=["step", "problem_id", "mean_score", "total_score"])
                    aggregate = {key: state[key] for key in ("step", "rollout_count", "mean_score", "task_total",
                                                            "overall_mean", "scored_rollouts")}
                    progress = state.get("task_progress", {})
                    current = progress.get("current_task", {})
                    totals = progress.get("summary", {})
                    aggregate.update(category=state["category"], task_attempts=current.get("attempts", 0),
                                     task_failures=current.get("failures", 0), task_failed_rollouts=current.get("failed_rollouts", 0),
                                     total_failed_attempts=totals.get("failed_attempts", 0),
                                     total_failed_rollouts=totals.get("failed_rollouts", 0),
                                     solved_tasks=totals.get("solved", 0), unsolved_tasks=totals.get("unsolved", 0),
                                     waiting_approval=totals.get("waiting_approval", 0))
                    category_rows = [[row[key] for key in ("category", "total", "solved", "unsolved", "solve_rate",
                                                          "attempted", "failed_attempts", "waiting_approval", "removed")]
                                     for row in progress.get("categories", [])]
                    task_rows = [[row.get(key) for key in ("problem_id", "category", "status", "attempts", "failures", "failed_rollouts",
                                                          "next_position", "last_step", "reason")]
                                 for row in progress.get("tasks", [])]
                    waiting = progress.get("waitlist", [])
                    wait_rows = [[row[key] for key in ("problem_id", "category", "failures", "last_truth", "last_step", "problem")]
                                 for row in waiting]
                    choices = [(f"{row['problem_id']} ({row['row_hash']})", row["row_hash"]) for row in waiting]
                    active_keys = {row["row_hash"] for row in waiting}
                    selection = gr.Dropdown(choices=choices, value=review_key if review_key in active_keys else None,
                                            interactive=True)
                    retry_text = current.get("status", "")
                    if current.get("next_position") is not None:
                        retry_text += f"; next queue position {current['next_position']}"
                    return [state["status"], state["problem"], aggregate, str(current.get("failures", 0)), retry_text,
                            state.get("checkpoint_upload", {}), category_rows, task_rows, wait_rows, selection,
                            *generation, score_rows, frame,
                            [[h["step"], h["problem_id"], h["mean_score"], h["total_score"]] for h in state["history"]],
                            detail.get("judge_scores", {}), detail.get("judge_response", ""), detail.get("execution_evidence", {}),
                            detail.get("code_execution", {})]

                components = [status, task, summary, failures, retry_status, checkpoint, categories, tasks, waitlist, review_selected,
                              *outputs, scores, plot, history, quick, raw, evidence, code_results]
                timer = gr.Timer(self.refresh_interval)
                timer.tick(refresh, inputs=[selected, review_selected], outputs=components, queue=False)
                app.load(refresh, inputs=[selected, review_selected], outputs=components, queue=False)
                selected.change(refresh, inputs=[selected, review_selected], outputs=components, queue=False)
                approve.click(lambda key: self.review_task(key, "approve"), inputs=[review_selected], outputs=[review_result], queue=False)
                reject.click(lambda key: self.review_task(key, "reject"), inputs=[review_selected], outputs=[review_result], queue=False)
            self.app = app
            _, self.local_url, self.share_url = app.launch(share=True, prevent_thread_lock=True,
                                                         server_port=self.server_port, quiet=True,
                                                         show_error=False, inbrowser=False)
            print(f"[Sword logger] Public URL: {self.share_url or self.local_url}", flush=True)
        except Exception as exc:
            self.error = str(exc)
            print(f"[Sword logger] Dashboard unavailable ({exc}); training continues. Install sword[logger] for Gradio.", flush=True)


def create_training_logger(**options):
    return TrainingLogger(**options).start()


@contextmanager
def observe_vllm_generation(llm, callback, *, interval=0.25):
    """Observe the existing engine loop; returned outputs and sampling stay intact."""
    engine = getattr(llm, "llm_engine", None)
    if engine is None or not callable(getattr(engine, "step", None)) or not callable(getattr(engine, "add_request", None)):
        yield
        return
    original_step, original_add = engine.step, engine.add_request
    mapping, emitted = {}, {}
    try:
        from vllm.sampling_params import RequestOutputKind
        cumulative_kind = RequestOutputKind.CUMULATIVE
    except ImportError:
        cumulative_kind = None

    def add_request(*args, **kwargs):
        request_id = kwargs.get("request_id", args[0] if args else None)
        if request_id is not None:
            mapping[str(request_id)] = len(mapping)
        # Offline LLM.generate may force FINAL_ONLY before submitting requests.
        # Change delivery at submission, leaving sampling and final results intact.
        params = kwargs.get("params", kwargs.get("sampling_params", args[2] if len(args) > 2 else None))
        if cumulative_kind is not None and hasattr(params, "output_kind"):
            params.output_kind = cumulative_kind
        return original_add(*args, **kwargs)

    def step(*args, **kwargs):
        outputs = original_step(*args, **kwargs)
        now = time.monotonic()
        for output in outputs:
            request_id = str(getattr(output, "request_id", ""))
            index = mapping.get(request_id)
            finished = bool(getattr(output, "finished", False))
            if index is None or (not finished and now - emitted.get(request_id, 0) < interval):
                continue
            responses = getattr(output, "outputs", [])
            if responses:
                try:
                    callback(index, responses[0].text, finished=finished)
                except Exception:
                    pass
                emitted[request_id] = now
        return outputs

    try:
        engine.step, engine.add_request = step, add_request
    except (AttributeError, TypeError):
        try:
            engine.step, engine.add_request = original_step, original_add
        except (AttributeError, TypeError):
            pass
        yield
        return
    try:
        yield
    finally:
        engine.step, engine.add_request = original_step, original_add
