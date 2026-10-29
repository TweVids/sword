"""Task outcomes, bounded retries, and durable decisions for the long trainer."""
import copy
import json
import os
import threading
import time
from collections import deque
from itertools import chain, islice


class TaskRetryQueue:
    def __init__(self, records, *, retry_limit=3, retry_delay=100, failure_action="waitlist",
                 enabled=True, completed_hashes=(), state=None, decisions_path=None):
        if int(retry_limit) < 1 or int(retry_delay) < 1:
            raise ValueError("task_retry_limit and task_retry_delay must be positive")
        if failure_action not in ("waitlist", "remove"):
            raise ValueError("task_failure_action must be 'waitlist' or 'remove'")
        self.lock = threading.RLock()
        self.decision_event = threading.Event()
        self.decisions_path = decisions_path
        self.config = dict(retry_limit=int(retry_limit), retry_delay=int(retry_delay),
                           failure_action=failure_action, enabled=bool(enabled))
        completed = set(completed_hashes)
        self.tasks = {}
        for row in records:
            self.tasks.setdefault(row.row_hash, {
                "row_hash": row.row_hash, "problem_id": row.problem_id, "category": row.category,
                "problem": row.problem[:1000], "status": "solved" if row.row_hash in completed else "pending",
                "solved": row.row_hash in completed, "attempts": 0, "failures": 0,
                "failed_rollouts": 0, "rollouts": 0, "consecutive_failures": 0, "deferred": False,
                "deferrals": 0, "last_step": None, "last_truth": None, "best_truth": None, "reason": "",
            })
        self.queue = deque(key for key in self.tasks if key not in completed)
        self.dispatch_count = 0
        self.run_id = os.urandom(16).hex()
        self.decision_revision = 0
        self.decisions = deque(maxlen=1000)
        if state is not None:
            self._restore(state)
        self._summary = {name: 0 for name in ("total", "solved", "unsolved", "attempted", "failed_tasks",
                                              "failed_attempts", "failed_rollouts", "waiting_approval", "removed")}
        self._categories, self._waiting, self._failed = {}, {}, {}
        self._queued = set(self.queue)
        for task in self.tasks.values():
            self._change_counts(task, 1)
            self._reindex_task(task)
        self.replay_decisions()

    def _change_counts(self, task, direction):
        values = {"total": 1, "solved": int(task["solved"]), "unsolved": int(not task["solved"]),
                  "attempted": int(task["attempts"] > 0), "failed_tasks": int(task["failures"] > 0),
                  "failed_attempts": task["failures"], "failed_rollouts": task["failed_rollouts"],
                  "waiting_approval": int(task["status"] == "waiting_approval"),
                  "removed": int(task["status"] == "rejected")}
        category = self._categories.setdefault(task["category"], {"category": task["category"], **dict.fromkeys(values, 0)})
        for name, value in values.items():
            self._summary[name] += direction * value
            category[name] += direction * value

    def _reindex_task(self, task):
        key = task["row_hash"]
        if task["status"] == "waiting_approval":
            self._waiting[key] = task
        else:
            self._waiting.pop(key, None)
        if task["failures"]:
            self._failed[key] = task

    def _restore(self, state):
        if state.get("version") != 1 or set(state.get("tasks", {})) != set(self.tasks):
            raise ValueError("Checkpoint task queue does not match this dataset")
        config = state["config"]
        if config["retry_limit"] < 1 or config["retry_delay"] < 1 or config["failure_action"] not in ("waitlist", "remove"):
            raise ValueError("Invalid checkpoint retry configuration")
        tasks = copy.deepcopy(state["tasks"])
        keys = state["queue"]
        queued = set(keys)
        active = {key for key, task in tasks.items() if task["status"] not in ("waiting_approval", "rejected")
                  and key in queued}
        required = {key for key, task in tasks.items() if task["status"] in ("pending", "retrying", "unsolved", "deferred")}
        if len(queued) != len(keys) or queued != active or not required.issubset(queued):
            raise ValueError("Invalid checkpoint task queue entries")
        self.tasks, self.queue, self.config = tasks, deque(keys), dict(config)
        self.run_id = state["run_id"]
        self.dispatch_count = int(state["dispatch_count"])
        self.decision_revision = int(state.get("decision_revision", 0))
        self.decisions.extend(state.get("decisions", []))

    def next_key(self):
        with self.lock:
            return self.queue[0] if self.queue else None

    def record_result(self, key, *, solved_rollouts, rollout_count, best_truth, step, reason=""):
        """Commit once after a training step, never for an infrastructure/OOM retry."""
        if rollout_count < 1 or not 0 <= solved_rollouts <= rollout_count:
            raise ValueError("Invalid task rollout outcome counts")
        with self.lock:
            task = self.tasks[key]
            self.queue.remove(key)
            self._queued.remove(key)
            self._change_counts(task, -1)
            self.dispatch_count += 1
            task["attempts"] += 1
            task["rollouts"] += rollout_count
            task["failed_rollouts"] += rollout_count - solved_rollouts
            task.update(last_step=step, last_truth=best_truth, solved=solved_rollouts > 0)
            task["best_truth"] = max(best_truth, task["best_truth"] or 0.0)
            if task["solved"]:
                task.update(status="solved", consecutive_failures=0, deferred=False, reason="")
                self.queue.append(key)
            else:
                task["failures"] += 1
                task["consecutive_failures"] += 1
                task["reason"] = reason or "No rollout satisfied the correctness checks"
                if not self.config["enabled"]:
                    task["status"] = "unsolved"
                    self.queue.append(key)
                elif task["deferred"]:
                    task["status"] = "waiting_approval" if self.config["failure_action"] == "waitlist" else "rejected"
                elif task["consecutive_failures"] >= self.config["retry_limit"]:
                    task.update(status="deferred", deferred=True, deferrals=task["deferrals"] + 1)
                    # At dispatch N, 99 tasks ahead puts this task at N+100.
                    # A short queue uses its tail; training never inserts idle steps.
                    self.queue.insert(min(self.config["retry_delay"] - 1, len(self.queue)), key)
                else:
                    task["status"] = "retrying"
                    self.queue.appendleft(key)
            if task["status"] not in ("waiting_approval", "rejected"):
                self._queued.add(key)
            self._change_counts(task, 1)
            self._reindex_task(task)
            return dict(task)

    def _apply_decision(self, entry):
        task = self.tasks.get(entry["row_hash"])
        if (entry.get("run_id") != self.run_id or task is None or task["status"] != "waiting_approval" or
                task["failures"] != entry["failures"] or task["last_step"] != entry["last_step"]):
            return False
        if entry["action"] not in ("approve", "reject"):
            return False
        self._change_counts(task, -1)
        if entry["action"] == "approve":
            task.update(status="retrying", consecutive_failures=0, deferred=False)
            self.queue.appendleft(entry["row_hash"])
            self._queued.add(entry["row_hash"])
        else:
            task["status"] = "rejected"
        self._change_counts(task, 1)
        self._reindex_task(task)
        self.decision_revision += 1
        self.decisions.append(dict(entry))
        return True

    def decide(self, key, action):
        if action not in ("approve", "reject"):
            raise ValueError("Unknown waitlist decision")
        with self.lock:
            task = self.tasks.get(key)
            if task is None or task["status"] != "waiting_approval":
                return False
            entry = {"row_hash": key, "action": action, "failures": task["failures"],
                     "last_step": task["last_step"], "timestamp": time.time(), "run_id": self.run_id}
            if self.decisions_path:
                with open(self.decisions_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            self._apply_decision(entry)
        self.decision_event.set()
        return True

    def replay_decisions(self):
        """Replay only decisions for the exact failed attempt saved in this checkpoint."""
        if not self.decisions_path or not os.path.isfile(self.decisions_path):
            return
        with self.lock, open(self.decisions_path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    self._apply_decision(json.loads(line))
                except (ValueError, KeyError, TypeError):
                    continue

    def export_state(self):
        with self.lock:
            return {"version": 1, "config": dict(self.config), "queue": list(self.queue),
                    "run_id": self.run_id,
                    "tasks": copy.deepcopy(self.tasks), "dispatch_count": self.dispatch_count,
                    "decision_revision": self.decision_revision, "decisions": list(self.decisions)}

    def solved_keys(self):
        with self.lock:
            return {key for key, task in self.tasks.items() if task["solved"]}

    def monitor_snapshot(self, current_key=None, limit=1000):
        with self.lock:
            summary = {**self._summary, "queued": len(self.queue)}
            categories = [dict(category) for category in self._categories.values()]
            waiting = [dict(task) for task in islice(self._waiting.values(), limit)]
            rows, shown = [], set()
            current_keys = [current_key] if current_key in self.tasks else []
            for key in chain(current_keys, self._waiting, self._failed, self.tasks):
                if len(rows) >= limit:
                    break
                if key not in shown:
                    rows.append(dict(self.tasks[key]))
                    shown.add(key)
            needed = (shown | set(current_keys)) & self._queued
            positions = {}
            if needed:
                for index, key in enumerate(self.queue):
                    if key in needed:
                        positions[key] = self.dispatch_count + index + 1
                        needed.remove(key)
                        if not needed:
                            break
            for task in rows:
                task["next_position"] = positions.get(task["row_hash"])
            current = dict(self.tasks.get(current_key, {}))
            if current:
                current["next_position"] = positions.get(current_key)
            for category in categories:
                category["solve_rate"] = 100.0 * category["solved"] / max(1, category["total"])
            summary["solve_rate"] = 100.0 * summary["solved"] / max(1, summary["total"])
            return {"summary": summary, "categories": sorted(categories, key=lambda row: row["category"]),
                    "tasks": rows, "waitlist": waiting, "current_task": current, "config": dict(self.config)}
