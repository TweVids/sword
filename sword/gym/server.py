"""
HTTP Server and zrok Tunnel Manager for the Sword Coding Gym.
Hosts Docker execution on the local machine and exposes endpoints for remote clients (e.g. Molab).
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import socket
import logging
import subprocess
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import urlparse, parse_qs

from sword.gym.schema import GymInstance, GymExecutionResult
from sword.gym.docker_gym import DockerCodingGym
from sword.gym.adapters import load_swe_dataset

logger = logging.getLogger("sword.gym.server")


class GymRequestHandler(BaseHTTPRequestHandler):
    """
    Handles HTTP requests for coding gym evaluation and problem distribution.
    """

    server: GymServer  # Type hint for custom server

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/health":
            self._handle_health()
        elif path in ("/instances/next", "/next_problem"):
            count = int(params.get("n", [1])[0])
            self._handle_next_instances(count)
        elif path == "/instances/count":
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "queue_count": len(self.server.instances_queue),
                "completed_count": len(self.server.completed_problems),
            }).encode("utf-8"))
        elif path == "/state":
            with self.server.queue_lock:
                state_data = {
                    "queue_count": len(self.server.instances_queue),
                    "completed_count": len(self.server.completed_problems),
                    "completed_problems": self.server.completed_problems[-50:],
                }
            self._set_headers(200)
            self.wfile.write(json.dumps(state_data).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"Not found: {path}"}).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else ""

        try:
            data = json.loads(body) if body else {}
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": f"Invalid JSON body: {e}"}).encode("utf-8"))
            return

        if path == "/evaluate":
            self._handle_evaluate(data)
        elif path == "/evaluate_batch":
            self._handle_evaluate_batch(data)
        elif path == "/instances/add":
            self._handle_add_instances(data)
        elif path == "/state/save":
            saved_path = self.server.save_state()
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "saved", "path": saved_path}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"Not found: {path}"}).encode("utf-8"))

    def _handle_health(self):
        info = {
            "status": "ok",
            "server": "Sword-Coding-Gym-Server",
            "docker_available": self.server.gym.client is not None,
            "pool_size": len(self.server.gym._pool),
            "queued_problems": len(self.server.instances_queue),
            "completed_problems": len(self.server.completed_problems),
            "state_file": self.server.state_file,
            "timestamp": time.time(),
        }
        self._set_headers(200)
        self.wfile.write(json.dumps(info).encode("utf-8"))

    def _handle_next_instances(self, count: int):
        pulled = []
        with self.server.queue_lock:
            for _ in range(min(count, len(self.server.instances_queue))):
                pulled.append(self.server.instances_queue.pop(0).to_dict())
            if pulled and self.server.state_file:
                self.server.save_state()

        self._set_headers(200)
        self.wfile.write(json.dumps({"instances": pulled}).encode("utf-8"))

    def _handle_add_instances(self, data: Dict[str, Any]):
        items = data.get("instances", [])
        if isinstance(data, list):
            items = data

        parsed_instances = []
        for item in items:
            try:
                parsed_instances.append(GymInstance.from_dict(item))
            except Exception as e:
                logger.warning(f"Failed to parse instance: {e}")

        added = self.server.add_instances(parsed_instances)
        self._set_headers(200)
        self.wfile.write(json.dumps({
            "added": added,
            "total_queued": len(self.server.instances_queue),
            "total_known": len(self.server.known_instance_ids),
        }).encode("utf-8"))

    def _handle_evaluate(self, data: Dict[str, Any]):
        raw_inst = data.get("instance")
        if not raw_inst:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Missing 'instance' field"}).encode("utf-8"))
            return

        patch = data.get("patch", "")
        timeout = data.get("timeout")
        instance = GymInstance.from_dict(raw_inst)

        logger.info(f"Evaluating patch for instance {instance.instance_id}...")
        result = self.server.gym.evaluate_patch(instance, patch, timeout=timeout)
        
        # Track completion status
        with self.server.queue_lock:
            self.server.completed_problems.append({
                "instance_id": instance.instance_id,
                "success": result.success,
                "reward": result.reward,
                "timestamp": time.time(),
            })
            if self.server.state_file:
                self.server.save_state()

        self._set_headers(200)
        self.wfile.write(json.dumps(result.to_dict()).encode("utf-8"))

    def _handle_evaluate_batch(self, data: Dict[str, Any]):
        raw_inst = data.get("instance")
        trajectories = data.get("trajectories", [])
        if not raw_inst or not trajectories:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Missing 'instance' or 'trajectories'"}).encode("utf-8"))
            return

        instance = GymInstance.from_dict(raw_inst)
        max_workers = data.get("max_workers", 4)

        logger.info(f"Evaluating {len(trajectories)} rollouts for instance {instance.instance_id}...")
        results = self.server.gym.evaluate_batch(instance, trajectories, max_workers=max_workers)
        
        # Track completion status
        best_reward = max((r.reward for r in results), default=0.0)
        any_success = any(r.success for r in results)
        with self.server.queue_lock:
            self.server.completed_problems.append({
                "instance_id": instance.instance_id,
                "num_rollouts": len(results),
                "success": any_success,
                "best_reward": best_reward,
                "timestamp": time.time(),
            })
            if self.server.state_file:
                self.server.save_state()

        self._set_headers(200)
        self.wfile.write(json.dumps({"results": [r.to_dict() for r in results]}).encode("utf-8"))

    def log_message(self, format, *args):
        # Override to clean up standard output spam
        if "/health" not in args[0]:
            logger.info("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))


class GymServer(ThreadingHTTPServer):
    """Threading HTTP server with attached DockerCodingGym, dynamic updates, and persistent state."""

    def __init__(
        self,
        server_address: Tuple[str, int],
        gym: DockerCodingGym,
        instances: Optional[List[GymInstance]] = None,
        state_file: Optional[str] = None,
    ):
        super().__init__(server_address, GymRequestHandler)
        self.gym = gym
        self.state_file = state_file
        self.instances_queue: List[GymInstance] = list(instances or [])
        self.completed_problems: List[Dict[str, Any]] = []
        self.known_instance_ids: set = {inst.instance_id for inst in self.instances_queue}
        self.queue_lock = threading.Lock()

        # Load persisted state if file exists
        if self.state_file and os.path.exists(self.state_file):
            self.load_state(self.state_file)

    def save_state(self, path: Optional[str] = None) -> str:
        """Saves active queue, completed problems, and statistics to JSON state file."""
        target_path = path or self.state_file or "gym_state.json"
        with self.queue_lock:
            state = {
                "timestamp": time.time(),
                "queue_count": len(self.instances_queue),
                "completed_count": len(self.completed_problems),
                "instances_queue": [inst.to_dict() for inst in self.instances_queue],
                "completed_problems": self.completed_problems,
            }
            tmp_path = f"{target_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, target_path)
            logger.info(f"[Sword-Gym-Server] 💾 Persisted gym state to {target_path} ({len(self.instances_queue)} queued, {len(self.completed_problems)} completed)")
            return target_path

    def load_state(self, path: str) -> bool:
        """Restores queue and completed problem tracking from JSON state file."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            with self.queue_lock:
                loaded_queue = [GymInstance.from_dict(d) for d in state.get("instances_queue", [])]
                self.completed_problems = state.get("completed_problems", [])
                
                # Merge into active queue avoiding duplicates
                for inst in loaded_queue:
                    if inst.instance_id not in self.known_instance_ids:
                        self.instances_queue.append(inst)
                        self.known_instance_ids.add(inst.instance_id)

            logger.info(f"[Sword-Gym-Server] 📂 Loaded state from {path}: {len(self.instances_queue)} active items in queue.")
            return True
        except Exception as e:
            logger.error(f"[Sword-Gym-Server] Failed to load state from {path}: {e}")
            return False

    def add_instances(self, instances: List[GymInstance]) -> int:
        """Appends new problem instances dynamically, preventing duplicate IDs."""
        added = 0
        with self.queue_lock:
            for inst in instances:
                if inst.instance_id not in self.known_instance_ids:
                    self.instances_queue.append(inst)
                    self.known_instance_ids.add(inst.instance_id)
                    added += 1
            if added > 0 and self.state_file:
                self.save_state()
        return added


class ZrokTunnelManager:
    """
    Manages background zrok public sharing tunnel.
    """

    def __init__(self, target_port: int, zrok_cmd: str = "zrok2"):
        self.target_port = target_port
        self.zrok_cmd = zrok_cmd
        self.process: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None
        self.share_token: Optional[str] = None

    def start(self, timeout_sec: int = 20) -> str:
        """Starts zrok public tunnel and returns public HTTPS URL."""
        target = f"http://127.0.0.1:{self.target_port}"
        cmd = [self.zrok_cmd, "share", "public", target, "--headless"]

        logger.info(f"Starting zrok tunnel for {target}...")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        # Query zrok overview after a short delay to discover the new share token
        t0 = time.time()
        time.sleep(3)

        while time.time() - t0 < timeout_sec:
            token, url = self._query_active_share(target)
            if url:
                self.share_token = token
                self.public_url = url
                logger.info(f"✅ zrok public endpoint active: {self.public_url}")
                return self.public_url
            time.sleep(1)

        raise TimeoutError("Timed out waiting for zrok public share to establish")

    def _query_active_share(self, target: str) -> Tuple[Optional[str], Optional[str]]:
        """Queries zrok overview to locate the share token for target port."""
        try:
            res = subprocess.run(
                [self.zrok_cmd, "overview"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            out = res.stdout
            # Pattern in zrok overview table:
            # │ <share_token> │ public │ proxy   │ http://127.0.0.1:8765 │
            m = re.search(rf"│\s+([a-zA-Z0-9]+)\s+│\s+public\s+│\s+proxy\s+│\s+{re.escape(target)}", out)
            if m:
                token = m.group(1).strip()
                return token, f"https://{token}.shares.zrok.io"

            # Fallback: check shares listing under Shares section
            lines = out.splitlines()
            for i, line in enumerate(lines):
                if target in line:
                    parts = [p.strip() for p in line.split("│") if p.strip()]
                    if parts:
                        token = parts[0]
                        return token, f"https://{token}.shares.zrok.io"
        except Exception as e:
            logger.debug(f"zrok overview query error: {e}")
        return None, None

    def stop(self) -> None:
        """Stops the zrok tunnel and deletes the share."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        if self.share_token:
            try:
                subprocess.run(
                    [self.zrok_cmd, "delete", "share", self.share_token],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass
            self.share_token = None
            self.public_url = None
