"""
Remote Coding Gym Client for Molab / Distributed GPU environments.
Connects over HTTPS to the local machine's zrok public endpoint to execute Docker evaluations.
"""

from __future__ import annotations

import json
import os
import time
import logging
from typing import Dict, Any, List, Optional, Union
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from sword.gym.schema import GymInstance, GymExecutionResult

logger = logging.getLogger("sword.gym.client")


class RemoteCodingGym:
    """
    Drop-in replacement for DockerCodingGym that delegates code execution
    and scoring across the network via an HTTP / zrok tunnel.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: int = 180,
        retries: int = 3,
    ):
        # Resolve endpoint from argument, environment, or file
        self.endpoint = (
            endpoint
            or os.environ.get("SWORD_GYM_ENDPOINT")
            or self._read_endpoint_file()
        )
        if not self.endpoint:
            raise ValueError(
                "RemoteCodingGym requires an endpoint URL. "
                "Provide endpoint='https://<id>.shares.zrok.io' or set SWORD_GYM_ENDPOINT."
            )

        self.endpoint = self.endpoint.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        logger.info(f"[Sword-Gym-Client] Connected to remote gym at {self.endpoint}")

    @staticmethod
    def _read_endpoint_file(filename: str = "gym_endpoint.txt") -> Optional[str]:
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    url = f.read().strip()
                    if url:
                        return url
            except Exception:
                pass
        return None

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.endpoint}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Sword-Remote-Gym-Client/0.7.0",
        }

        last_error = None
        for attempt in range(self.retries):
            try:
                req = Request(url, data=data, headers=headers, method="POST")
                with urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (URLError, HTTPError, TimeoutError, Exception) as e:
                last_error = e
                logger.warning(f"[Sword-Gym-Client] POST {url} attempt {attempt + 1}/{self.retries} failed: {e}")
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Remote gym call to {url} failed after {self.retries} attempts: {last_error}")

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.endpoint}{path}"
        headers = {
            "User-Agent": "Sword-Remote-Gym-Client/0.7.0",
        }

        last_error = None
        for attempt in range(self.retries):
            try:
                req = Request(url, headers=headers, method="GET")
                with urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                last_error = e
                logger.warning(f"[Sword-Gym-Client] GET {url} attempt {attempt + 1}/{self.retries} failed: {e}")
                time.sleep(1)

        raise RuntimeError(f"Remote gym call to {url} failed: {last_error}")

    def health(self) -> Dict[str, Any]:
        """Checks remote server status."""
        return self._get("/health")

    def get_next_problem(self, n: int = 1) -> List[GymInstance]:
        """Pulls next n SWE problem instances from the remote gym queue."""
        res = self._get(f"/instances/next?n={n}")
        return [GymInstance.from_dict(item) for item in res.get("instances", [])]

    def evaluate_patch(
        self,
        instance: GymInstance,
        candidate_patch: str,
        timeout: Optional[int] = None,
    ) -> GymExecutionResult:
        """
        Submits candidate patch across network to remote Docker gym server.
        """
        payload = {
            "instance": instance.to_dict(),
            "patch": candidate_patch,
            "timeout": timeout or self.timeout,
        }
        res = self._post("/evaluate", payload)
        return GymExecutionResult(
            instance_id=res.get("instance_id", instance.instance_id),
            success=bool(res.get("success", False)),
            reward=float(res.get("reward", 0.0)),
            exit_code=int(res.get("exit_code", 0)),
            f2p_passed=int(res.get("f2p_passed", 0)),
            f2p_total=int(res.get("f2p_total", 0)),
            p2p_passed=int(res.get("p2p_passed", 0)),
            p2p_total=int(res.get("p2p_total", 0)),
            patch_applied=bool(res.get("patch_applied", True)),
            patch_error=res.get("patch_error"),
            stdout=res.get("stdout", ""),
            stderr=res.get("stderr", ""),
            duration_sec=float(res.get("duration_sec", 0.0)),
            details=res.get("details", {}),
        )

    def evaluate_batch(
        self,
        instance: GymInstance,
        candidate_trajectories: List[str],
        max_workers: Optional[int] = None,
    ) -> List[GymExecutionResult]:
        """
        Submits G rollouts concurrently for batch execution on remote Docker pool.
        """
        payload = {
            "instance": instance.to_dict(),
            "trajectories": candidate_trajectories,
            "max_workers": max_workers or len(candidate_trajectories),
        }
        res = self._post("/evaluate_batch", payload)
        results = []
        for r in res.get("results", []):
            results.append(
                GymExecutionResult(
                    instance_id=r.get("instance_id", instance.instance_id),
                    success=bool(r.get("success", False)),
                    reward=float(r.get("reward", 0.0)),
                    exit_code=int(r.get("exit_code", 0)),
                    f2p_passed=int(r.get("f2p_passed", 0)),
                    f2p_total=int(r.get("f2p_total", 0)),
                    p2p_passed=int(r.get("p2p_passed", 0)),
                    p2p_total=int(r.get("p2p_total", 0)),
                    patch_applied=bool(r.get("patch_applied", True)),
                    patch_error=r.get("patch_error"),
                    stdout=r.get("stdout", ""),
                    stderr=r.get("stderr", ""),
                    duration_sec=float(r.get("duration_sec", 0.0)),
                    details=r.get("details", {}),
                )
            )
        return results

    def add_instances(self, instances: List[GymInstance]) -> Dict[str, Any]:
        """Uploads new SWE problem instances dynamically to the gym server."""
        payload = {"instances": [inst.to_dict() for inst in instances]}
        return self._post("/instances/add", payload)

    def save_state(self) -> Dict[str, Any]:
        """Triggers persistent JSON state save on the host server."""
        return self._post("/state/save", {})

    def get_state(self) -> Dict[str, Any]:
        """Inspects current queue progress and completed problems from host server."""
        return self._get("/state")

    def close(self) -> None:
        pass
