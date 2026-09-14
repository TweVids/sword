"""
Docker Coding Gym for GRPO Reinforcement Learning.
Executes candidate patches inside isolated Docker containers, supports OpenSWE and Scale-SWE
harnesses, and provides deterministic execution rewards for GRPO rollouts.
"""

from __future__ import annotations

import io
import os
import re
import shlex
import tarfile
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import docker
    from docker.models.containers import Container
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    Container = Any

from sword.gym.schema import GymInstance, GymExecutionResult

logger = logging.getLogger("sword.gym")


def extract_unified_diff(text: str) -> str:
    """
    Extracts unified diff patch from model output.
    Looks for ```diff ... ``` code blocks, or raw unified diff headers.
    Always ensures trailing newline is present as required by git apply and patch.
    """
    if not text:
        return ""

    # 1. Look for ```diff blocks
    diff_blocks = re.findall(r"```(?:diff|patch)?\n(.*?)\n```", text, flags=re.DOTALL)
    if diff_blocks:
        result = "\n".join(diff_blocks).strip()
    # 2. Check for raw diff headers: --- a/ and +++ b/ or diff --git
    elif "diff --git" in text or ("--- " in text and "+++ " in text):
        lines = text.splitlines()
        start_idx = -1
        for i, line in enumerate(lines):
            if line.startswith("diff --git") or line.startswith("--- "):
                start_idx = i
                break
        if start_idx != -1:
            diff_lines = []
            for line in lines[start_idx:]:
                if line.startswith("```"):
                    break
                diff_lines.append(line)
            result = "\n".join(diff_lines).strip()
        else:
            result = text.strip()
    else:
        result = text.strip()

    if result and not result.endswith("\n"):
        result += "\n"
    return result


class ContainerSlot:
    """A managed Docker container wrapper for fast execution and resets."""

    def __init__(self, container: Container, base_image: str):
        self.container = container
        self.base_image = base_image
        self.in_use = False

    def exec(
        self,
        command: str,
        workdir: str = "/testbed",
        timeout: int = 60,
        env: Optional[Dict[str, str]] = None
    ) -> Tuple[int, str, str]:
        """Runs a bash command inside container."""
        result = self.container.exec_run(
            ["bash", "-c", command],
            workdir=workdir,
            environment=env or None,
            demux=True,
        )
        exit_code = result.exit_code
        output = result.output

        stdout = ""
        stderr = ""
        if isinstance(output, tuple):
            stdout = (output[0] or b"").decode("utf-8", errors="replace")
            stderr = (output[1] or b"").decode("utf-8", errors="replace")
        elif isinstance(output, bytes):
            stdout = output.decode("utf-8", errors="replace")

        return exit_code, stdout, stderr

    def write_file(self, container_path: str, content: str) -> None:
        """Writes a string into a file inside the container via tar stream."""
        data = content.encode("utf-8")
        buf = io.BytesIO()
        filename = os.path.basename(container_path)
        dirname = os.path.dirname(container_path) or "/"

        tar_info = tarfile.TarInfo(name=filename)
        tar_info.size = len(data)
        tar_info.mtime = int(time.time())

        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.addfile(tar_info, io.BytesIO(data))

        buf.seek(0)
        self.container.put_archive(dirname, buf.getvalue())

    def reset_workspace(self, workdir: str = "/testbed") -> None:
        """Cleans working directory back to base state."""
        self.exec(
            "git reset --hard HEAD && git clean -fdx || true",
            workdir=workdir,
            timeout=10,
        )

    def close(self) -> None:
        try:
            self.container.remove(force=True)
        except Exception:
            pass


class DockerCodingGym:
    """
    High-Performance Docker Coding Gym for RL / GRPO.
    Manages warm container pools, executes test suites (OpenSWE / Scale-SWE),
    and computes verified execution rewards for LLM trajectories.
    """

    def __init__(
        self,
        default_image: str = "sword-coding-gym:latest",
        pool_size: int = 2,
        mem_limit: str = "4g",
        nano_cpus: int = 2_000_000_000,
        timeout: int = 60,
        network_disabled: bool = True,
    ):
        self.default_image = default_image
        self.pool_size = pool_size
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self.timeout = timeout
        self.network_disabled = network_disabled
        self.client: Optional[docker.DockerClient] = None
        self._pool: List[ContainerSlot] = []

        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
                self._init_pool()
            except Exception as e:
                logger.warning(f"[Sword-Gym] Could not connect to Docker daemon: {e}")
                self.client = None

    def _init_pool(self) -> None:
        """Pre-warms background container slots for zero-latency execution."""
        if not self.client:
            return

        for _ in range(self.pool_size):
            try:
                c = self.client.containers.run(
                    self.default_image,
                    command="sleep infinity",
                    detach=True,
                    mem_limit=self.mem_limit,
                    nano_cpus=self.nano_cpus,
                    network_mode="none" if self.network_disabled else "bridge",
                    working_dir="/testbed",
                )
                self._pool.append(ContainerSlot(c, self.default_image))
            except Exception as e:
                logger.warning(f"[Sword-Gym] Failed to pre-warm container: {e}")
                break

    def _acquire_slot(self, image: str) -> ContainerSlot:
        """Acquires an idle slot from the pool, or spins up an ephemeral container."""
        # Check if an idle slot with matching image exists
        for slot in self._pool:
            if not slot.in_use and slot.base_image == image:
                slot.in_use = True
                return slot

        # Otherwise create an on-demand slot
        c = self.client.containers.run(
            image,
            command="sleep infinity",
            detach=True,
            mem_limit=self.mem_limit,
            nano_cpus=self.nano_cpus,
            network_mode="none" if self.network_disabled else "bridge",
            working_dir="/testbed",
        )
        slot = ContainerSlot(c, image)
        slot.in_use = True
        return slot

    def _release_slot(self, slot: ContainerSlot, workdir: str = "/testbed") -> None:
        """Releases or recycles a slot."""
        if slot in self._pool:
            try:
                slot.reset_workspace(workdir)
            except Exception:
                pass
            slot.in_use = False
        else:
            # Ephemeral slot
            slot.close()

    def evaluate_patch(
        self,
        instance: GymInstance,
        candidate_patch: str,
        timeout: Optional[int] = None,
    ) -> GymExecutionResult:
        """
        Applies a candidate patch to the instance environment and runs its test harness.
        Supports both OpenSWE and Scale-SWE protocols.
        """
        if not self.client:
            return GymExecutionResult(
                instance_id=instance.instance_id,
                success=False,
                reward=-0.1,
                patch_applied=False,
                patch_error="Docker is not available on this host",
            )

        t_start = time.perf_counter()
        image = instance.docker_image or self.default_image
        timeout = timeout or self.timeout

        # Ensure image is locally present; fallback to default if not
        try:
            self.client.images.get(image)
        except Exception:
            image = self.default_image

        slot = self._acquire_slot(image)
        try:
            # 1. Setup workspace (pre_commands)
            workdir = instance.workdir or "/testbed"
            for cmd in instance.pre_commands:
                slot.exec(cmd, workdir=workdir, timeout=20)

            # 2. Extract and apply candidate patch
            clean_patch = extract_unified_diff(candidate_patch)
            if not clean_patch:
                return GymExecutionResult(
                    instance_id=instance.instance_id,
                    success=False,
                    reward=-0.3,
                    patch_applied=False,
                    patch_error="No valid unified diff could be extracted from rollout",
                    duration_sec=time.perf_counter() - t_start,
                )

            slot.write_file(f"{workdir}/candidate.patch", clean_patch)
            apply_code, apply_out, apply_err = slot.exec(
                "git apply --verbose --whitespace=fix candidate.patch || git apply -p0 --whitespace=fix candidate.patch || patch -p1 -N < candidate.patch || patch -p0 -N < candidate.patch",
                workdir=workdir,
                timeout=15,
            )
            if apply_code != 0:
                return GymExecutionResult(
                    instance_id=instance.instance_id,
                    success=False,
                    reward=-0.2,
                    patch_applied=False,
                    patch_error=f"Patch failed to apply (exit {apply_code}): {apply_err[:400]}",
                    stdout=apply_out,
                    stderr=apply_err,
                    duration_sec=time.perf_counter() - t_start,
                )

            # 3. Dispatch to test harness based on source
            if instance.source == "openswe" or instance.eval_script:
                result = self._run_openswe_harness(instance, slot, workdir, timeout)
            else:
                result = self._run_scaleswe_harness(instance, slot, workdir, timeout)

            result.duration_sec = time.perf_counter() - t_start
            return result

        finally:
            self._release_slot(slot, workdir=instance.workdir or "/testbed")

    def _run_openswe_harness(
        self,
        instance: GymInstance,
        slot: ContainerSlot,
        workdir: str,
        timeout: int,
    ) -> GymExecutionResult:
        """Executes OpenSWE eval.sh script and checks OPENSWE_EXIT_CODE."""
        eval_script = instance.eval_script
        if not eval_script and instance.test_patch:
            # Apply test patch and run pytest
            slot.write_file(f"{workdir}/test.patch", instance.test_patch)
            slot.exec("git apply test.patch || patch -p1 < test.patch", workdir=workdir, timeout=10)
            eval_script = "pytest"

        if eval_script:
            slot.write_file(f"{workdir}/eval.sh", eval_script)
            cmd = "bash eval.sh"
        else:
            cmd = "pytest"

        exit_code, stdout, stderr = slot.exec(cmd, workdir=workdir, timeout=timeout)
        success = (exit_code == 0)

        # Check for OPENSWE_EXIT_CODE=0 marker in output
        if "OPENSWE_EXIT_CODE: 0" in stdout or "OPENSWE_EXIT_CODE=0" in stdout:
            success = True
        elif "OPENSWE_EXIT_CODE" in stdout:
            success = False

        reward = 1.0 if success else -0.1
        return GymExecutionResult(
            instance_id=instance.instance_id,
            success=success,
            reward=reward,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            details={"harness": "openswe"},
        )

    def _run_scaleswe_harness(
        self,
        instance: GymInstance,
        slot: ContainerSlot,
        workdir: str,
        timeout: int,
    ) -> GymExecutionResult:
        """Executes Scale-SWE fail-to-pass reproduction and pass-to-pass regression tests."""
        # 1. Apply f2p_patch if present
        if instance.f2p_patch:
            slot.write_file(f"{workdir}/f2p.patch", instance.f2p_patch)
            slot.exec("git apply f2p.patch || patch -p1 < f2p.patch", workdir=workdir, timeout=10)

        # 2. Upload f2p_script as test_fail_to_pass.py if present
        if instance.f2p_script:
            slot.write_file(f"{workdir}/test_fail_to_pass.py", instance.f2p_script)

        # 3. Assemble target test cases
        all_tests = instance.fail_to_pass + instance.pass_to_pass
        if not all_tests:
            if instance.f2p_script:
                all_tests = ["test_fail_to_pass.py"]
            else:
                all_tests = ["tests"]

        test_targets = " ".join(shlex.quote(t) for t in all_tests)
        test_cmd = f"pytest -v {test_targets}"

        exit_code, stdout, stderr = slot.exec(test_cmd, workdir=workdir, timeout=timeout)

        # Parse test outcomes
        combined = stdout + "\n" + stderr
        f2p_passed = 0
        f2p_total = max(1, len(instance.fail_to_pass))
        for t in instance.fail_to_pass:
            if re.search(rf"{re.escape(t)}\s+PASSED", combined, re.IGNORECASE):
                f2p_passed += 1

        p2p_passed = 0
        p2p_total = len(instance.pass_to_pass)
        if p2p_total > 0:
            for t in instance.pass_to_pass:
                if re.search(rf"{re.escape(t)}\s+PASSED", combined, re.IGNORECASE):
                    p2p_passed += 1
        else:
            p2p_passed = p2p_total

        # Compute fine-grained reward
        all_passed = (exit_code == 0) or (f2p_passed == f2p_total and p2p_passed == p2p_total)
        if all_passed:
            reward = 1.0
        else:
            # Partial reward for turning some F2P to green
            reward = (0.5 * (f2p_passed / f2p_total))
            if p2p_total > 0 and p2p_passed < p2p_total:
                # Regressed on existing tests
                reward -= 0.5

        return GymExecutionResult(
            instance_id=instance.instance_id,
            success=all_passed,
            reward=round(reward, 3),
            exit_code=exit_code,
            f2p_passed=f2p_passed,
            f2p_total=f2p_total,
            p2p_passed=p2p_passed,
            p2p_total=p2p_total,
            stdout=stdout,
            stderr=stderr,
            details={"harness": "scale_swe"},
        )

    def evaluate_batch(
        self,
        instance: GymInstance,
        candidate_trajectories: List[str],
        max_workers: int = 4,
    ) -> List[GymExecutionResult]:
        """
        Concurrently evaluates G rollouts for the same SWE problem instance during GRPO.
        """
        with ThreadPoolExecutor(max_workers=min(max_workers, len(candidate_trajectories))) as pool:
            futures = [
                pool.submit(self.evaluate_patch, instance, traj)
                for traj in candidate_trajectories
            ]
            return [f.result() for f in futures]

    def close(self) -> None:
        """Tears down all warm container slots."""
        for slot in self._pool:
            slot.close()
        self._pool.clear()

    def __enter__(self) -> DockerCodingGym:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
