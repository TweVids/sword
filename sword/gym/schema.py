"""
Data schemas for the Sword Coding Gym (OpenSWE / Scale-SWE).
Defines contracts for SWE instances, execution results, and adapters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from sword.rl.schema import DatasetRow, DomainType, EffortTier, Difficulty


@dataclass
class GymInstance:
    """
    Unified representation of an executable SWE instance (OpenSWE / Scale-SWE / SWE-bench).
    """
    instance_id: str
    repo: str
    problem_statement: str
    source: str = "scale_swe"                 # "scale_swe" | "openswe" | "swe_bench"
    base_commit: str = ""
    workdir: str = "/testbed"
    docker_image: str = "sword-coding-gym:latest"
    pre_commands: List[str] = field(default_factory=list)
    golden_patch: str = ""
    test_patch: str = ""
    f2p_patch: str = ""
    f2p_script: str = ""
    eval_script: str = ""
    fail_to_pass: List[str] = field(default_factory=list)
    pass_to_pass: List[str] = field(default_factory=list)
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataset_row(self) -> DatasetRow:
        """
        Converts this GymInstance into a DatasetRow for the Sword Continual RL streaming queue.
        """
        prompt = (
            f"Repository: {self.repo}\n"
            f"Problem Statement:\n{self.problem_statement.strip()}\n\n"
            "Please solve this issue. Provide your fix as a surgical unified diff inside ```diff ... ``` code block."
        )
        return DatasetRow(
            problem_id=self.instance_id,
            user_problem=prompt,
            domain=DomainType.CODE,
            effort_tier=EffortTier.HIGH,
            difficulty=Difficulty.HARD if (len(self.fail_to_pass) > 2) else Difficulty.MEDIUM,
            reference_answer=self.golden_patch,
            task_scope=[],
            extra_metadata={
                "gym_instance": self.to_dict(),
                "source": self.source,
                "docker_image": self.docker_image,
                "workdir": self.workdir,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "problem_statement": self.problem_statement,
            "source": self.source,
            "base_commit": self.base_commit,
            "workdir": self.workdir,
            "docker_image": self.docker_image,
            "pre_commands": self.pre_commands,
            "golden_patch": self.golden_patch,
            "test_patch": self.test_patch,
            "f2p_patch": self.f2p_patch,
            "f2p_script": self.f2p_script,
            "eval_script": self.eval_script,
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass,
            "language": self.language,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GymInstance:
        return cls(
            instance_id=data["instance_id"],
            repo=data.get("repo", ""),
            problem_statement=data.get("problem_statement", ""),
            source=data.get("source", "scale_swe"),
            base_commit=data.get("base_commit", ""),
            workdir=data.get("workdir", "/testbed"),
            docker_image=data.get("docker_image") or data.get("image_url") or "sword-coding-gym:latest",
            pre_commands=data.get("pre_commands", []),
            golden_patch=data.get("golden_patch") or data.get("patch", ""),
            test_patch=data.get("test_patch", ""),
            f2p_patch=data.get("f2p_patch", ""),
            f2p_script=data.get("f2p_script", ""),
            eval_script=data.get("eval_script", ""),
            fail_to_pass=data.get("fail_to_pass") or data.get("FAIL_TO_PASS") or [],
            pass_to_pass=data.get("pass_to_pass") or data.get("PASS_TO_PASS") or [],
            language=data.get("language", "python"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GymExecutionResult:
    """
    Detailed execution result returned from running tests in the Docker coding gym.
    """
    instance_id: str
    success: bool
    reward: float
    exit_code: int = 0
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    patch_applied: bool = True
    patch_error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "success": self.success,
            "reward": self.reward,
            "exit_code": self.exit_code,
            "f2p_passed": self.f2p_passed,
            "f2p_total": self.f2p_total,
            "p2p_passed": self.p2p_passed,
            "p2p_total": self.p2p_total,
            "patch_applied": self.patch_applied,
            "patch_error": self.patch_error,
            "stdout": self.stdout[-2000:] if self.stdout else "",
            "stderr": self.stderr[-2000:] if self.stderr else "",
            "duration_sec": self.duration_sec,
            "details": self.details,
        }
