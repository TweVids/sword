"""
Dataset Adapters for GAIR/OpenSWE and AweAI-Team/Scale-SWE.
Parses raw dataset files into standard GymInstance objects and DatasetRows.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Generator

from sword.gym.schema import GymInstance
from sword.rl.schema import DatasetRow


class OpenSWEAdapter:
    """
    Adapter for GAIR-NLP/OpenSWE dataset schema (daVinci-Env).
    """

    @staticmethod
    def parse_item(data: Dict[str, Any]) -> GymInstance:
        instance_id = str(data.get("instance_id", ""))
        repo = data.get("repo", "")
        problem = data.get("problem_statement", "")
        base_commit = data.get("base_commit", "")
        golden_patch = data.get("patch", "")
        test_patch = data.get("test_patch", "")
        eval_script = data.get("eval_script", "")
        language = data.get("language", "python")

        # OpenSWE docker image naming convention
        docker_image = data.get("docker_image") or data.get("image_name") or "sword-coding-gym:latest"

        return GymInstance(
            instance_id=instance_id,
            repo=repo,
            problem_statement=problem,
            source="openswe",
            base_commit=base_commit,
            workdir=data.get("workdir", "/testbed"),
            docker_image=docker_image,
            golden_patch=golden_patch,
            test_patch=test_patch,
            eval_script=eval_script,
            language=language,
            metadata={
                "end_commit": data.get("end_commit", ""),
                "Dockerfile": data.get("Dockerfile", ""),
            },
        )

    @classmethod
    def load_file(cls, path: Union[str, Path], limit: Optional[int] = None) -> List[GymInstance]:
        instances = []
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"OpenSWE file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                instances.append(cls.parse_item(item))
                if limit and len(instances) >= limit:
                    break
        return instances


class ScaleSWEAdapter:
    """
    Adapter for AweAI-Team/Scale-SWE dataset schema (and BeyondSWE/AweAgent).
    """

    @staticmethod
    def parse_item(data: Dict[str, Any]) -> GymInstance:
        instance_id = str(data.get("instance_id", ""))
        repo = data.get("repo", "")
        user = data.get("user", "")
        if user and "/" not in repo:
            full_repo = f"{user}/{repo}"
        else:
            full_repo = repo

        problem = data.get("problem_statement", "")
        golden_patch = data.get("patch", "")
        f2p_patch = data.get("f2p_patch", "")
        f2p_script = data.get("f2p_script", "")

        # Parse test list fields (may be json string or list)
        f2p = data.get("FAIL_TO_PASS") or data.get("fail_to_pass", [])
        if isinstance(f2p, str):
            try:
                f2p = json.loads(f2p)
            except Exception:
                f2p = [t.strip() for t in f2p.split(",") if t.strip()]

        p2p = data.get("PASS_TO_PASS") or data.get("pass_to_pass", [])
        if isinstance(p2p, str):
            try:
                p2p = json.loads(p2p)
            except Exception:
                p2p = [t.strip() for t in p2p.split(",") if t.strip()]

        pre_cmds = data.get("pre_commands", [])
        if isinstance(pre_cmds, str):
            pre_cmds = [pre_cmds]

        docker_image = data.get("image_url") or data.get("docker_image") or "sword-coding-gym:latest"
        workdir = data.get("workdir", "/testbed")

        return GymInstance(
            instance_id=instance_id,
            repo=full_repo,
            problem_statement=problem,
            source="scale_swe",
            base_commit=data.get("parent_commit") or data.get("base_commit", ""),
            workdir=workdir,
            docker_image=docker_image,
            pre_commands=pre_cmds,
            golden_patch=golden_patch,
            f2p_patch=f2p_patch,
            f2p_script=f2p_script,
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            language=data.get("language", "python"),
            metadata={
                "pr_commit": data.get("pr_commit", ""),
                "github_url": data.get("github_url", ""),
            },
        )

    @classmethod
    def load_file(cls, path: Union[str, Path], limit: Optional[int] = None) -> List[GymInstance]:
        instances = []
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scale-SWE file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                instances.append(cls.parse_item(item))
                if limit and len(instances) >= limit:
                    break
        return instances


def load_swe_dataset(
    path: Union[str, Path],
    format: str = "auto",
    limit: Optional[int] = None
) -> List[GymInstance]:
    """
    Universal SWE dataset loader.
    Automatically detects whether the file is OpenSWE or Scale-SWE format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    # Inspect first non-empty line
    first_item = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                first_item = json.loads(line)
                break

    if format == "auto":
        if "FAIL_TO_PASS" in first_item or "f2p_script" in first_item or "f2p_patch" in first_item:
            format = "scale_swe"
        elif "eval_script" in first_item or "Dockerfile" in first_item:
            format = "openswe"
        else:
            format = "scale_swe"

    if format == "openswe":
        return OpenSWEAdapter.load_file(path, limit=limit)
    else:
        return ScaleSWEAdapter.load_file(path, limit=limit)
