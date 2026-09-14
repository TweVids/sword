"""
Comprehensive Test Suite for Sword Coding Gym (OpenSWE & Scale-SWE).
Tests:
1. Unified diff extraction
2. GymInstance and GymExecutionResult schemas
3. OpenSWE and Scale-SWE adapters
4. Universal dataset loading and queue ingestion
5. DockerCodingGym container pool and execution on sword-coding-gym:latest
6. End-to-end patch application and test evaluation (Success, Failure, Syntax Error)
7. Batch rollout evaluation
8. PrimaryScorer integration with DockerCodingGym
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import json
import tempfile
import shutil

from sword.gym.schema import GymInstance, GymExecutionResult
from sword.gym.adapters import OpenSWEAdapter, ScaleSWEAdapter, load_swe_dataset
from sword.gym.docker_gym import DockerCodingGym, extract_unified_diff
from sword.rl.schema import DatasetRow, DomainType, Trajectory
from sword.rl.queue import ContinuousStreamingQueue
from sword.rl.scorer import PrimaryScorer


class TestDockerGym(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        # Initialize gym with sword-coding-gym:latest
        cls.gym = DockerCodingGym(
            default_image="sword-coding-gym:latest",
            pool_size=1,
            timeout=30,
        )

    @classmethod
    def tearDownClass(cls):
        cls.gym.close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # =========================================================
    # 1. Diff Extraction
    # =========================================================
    def test_extract_unified_diff(self):
        # Markdown fenced diff
        md_text = (
            "Here is the fix for the bug:\n"
            "```diff\n"
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-def add(a, b): return a - b\n"
            "+def add(a, b): return a + b\n"
            "```\n"
            "Hope this helps!"
        )
        extracted = extract_unified_diff(md_text)
        self.assertTrue(extracted.startswith("--- a/calc.py"))
        self.assertIn("+def add(a, b): return a + b", extracted)
        self.assertNotIn("```", extracted)

        # Raw diff without markdown fences
        raw_diff = (
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "-def add(a, b): return a - b\n"
            "+def add(a, b): return a + b\n"
        )
        self.assertEqual(extract_unified_diff(raw_diff), raw_diff.strip() + "\n")

    # =========================================================
    # 2. Schema Serialization & DatasetRow Conversion
    # =========================================================
    def test_gym_instance_schema_and_conversion(self):
        inst = GymInstance(
            instance_id="psf__requests-1234",
            repo="psf/requests",
            problem_statement="Fix connection pooling retry bug.",
            source="scale_swe",
            fail_to_pass=["test_retry_logic"],
            pass_to_pass=["test_basic_get"],
            workdir="/testbed",
            golden_patch="diff --git a/requests/pool.py b/requests/pool.py",
        )
        row = inst.to_dataset_row()
        self.assertEqual(row.problem_id, "psf__requests-1234")
        self.assertEqual(row.domain, DomainType.CODE)
        self.assertIn("psf/requests", row.user_problem)
        self.assertIn("Fix connection pooling retry bug", row.user_problem)
        self.assertIn("gym_instance", row.extra_metadata)
        self.assertEqual(row.extra_metadata["source"], "scale_swe")

        # Roundtrip from dict
        inst2 = GymInstance.from_dict(inst.to_dict())
        self.assertEqual(inst2.instance_id, inst.instance_id)
        self.assertEqual(inst2.fail_to_pass, ["test_retry_logic"])

    # =========================================================
    # 3. OpenSWE and Scale-SWE Adapters
    # =========================================================
    def test_openswe_adapter(self):
        raw_openswe = {
            "instance_id": "bottlepy__bottle-456",
            "repo": "bottlepy/bottle",
            "base_commit": "abc1234",
            "problem_statement": "Header escaping issue in HTTP response.",
            "patch": "--- a/bottle.py\n+++ b/bottle.py\n@@ -1 +1 @@\n",
            "test_patch": "--- a/test.py\n+++ b/test.py\n",
            "eval_script": "pytest tests/test_http.py\necho OPENSWE_EXIT_CODE: $?",
            "language": "python",
        }
        inst = OpenSWEAdapter.parse_item(raw_openswe)
        self.assertEqual(inst.instance_id, "bottlepy__bottle-456")
        self.assertEqual(inst.source, "openswe")
        self.assertIn("OPENSWE_EXIT_CODE", inst.eval_script)

    def test_scaleswe_adapter(self):
        raw_scaleswe = {
            "instance_id": "pallets_flask_pr789",
            "user": "pallets",
            "repo": "flask",
            "workdir": "/testbed",
            "problem_statement": "Blueprint route collision.",
            "patch": "--- a/src/flask/blueprints.py\n+++ b/src/flask/blueprints.py\n",
            "f2p_script": "def test_collision(): assert True",
            "FAIL_TO_PASS": json.dumps(["test_collision"]),
            "PASS_TO_PASS": json.dumps(["test_app_init"]),
        }
        inst = ScaleSWEAdapter.parse_item(raw_scaleswe)
        self.assertEqual(inst.instance_id, "pallets_flask_pr789")
        self.assertEqual(inst.repo, "pallets/flask")
        self.assertEqual(inst.source, "scale_swe")
        self.assertEqual(inst.fail_to_pass, ["test_collision"])
        self.assertEqual(inst.pass_to_pass, ["test_app_init"])

    # =========================================================
    # 4. Universal SWE Dataset Ingestion into RL Queue
    # =========================================================
    def test_queue_swe_dataset_ingestion(self):
        # Create a sample JSONL file with mixed OpenSWE and Scale-SWE tasks
        dataset_path = os.path.join(self.temp_dir, "mixed_swe.jsonl")
        samples = [
            {
                "instance_id": "swe_1",
                "repo": "org/repo1",
                "problem_statement": "Bug 1",
                "FAIL_TO_PASS": ["test_f1"],
                "PASS_TO_PASS": ["test_p1"],
            },
            {
                "instance_id": "swe_2",
                "repo": "org/repo2",
                "problem_statement": "Bug 2",
                "eval_script": "pytest",
            },
        ]
        with open(dataset_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        queue = ContinuousStreamingQueue(cache_dir=self.temp_dir)
        count = queue.add_data_source(dataset_path)
        self.assertEqual(count, 2)
        batch = queue.get_batch(2)
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].domain, DomainType.CODE)
        self.assertEqual(batch[1].domain, DomainType.CODE)

    # =========================================================
    # 5. Docker Gym End-to-End Execution on sword-coding-gym:latest
    # =========================================================
    def test_docker_gym_end_to_end_success(self):
        """Simulates a task where a model provides a correct patch fixing a failing test."""
        # 1. Setup instance with initial buggy repository in container
        pre_cmds = [
            "git init .",
            "git checkout -b main",
            "echo 'def add(a, b): return a - b' > math_util.py",
            "git add math_util.py && git commit -m 'Initial commit'",
        ]
        # Failing reproduction test (FAIL_TO_PASS)
        f2p_script = (
            "import math_util\n"
            "def test_addition():\n"
            "    assert math_util.add(2, 3) == 5\n"
        )
        instance = GymInstance(
            instance_id="test_task_add",
            repo="local/math_util",
            problem_statement="add(a, b) incorrectly subtracts instead of adding.",
            source="scale_swe",
            workdir="/testbed",
            docker_image="sword-coding-gym:latest",
            pre_commands=pre_cmds,
            f2p_script=f2p_script,
            fail_to_pass=["test_fail_to_pass.py::test_addition"],
        )

        # 2. Correct patch
        correct_patch = (
            "```diff\n"
            "--- a/math_util.py\n"
            "+++ b/math_util.py\n"
            "@@ -1 +1 @@\n"
            "-def add(a, b): return a - b\n"
            "+def add(a, b): return a + b\n"
            "```"
        )
        res = self.gym.evaluate_patch(instance, correct_patch)
        self.assertTrue(res.patch_applied, f"Patch apply failed: {res.patch_error}")
        self.assertTrue(res.success, f"Test failed with stdout: {res.stdout}, stderr: {res.stderr}")
        self.assertEqual(res.reward, 1.0)
        self.assertEqual(res.f2p_passed, 1)

    def test_docker_gym_broken_patch_fails(self):
        """Simulates a model providing an incorrect patch that fails tests."""
        pre_cmds = [
            "git init .",
            "git checkout -b main",
            "echo 'def add(a, b): return a - b' > math_util.py",
            "git add math_util.py && git commit -m 'Initial commit'",
        ]
        f2p_script = (
            "import math_util\n"
            "def test_addition():\n"
            "    assert math_util.add(2, 3) == 5\n"
        )
        instance = GymInstance(
            instance_id="test_task_add_fail",
            repo="local/math_util",
            problem_statement="add(a, b) incorrectly subtracts.",
            source="scale_swe",
            workdir="/testbed",
            docker_image="sword-coding-gym:latest",
            pre_commands=pre_cmds,
            f2p_script=f2p_script,
            fail_to_pass=["test_fail_to_pass.py::test_addition"],
        )

        # Broken patch (still wrong answer: 0)
        wrong_patch = (
            "```diff\n"
            "--- a/math_util.py\n"
            "+++ b/math_util.py\n"
            "@@ -1 +1 @@\n"
            "-def add(a, b): return a - b\n"
            "+def add(a, b): return 0\n"
            "```"
        )
        res = self.gym.evaluate_patch(instance, wrong_patch)
        self.assertTrue(res.patch_applied)
        self.assertFalse(res.success)
        self.assertEqual(res.f2p_passed, 0)
        self.assertLess(res.reward, 0.5)

    def test_docker_gym_corrupted_patch(self):
        """Simulates a patch with invalid diff syntax that cannot be applied."""
        instance = GymInstance(
            instance_id="test_task_corrupted",
            repo="local/test",
            problem_statement="test",
            workdir="/testbed",
            docker_image="sword-coding-gym:latest",
            pre_commands=["git init .", "echo hello > test.txt", "git add . && git commit -m init"],
        )
        corrupted_patch = "I changed the code from foo to bar without diff format."
        res = self.gym.evaluate_patch(instance, corrupted_patch)
        self.assertFalse(res.patch_applied)
        self.assertFalse(res.success)
        self.assertEqual(res.reward, -0.2)

    # =========================================================
    # 6. Batch Rollout Evaluation (GRPO Concurrency)
    # =========================================================
    def test_docker_gym_batch_rollout_evaluation(self):
        """Tests concurrent evaluation of multiple candidate rollouts for GRPO."""
        pre_cmds = [
            "git init .",
            "git checkout -b main",
            "echo 'def mult(a, b): return 0' > calc.py",
            "git add calc.py && git commit -m 'Initial commit'",
        ]
        f2p_script = (
            "import calc\n"
            "def test_mult():\n"
            "    assert calc.mult(3, 4) == 12\n"
        )
        instance = GymInstance(
            instance_id="test_batch_eval",
            repo="local/calc",
            problem_statement="Fix mult function.",
            workdir="/testbed",
            docker_image="sword-coding-gym:latest",
            pre_commands=pre_cmds,
            f2p_script=f2p_script,
            fail_to_pass=["test_fail_to_pass.py::test_mult"],
        )

        rollouts = [
            # Rollout 0: correct fix
            "```diff\n--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-def mult(a, b): return 0\n+def mult(a, b): return a * b\n```",
            # Rollout 1: wrong fix
            "```diff\n--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-def mult(a, b): return 0\n+def mult(a, b): return a + b\n```",
        ]

        results = self.gym.evaluate_batch(instance, rollouts, max_workers=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].reward, 1.0)
        self.assertFalse(results[1].success)

    # =========================================================
    # 7. PrimaryScorer Integration with DockerCodingGym
    # =========================================================
    def test_primary_scorer_with_docker_gym(self):
        scorer = PrimaryScorer(gym=self.gym)

        pre_cmds = [
            "git init .",
            "git checkout -b main",
            "echo 'def power(a, b): return 1' > math_pow.py",
            "git add math_pow.py && git commit -m 'init'",
        ]
        f2p_script = (
            "import math_pow\n"
            "def test_pow():\n"
            "    assert math_pow.power(2, 3) == 8\n"
        )
        instance = GymInstance(
            instance_id="scorer_gym_test",
            repo="local/math_pow",
            problem_statement="Implement power function.",
            workdir="/testbed",
            docker_image="sword-coding-gym:latest",
            pre_commands=pre_cmds,
            f2p_script=f2p_script,
            fail_to_pass=["test_fail_to_pass.py::test_pow"],
        )
        problem = instance.to_dataset_row()

        good_traj = Trajectory(
            prompt=problem.user_problem,
            full_text=(
                "I will fix power in math_pow.py with surgical edit.\n"
                "```diff\n"
                "--- a/math_pow.py\n"
                "+++ b/math_pow.py\n"
                "@@ -1 +1 @@\n"
                "-def power(a, b): return 1\n"
                "+def power(a, b): return a ** b\n"
                "```"
            ),
            reasoning_trace="surgical edit on math_pow.py to fix power",
            final_answer="done",
        )

        scored = scorer.score_trajectory(problem, good_traj)
        self.assertIn("gym_execution", scored.audit_log["code"])
        self.assertTrue(scored.audit_log["code"]["gym_execution"]["success"])
        self.assertEqual(scored.component_scores["execution"], 1.0)
        self.assertGreater(scored.total_reward, 1.0)


if __name__ == "__main__":
    unittest.main()
