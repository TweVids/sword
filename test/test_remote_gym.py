"""
Test Suite for GymServer and RemoteCodingGym Client (Molab <-> Local Tunnel Protocol).
Tests:
1. Local HTTP GymServer startup and health endpoint
2. RemoteCodingGym client connection and health check
3. Over-the-network evaluate_patch call executing inside local Docker sandbox
4. Over-the-network evaluate_batch call executing concurrent rollouts
5. Instance queue distribution (/instances/next)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import threading
import time
import socket

from sword.gym.schema import GymInstance
from sword.gym.docker_gym import DockerCodingGym
from sword.gym.server import GymServer
from sword.gym.client import RemoteCodingGym


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestRemoteGym(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.port = get_free_port()
        cls.gym = DockerCodingGym(
            default_image="sword-coding-gym:latest",
            pool_size=1,
            timeout=30,
        )
        sample_instance = GymInstance(
            instance_id="seeded_prob_1",
            repo="org/repo",
            problem_statement="Solve seeded problem",
            source="scale_swe",
        )
        cls.server = GymServer(("127.0.0.1", cls.port), gym=cls.gym, instances=[sample_instance])
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

        cls.client = RemoteCodingGym(endpoint=f"http://127.0.0.1:{cls.port}", timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.gym.close()

    def test_health_check(self):
        info = self.client.health()
        self.assertEqual(info["status"], "ok")
        self.assertEqual(info["server"], "Sword-Coding-Gym-Server")
        self.assertTrue(info["docker_available"])

    def test_get_next_problem(self):
        problems = self.client.get_next_problem(n=1)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].instance_id, "seeded_prob_1")

    def test_remote_evaluate_patch(self):
        """Validates that Molab HTTP requests execute in local Docker and return score."""
        instance = GymInstance(
            instance_id="remote_test_sub",
            repo="local/sub",
            problem_statement="Fix subtraction bug",
            workdir="/testbed",
            pre_commands=[
                "git init .",
                "echo 'def sub(a, b): return a + b' > sub.py",
                "git add sub.py && git commit -m 'init'",
            ],
            f2p_script=(
                "import sub\n"
                "def test_subtraction():\n"
                "    assert sub.sub(10, 4) == 6\n"
            ),
            fail_to_pass=["test_fail_to_pass.py::test_subtraction"],
        )

        correct_patch = (
            "```diff\n"
            "--- a/sub.py\n"
            "+++ b/sub.py\n"
            "@@ -1 +1 @@\n"
            "-def sub(a, b): return a + b\n"
            "+def sub(a, b): return a - b\n"
            "```"
        )

        res = self.client.evaluate_patch(instance, correct_patch)
        self.assertTrue(res.patch_applied)
        self.assertTrue(res.success)
        self.assertEqual(res.reward, 1.0)
        self.assertEqual(res.f2p_passed, 1)

    def test_remote_evaluate_batch(self):
        instance = GymInstance(
            instance_id="remote_batch_test",
            repo="local/div",
            problem_statement="Fix div bug",
            workdir="/testbed",
            pre_commands=[
                "git init .",
                "echo 'def div(a, b): return 0' > div.py",
                "git add div.py && git commit -m 'init'",
            ],
            f2p_script=(
                "import div\n"
                "def test_div():\n"
                "    assert div.div(12, 3) == 4\n"
            ),
            fail_to_pass=["test_fail_to_pass.py::test_div"],
        )

        rollouts = [
            "```diff\n--- a/div.py\n+++ b/div.py\n@@ -1 +1 @@\n-def div(a, b): return 0\n+def div(a, b): return a // b\n```",
            "```diff\n--- a/div.py\n+++ b/div.py\n@@ -1 +1 @@\n-def div(a, b): return 0\n+def div(a, b): return a + b\n```",
        ]

        results = self.client.evaluate_batch(instance, rollouts)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].reward, 1.0)
        self.assertFalse(results[1].success)

    def test_remote_state_persistence_and_dynamic_add(self):
        """Validates that adding instances dynamically and saving/loading state works over network."""
        # 1. Add new problem dynamically
        new_inst = GymInstance(
            instance_id="dynamic_problem_42",
            repo="org/new_repo",
            problem_statement="A dynamically added problem",
            source="openswe",
        )
        add_res = self.client.add_instances([new_inst])
        self.assertEqual(add_res["added"], 1)

        # 2. Inspect state
        state = self.client.get_state()
        self.assertGreaterEqual(state["queue_count"], 1)
        self.assertGreaterEqual(state["completed_count"], 1)

        # 3. Save state on host
        save_res = self.client.save_state()
        self.assertEqual(save_res["status"], "saved")
        self.assertTrue(os.path.exists(save_res["path"]))


if __name__ == "__main__":
    unittest.main()
