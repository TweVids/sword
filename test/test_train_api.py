import unittest
import sys
from unittest.mock import patch, MagicMock

import sword
from sword import bootstrap_environment, vllm_patcher
from sword.train_entry import resolve_prompts, train


class TestTrainAPI(unittest.TestCase):
    def test_imports_and_aliases(self):
        self.assertTrue(callable(sword.train))
        self.assertTrue(callable(sword.vllm_patcher))
        self.assertTrue(callable(sword.bootstrap_environment))
        hyphen_fn = getattr(sword, "vllm-patcher", None)
        self.assertIsNotNone(hyphen_fn)
        self.assertEqual(hyphen_fn, vllm_patcher)

    def test_resolve_prompts_integer(self):
        prompts = resolve_prompts(4, 4)
        self.assertEqual(len(prompts), 4)
        self.assertIsInstance(prompts[0], str)

    def test_resolve_prompts_auto_string(self):
        prompts = resolve_prompts("interger - system auto generate prompt to match the number of concurrent/string", 8)
        self.assertEqual(len(prompts), 8)
        self.assertEqual(len(set(prompts)), 8)

    def test_resolve_prompts_literal_string(self):
        prompts = resolve_prompts("What is the speed of light?", 4)
        self.assertEqual(len(prompts), 4)
        self.assertEqual(prompts, ["What is the speed of light?"] * 4)

    def test_train_mock_serving(self):
        res = train(
            test=True,
            model="Qwen/Qwen3.5-0.8B",
            concurrent=4,
            prompt=4,
            max_new_tokens=16,
            mock_test=True,
        )
        self.assertTrue(res.get("ok"))
        self.assertEqual(res.get("concurrent"), 4)
        self.assertEqual(len(res.get("responses")), 4)
        self.assertGreater(res.get("aggregate_tps"), 0.0)

    @patch("sword.vllm_patch.bootstrap_environment")
    def test_vllm_patcher_execution(self, mock_bootstrap):
        mock_bootstrap.return_value = {
            "ok": True,
            "restart_kernel_required": True,
            "steps": [
                {"command": "uv sync --exact --active", "returncode": 0, "tail": ""},
                {"command": "uv pip install vllm==0.23.0", "returncode": 0, "tail": ""},
                {"patch_file": "/tmp/vllm/common.py", "patched": True},
            ],
        }
        out = vllm_patcher(verbose=False)
        self.assertTrue(out.get("ok"))
        self.assertEqual(len(out.get("steps")), 3)


if __name__ == "__main__":
    unittest.main()
