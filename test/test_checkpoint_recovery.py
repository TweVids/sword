"""Checkpoint atomicity and Hub recovery with storage/API doubles."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sword.checkpoint as checkpoint
from test.task_retry_fakes import Hub, StorageModel, StorageOptimizer, StorageTorch


class TestCheckpointRecovery(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.torch = StorageTorch()
        self.hub = Hub()
        patches = [patch.object(checkpoint, "torch", self.torch),
                   patch.object(checkpoint, "HAS_HF_HUB", True),
                   patch.object(checkpoint, "HfApi", lambda **kwargs: self.hub, create=True),
                   patch.object(checkpoint, "hf_hub_download", self.hub.download, create=True),
                   patch.object(checkpoint.time, "sleep", lambda seconds: None),
                   patch.dict(sys.modules, {"safetensors.torch": None, "numpy": None})]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.directory.cleanup)

    def manager(self, name="local", upload=False, retain=2):
        return checkpoint.CheckpointManager(output_dir=str(self.root / name), num_latest_checkpoint_save=retain,
                                             upload_id="user/run" if upload else None,
                                             is_save_and_upload_checkpoint=upload)

    def test_same_step_save_rewrites_weights_and_restores_optimizer_and_state(self):
        manager = self.manager()
        model, optimizer = StorageModel(2.0), StorageOptimizer(lr=0.1)
        state = {"failure_count": 3}
        manager.save_checkpoint(3, model, optimizer, training_state=state)
        model.weight.value = 9.0
        optimizer.state["momentum"] = [8.0]
        saved = manager.save_checkpoint(3, model, optimizer, training_state=state)
        fresh, fresh_optimizer = StorageModel(0.0), StorageOptimizer()
        restored = manager.load_checkpoint(saved, fresh, fresh_optimizer)
        self.assertEqual(fresh.weight.value, 9.0)
        self.assertEqual(fresh_optimizer.state, optimizer.state)
        self.assertTrue(restored["optimizer_restored"])
        self.assertEqual(restored["training_state"]["failure_count"], 3)
        self.assertNotIn("step", state)

    def test_failed_save_preserves_previous_complete_checkpoint(self):
        manager = self.manager()
        saved = manager.save_checkpoint(2, StorageModel())
        with patch.object(manager, "_write_checkpoint", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manager.save_checkpoint(3, StorageModel())
        self.assertEqual(manager.find_latest_checkpoint("latest"), saved)
        self.assertEqual(list(Path(manager.output_dir).glob(".checkpoint-*")), [])

    def test_incomplete_higher_checkpoint_is_not_discovered(self):
        manager = self.manager()
        saved = manager.save_checkpoint(1, StorageModel())
        incomplete = Path(manager.output_dir) / "checkpoint-999"
        incomplete.mkdir()
        (incomplete / "training_state.json").write_text('{"step":999}')
        self.assertEqual(manager.find_latest_checkpoint("latest"), saved)
        self.assertIsNone(manager.find_latest_checkpoint("999"))

    def test_corrupted_checkpoint_is_rejected_before_loading_weights(self):
        manager = self.manager()
        saved = manager.save_checkpoint(1, StorageModel())
        (Path(saved) / "adapter_model.bin").write_text("corrupt")
        with self.assertRaises(RuntimeError):
            manager.load_checkpoint(saved, StorageModel())

    def test_missing_manifest_file_is_rejected(self):
        manager = self.manager()
        saved = manager.save_checkpoint(1, StorageModel(), StorageOptimizer())
        (Path(saved) / "optimizer.pt").unlink()
        with self.assertRaises(RuntimeError):
            manager.load_checkpoint(saved, StorageModel(), StorageOptimizer())

    def test_dataset_and_completion_log_survive_a_cold_hub_restore(self):
        manager = self.manager(upload=True)
        manager.mark_dataset_completed("row-hash", "row_1")
        snapshot = {"dataset_sha256": "dataset-hash", "records": [{"problem": "A", "solution": "B"}]}
        manager.save_checkpoint(6, StorageModel(), StorageOptimizer(), dataset_snapshot=snapshot)
        self.hub.files["checkpoint-7/training_state.json"] = b'{"step":7}'
        cold = self.manager("cold", upload=True)
        restored = cold.find_latest_checkpoint("latest")
        self.assertEqual(Path(restored).name, "checkpoint-6")
        self.assertEqual(cold.read_dataset_snapshot(restored), snapshot)
        cold.load_checkpoint(restored, StorageModel(), StorageOptimizer())
        self.assertIn("row-hash", cold.get_completed_dataset_hashes())
        self.assertEqual(cold.upload_info()["last_uploaded_step"], 6)
        self.assertEqual(list(Path(cold.output_dir).glob(".resume-*")), [])

    def test_failed_uploads_are_retained_and_retried_after_restart(self):
        manager = self.manager(upload=True, retain=1)
        self.hub.offline = True
        first = manager.save_checkpoint(1, StorageModel())
        manager.save_checkpoint(2, StorageModel())
        status = manager.upload_info()
        self.assertEqual(status["pending_uploads"], ["checkpoint-1", "checkpoint-2"])
        self.assertIsNone(status["last_uploaded_step"])
        self.assertEqual(status["upload_error"], "hub offline")
        self.assertTrue(Path(first).is_dir())
        self.assertEqual(len(self.hub.uploads), 6)
        restarted = self.manager(upload=True, retain=1)
        self.hub.offline = False
        restarted.retry_pending_uploads()
        restarted.prune_local_checkpoints()
        self.assertEqual(restarted.upload_info()["last_uploaded_step"], 2)
        self.assertEqual(restarted.upload_info()["pending_uploads"], [])
        self.assertIsNone(restarted.upload_info()["upload_error"])
        self.assertFalse(Path(first).exists())

    def test_remote_specific_missing_step_returns_no_checkpoint(self):
        manager = self.manager(upload=True)
        manager.save_checkpoint(3, StorageModel())
        cold = self.manager("cold", upload=True)
        self.assertIsNone(cold.find_latest_checkpoint("400"))

    def test_random_state_is_restored_with_the_checkpoint(self):
        manager = self.manager()
        random.seed(123)
        saved_python = random.getstate()
        saved = manager.save_checkpoint(1, StorageModel())
        self.torch.rng = [99]
        random.random()
        restored = manager.load_checkpoint(saved, StorageModel())
        self.assertTrue(restored["rng_restored"])
        self.assertEqual(self.torch.rng, [1, 2, 3])
        self.assertEqual(random.getstate(), saved_python)

    def test_optimizer_restore_failure_is_explicit(self):
        manager = self.manager()
        saved = manager.save_checkpoint(1, StorageModel(), StorageOptimizer())
        incompatible = StorageOptimizer()
        incompatible.fail_restore = True
        with self.assertRaises(RuntimeError):
            manager.load_checkpoint(saved, StorageModel(), incompatible)

    def test_upload_does_not_run_for_an_incomplete_save(self):
        manager = self.manager(upload=True)
        with patch.object(manager, "_write_checkpoint", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manager.save_checkpoint(1, StorageModel())
        self.assertEqual(self.hub.uploads, [])

    def test_cleanup_rejects_paths_outside_output_directory(self):
        manager = self.manager()
        with self.assertRaises(ValueError):
            manager._remove_checkpoint_directory(str(self.root))

    def test_failed_serialization_cannot_mark_a_dataset_complete(self):
        manager = self.manager()
        manager.save_checkpoint(1, StorageModel())
        optimizer = StorageOptimizer()
        with patch.object(optimizer, "state_dict", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manager.save_checkpoint(2, StorageModel(), optimizer, completed_dataset_hash="unfinished")
        self.assertNotIn("unfinished", manager.get_completed_dataset_hashes())

    def test_completion_log_tracks_latest_outcome_after_a_regression(self):
        manager = self.manager()
        state = {"dataset_sha256": "dataset", "visited_row_hashes": ["row"],
                 "task_queue": {"version": 1, "config": {"enabled": True}, "tasks": {
                     "row": {"solved": True, "problem_id": "1"}}}}
        manager.save_checkpoint(1, StorageModel(), training_state=state, completed_dataset_hash="dataset")
        self.assertEqual(manager.get_completed_dataset_hashes(), {"row", "dataset"})
        state["task_queue"]["tasks"]["row"]["solved"] = False
        latest = manager.save_checkpoint(2, StorageModel(), training_state=state)
        self.assertEqual(manager.get_completed_dataset_hashes(), set())
        manager.mark_dataset_completed("row")
        manager.restore_completion_log(latest)
        self.assertNotIn("row", manager.get_completed_dataset_hashes())

    def test_generic_training_metadata_is_preserved(self):
        manager = self.manager()
        state = {"task_queue": {"failures": 3}, "custom": "metadata"}
        saved = manager.save_checkpoint(1, StorageModel(), training_state=state)
        restored = manager.read_training_state(saved)
        self.assertEqual(restored["task_queue"], state["task_queue"])
        self.assertEqual(restored["custom"], "metadata")

    def test_cuda_random_state_restores_matching_devices_after_hardware_change(self):
        manager = self.manager()
        self.torch.cuda.is_initialized = lambda: True
        self.torch.cuda.get_rng_state_all = lambda: [[10], [20]]
        saved = manager.save_checkpoint(1, StorageModel())
        restored_devices = []
        self.torch.cuda.is_available = lambda: True
        self.torch.cuda.device_count = lambda: 1
        self.torch.cuda.set_rng_state = lambda state, device: restored_devices.append((device, state))
        manager.load_checkpoint(saved, StorageModel())
        self.assertEqual(restored_devices, [(0, [10])])


if __name__ == "__main__":
    unittest.main()
