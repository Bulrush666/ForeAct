"""Run with: python -m unittest discover -s tests -p test_libero_integration.py -v"""
import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_libero_data import check_depth, validate_contract
from launch_libero import resolve_config
from foreact.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from foreact.dataloader.gr00t_lerobot.depth_cache import MoGeDepthCacheReader, depth_cache_path
from foreact.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP, DATASET_NAMED_MIXTURES


class LiberoIntegrationTests(unittest.TestCase):
    def stub(self):
        dataset = object.__new__(LeRobotSingleDataset)
        dataset._dataset_path = Path("/dataset")
        dataset._lerobot_version = "v3.0"
        dataset._chunk_size = 1000
        dataset._lerobot_modality_meta = SimpleNamespace(video={"primary_image": SimpleNamespace(original_key="main"), "wrist_image": SimpleNamespace(original_key="wrist")})
        dataset._video_path_pattern = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        dataset.trajectory_ids_to_metadata = {17: {"data/chunk_index": 1, "data/file_index": 2, "videos/file_indices": {"main": {"chunk_index": 3, "file_index": 4}, "wrist": {"chunk_index": 5, "file_index": 6}}, "videos/from_timestamps": {"main": 12.0, "wrist": 24.0}}}
        dataset._trajectory_ids = np.array([17])
        dataset._trajectory_lengths = np.array([3])
        dataset._delta_indices = {"video.primary_image": np.array([0, 1])}
        dataset.curr_traj_data = pd.DataFrame({"timestamp": [0.0, 0.05, 0.1]})
        dataset.video_backend = "torchvision_av"
        dataset.video_backend_kwargs = {}
        return dataset

    def test_camera_shards_independent_of_action_shards(self):
        dataset = self.stub()
        self.assertEqual(dataset.get_video_path(17, "primary_image").as_posix(), "/dataset/videos/main/chunk-003/file-004.mp4")
        self.assertEqual(dataset.get_video_path(17, "wrist_image").as_posix(), "/dataset/videos/wrist/chunk-005/file-006.mp4")

    def test_legacy_fallback_and_v2_unchanged(self):
        dataset = self.stub()
        dataset.trajectory_ids_to_metadata[17]["videos/file_indices"] = {}
        self.assertTrue(str(dataset.get_video_path(17, "primary_image")).endswith("chunk-001/file-002.mp4"))
        dataset._lerobot_version = "v2.0"
        dataset._video_path_pattern = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        self.assertTrue(str(dataset.get_video_path(17, "primary_image")).endswith("chunk-000/main/episode_000017.mp4"))

    def test_camera_timestamp_and_tail_padding(self):
        dataset = self.stub()
        with patch("foreact.dataloader.gr00t_lerobot.datasets.get_frames_by_timestamps") as decode:
            dataset.get_video(17, "video.primary_image", 2)
        np.testing.assert_allclose(decode.call_args.args[1], [12.1, 12.1])
        self.assertIn("chunk-003/file-004", decode.call_args.args[0])

    def test_registry_isolated_from_two_view_configs(self):
        config = ROBOT_TYPE_CONFIG_MAP["libero_franka_main"]
        self.assertEqual(config.video_keys, ["video.primary_image"])
        self.assertEqual(config.action_indices, list(range(8)))
        self.assertEqual(len(ROBOT_TYPE_CONFIG_MAP["libero_franka"].video_keys), 2)
        mixture = DATASET_NAMED_MIXTURES["libero_all_v30_main"]
        self.assertEqual(len(mixture), 4)
        self.assertEqual(len({item[0] for item in mixture}), 4)
        self.assertTrue(all(item[1:] == (1.0, "libero_franka_main") for item in mixture))

    def test_pack_main_image_eight_actions_no_state(self):
        dataset = self.stub()
        config = ROBOT_TYPE_CONFIG_MAP["libero_franka_main"]
        dataset._modality_keys = {"video": config.video_keys, "action": config.action_keys, "language": config.language_keys}
        dataset.data_cfg = {"include_state": False, "obs_image_size_hw": [224, 224]}
        dataset._depth_cache_reader = None
        dataset.tag = "franka"
        values = {key: np.zeros((8, 1)) for key in config.action_keys}
        values["action.gripper"] = np.arange(8).reshape(8, 1) % 2
        values[config.video_keys[0]] = np.zeros((1, 256, 256, 3), dtype=np.uint8)
        values[config.language_keys[0]] = ["pick up the bowl"]
        sample = dataset._pack_sample(values, 17, 0)
        self.assertEqual(sample["action"].shape, (8, 7))
        self.assertEqual(len(sample["image"]), 1)
        self.assertEqual(sample["image"][0].size, (224, 224))
        self.assertNotIn("state", sample)
        np.testing.assert_array_equal(sample["action"][:, 6], np.arange(8) % 2)

    def test_depth_suite_identity_offset_and_tail(self):
        with TemporaryDirectory() as tmp:
            samples = []
            for suite, value in (("libero_goal_no_noops_lerobot", 1.0), ("libero_10_no_noops_lerobot", 2.0)):
                path = depth_cache_path(tmp, suite, 0)
                path.parent.mkdir(parents=True)
                depth = np.full((3, 2, 3), value, np.float32)
                np.savez(path, depth=depth, valid_mask=np.ones_like(depth, dtype=bool), available=np.ones(3, dtype=bool), episode_index=0, horizon=1)
                check_depth(path, 0, 3, (2, 3))
                reader = MoGeDepthCacheReader(tmp, dataset_name=suite, frame_offsets=(1,), expected_horizon=1, require_depth=True, model_size=(2, 3), normalization="metric_log")
                sample = reader.load_sample(0, 0, 3)
                self.assertEqual(sample["depth_target_indices"].tolist(), [1])
                tail = reader.load_sample(0, 2, 3)
                self.assertEqual(tail["depth_target_indices"].tolist(), [2])
                self.assertEqual(tail["depth_target_is_padded"].tolist(), [True])
                samples.append(sample["depth_target"].numpy())
                with self.assertRaises(ValueError):
                    check_depth(path, 0, 4, (2, 3))
            self.assertFalse(np.array_equal(*samples))

    def test_config_overrides_applied_to_training_and_checks(self):
        config = yaml.safe_load((ROOT / "configs/train/libero.yaml").read_text())
        validate_contract(config)
        resolved = resolve_config(config, {"GPU_IDS": "4,5", "DEPTH_CACHE_ROOT": "/tmp/depth", "DATA_ROOT_DIR": "/tmp/rgb", "PER_DEVICE_BATCH_SIZE": "3", "SEED": "7"})
        self.assertEqual(resolved["runtime"]["nproc_per_node"], 2)
        self.assertEqual(resolved["datasets"]["vla_data"]["depth_cache_root"], "/tmp/depth")
        self.assertEqual(resolved["datasets"]["vla_data"]["data_root_dir"], "/tmp/rgb")
        self.assertEqual(resolved["datasets"]["vla_data"]["per_device_batch_size"], 3)
        self.assertEqual(resolved["seed"], 7)
        config["framework"]["action_model"]["action_horizon"] = 16
        with self.assertRaises(ValueError):
            validate_contract(config)


if __name__ == "__main__":
    unittest.main()
