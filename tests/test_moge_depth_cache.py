import numpy as np

from foreact.dataloader.gr00t_lerobot.depth_cache import (
    MoGeDepthCacheReader,
    depth_cache_path,
)


def _write_episode(root, episode=7, length=20, horizon=16):
    path = depth_cache_path(root, "bridge_orig_1.0.0_lerobot", episode)
    path.parent.mkdir(parents=True)
    depth = np.zeros((length, 4, 6), dtype=np.float16)
    valid = np.zeros_like(depth, dtype=np.bool_)
    available = np.zeros(length, dtype=np.bool_)
    for frame in range(horizon, length):
        depth[frame] = 0.5 + 0.01 * frame
        valid[frame] = True
        available[frame] = True
    np.savez_compressed(
        path,
        depth=depth,
        valid_mask=valid,
        available=available,
        episode_index=np.int64(episode),
        horizon=np.int64(horizon),
    )
    return path


def test_reader_matches_moge_chunk_layout_and_t_plus_16(tmp_path):
    _write_episode(tmp_path)
    reader = MoGeDepthCacheReader(
        tmp_path,
        dataset_name="bridge_orig_1.0.0_lerobot",
        frame_offsets=(16,),
        expected_horizon=16,
        require_depth=True,
        model_size=(4, 6),
    )
    sample = reader.load_sample(7, frame_index=1, episode_length=20)
    assert sample["depth_target"].shape == (1, 1, 4, 6)
    assert sample["depth_target_indices"].tolist() == [17]
    assert sample["depth_frame_mask"].tolist() == [True]
    assert sample["depth_target_is_padded"].tolist() == [False]


def test_episode_tail_reuses_last_real_depth_without_masking_it(tmp_path):
    _write_episode(tmp_path)
    reader = MoGeDepthCacheReader(
        tmp_path,
        dataset_name="bridge_orig_1.0.0_lerobot",
        frame_offsets=(16,),
        expected_horizon=16,
        require_depth=True,
        model_size=(4, 6),
    )
    sample = reader.load_sample(7, frame_index=18, episode_length=20)
    assert sample["depth_target_indices"].tolist() == [19]
    assert sample["depth_target_is_padded"].tolist() == [True]
    # Padding describes the requested temporal offset, not an invalid target:
    # the real final frame remains a valid supervision signal.
    assert sample["depth_frame_mask"].tolist() == [True]
    assert sample["depth_valid_mask"].all()


def test_reader_loads_t_plus_1_target_without_masking_tail(tmp_path):
    _write_episode(tmp_path, horizon=1)
    reader = MoGeDepthCacheReader(
        tmp_path,
        dataset_name="bridge_orig_1.0.0_lerobot",
        frame_offsets=(1,),
        expected_horizon=1,
        require_depth=True,
        model_size=(4, 6),
    )
    sample = reader.load_sample(7, frame_index=5, episode_length=20)
    assert sample["depth_target_indices"].tolist() == [6]
    assert sample["depth_target_is_padded"].tolist() == [False]
    assert sample["depth_frame_mask"].tolist() == [True]

    tail = reader.load_sample(7, frame_index=19, episode_length=20)
    assert tail["depth_target_indices"].tolist() == [19]
    assert tail["depth_target_is_padded"].tolist() == [True]
    assert tail["depth_frame_mask"].tolist() == [True]
    assert tail["depth_valid_mask"].all()


def test_reader_rejects_wrong_cache_horizon(tmp_path):
    _write_episode(tmp_path, horizon=16)
    reader = MoGeDepthCacheReader(
        tmp_path,
        dataset_name="bridge_orig_1.0.0_lerobot",
        frame_offsets=(1,),
        expected_horizon=1,
        require_depth=True,
        model_size=(4, 6),
    )
    try:
        reader.load_sample(7, frame_index=1, episode_length=20)
    except ValueError as error:
        assert "expected 1, got 16" in str(error)
    else:
        raise AssertionError("Expected a cache-horizon mismatch")
