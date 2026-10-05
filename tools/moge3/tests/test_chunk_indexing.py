from tools.moge3.moge3_depth.config import target_frame_index, target_frame_indices


def test_regular_episode_uses_t_plus_horizon_and_repeats_last() -> None:
    length = 26
    assert target_frame_index(0, length, 16) == 16
    assert target_frame_index(9, length, 16) == 25
    assert target_frame_index(10, length, 16) == 25
    assert target_frame_index(25, length, 16) == 25
    assert target_frame_indices(length, 16) == list(range(16, 26))


def test_short_episode_only_needs_last_frame() -> None:
    assert target_frame_indices(8, 16) == [7]
    assert target_frame_index(0, 8, 16) == 7
    assert target_frame_index(7, 8, 16) == 7
