from __future__ import annotations

import h5py
import numpy as np

from scripts.h5_to_lerobot import (
    build_episode_shards,
    compute_pi06_returns,
    inspect_h5_layout,
)


def test_pi06_returns_are_recomputed_from_episode_success() -> None:
    raw, normalized, penalty = compute_pi06_returns(
        np.asarray([3, 2]),
        np.asarray([1, 0]),
        failure_penalty=None,
        step_scale=1.0,
        normalization="task_max",
    )

    assert penalty == 3.0
    np.testing.assert_array_equal(raw, np.asarray([-2, -1, 0, -4, -3], dtype=np.float32))
    np.testing.assert_allclose(
        normalized,
        np.asarray([-2 / 3, -1 / 3, 0, -1, -1], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_global_minmax_matches_legacy_h5_scaling() -> None:
    raw, normalized, penalty = compute_pi06_returns(
        np.asarray([2, 2]),
        np.asarray([1, 0]),
        failure_penalty=300.0,
        step_scale=1.0,
        normalization="global_minmax",
    )

    assert penalty == 300.0
    np.testing.assert_array_equal(raw, np.asarray([-1, 0, -301, -300], dtype=np.float32))
    assert float(normalized.min()) == -1.0
    assert float(normalized.max()) == 1.0


def test_episode_shards_never_split_an_episode() -> None:
    shards = build_episode_shards(np.asarray([3, 4, 2, 8]), max_frames=7)
    assert [
        (shard.episode_start, shard.episode_end, shard.frame_start, shard.frame_end)
        for shard in shards
    ] == [
        (0, 2, 0, 7),
        (2, 3, 7, 9),
        (3, 4, 9, 17),
    ]


def test_h5_layout_accepts_the_requested_schema_and_does_not_read_value(tmp_path) -> None:
    path = tmp_path / "maniskill.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((5, 7), dtype=np.float32))
        h5_file.create_dataset("state", data=np.zeros((5, 9), dtype=np.float32))
        h5_file.create_dataset("pixels", data=np.zeros((5, 8, 8, 3), dtype=np.uint8))
        h5_file.create_dataset("ep_len", data=np.asarray([3, 2], dtype=np.int32))
        h5_file.create_dataset("ep_offset", data=np.asarray([0, 3], dtype=np.int64))
        h5_file.create_dataset("ep_success", data=np.asarray([1, 0], dtype=np.int8))
        h5_file.create_dataset("step_idx", data=np.asarray([0, 1, 2, 0, 1], dtype=np.int32))
        h5_file.create_dataset("value", data=np.full((5, 1), np.nan, dtype=np.float32))
        h5_file.create_dataset("value_raw", data=np.full((5, 1), 1e20, dtype=np.float32))

    layout = inspect_h5_layout(path)
    assert layout.total_frames == 5
    assert layout.total_episodes == 2
    assert layout.action_dim == 7
    assert layout.state_dim == 9
    assert layout.image_shape == (8, 8, 3)
    np.testing.assert_array_equal(layout.episode_success, np.asarray([1, 0], dtype=np.int8))
