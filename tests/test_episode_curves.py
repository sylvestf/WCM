from __future__ import annotations

import math

import pytest

from world_critic.curves import build_episode_curves, write_episode_curve_artifacts


def test_episode_curves_sort_by_actual_frame_and_keep_targets_aligned() -> None:
    curves = build_episode_curves(
        [
            {"episode_id": 2, "frame_index": 8, "value": 0.8, "return": 0.7},
            {"episode_id": 1, "frame_index": 4, "value": 0.4, "return": 0.5},
            {"episode_id": 1, "frame_index": 2, "value": 0.2, "return": 0.1},
        ]
    )
    assert [curve["episode_id"] for curve in curves] == [1, 2]
    assert curves[0]["frame_indices"] == [2, 4]
    assert curves[0]["values"] == [0.2, 0.4]
    assert curves[0]["returns"] == [0.1, 0.5]
    assert curves[0]["metrics"]["count"] == 2


def test_episode_curves_reject_duplicate_episode_frame() -> None:
    with pytest.raises(ValueError, match="Duplicate episode/frame"):
        build_episode_curves(
            [
                {"episode_id": 1, "frame_index": 2, "value": 0.2, "return": 0.1},
                {"episode_id": 1, "frame_index": 2, "value": 0.3, "return": 0.1},
            ]
        )


def test_episode_curves_nonfinite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        build_episode_curves(
            [{"episode_id": 1, "frame_index": 2, "value": math.nan, "return": 0.1}]
        )


def test_episode_curve_artifacts_have_machine_readable_summary(tmp_path) -> None:
    curves = build_episode_curves(
        [{"episode_id": 3, "frame_index": 4, "value": 0.4, "return": 0.5}]
    )
    summary = write_episode_curve_artifacts(curves, tmp_path, render_plots=False)
    assert (tmp_path / "episode_curves.json").is_file()
    assert (tmp_path / "episode_metrics.json").is_file()
    assert (tmp_path / "episode_curves_summary.json").is_file()
    assert summary["num_points"] == 1
