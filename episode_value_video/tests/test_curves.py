from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from episode_value_video.curves import (
    CurveDataError,
    EpisodeCurve,
    load_episode_curves,
    select_curves,
    value_domain,
)


class CurveTests(unittest.TestCase):
    def test_load_sorts_actual_frame_keys_and_ignores_returns(self) -> None:
        payload = [
            {
                "episode_id": 4,
                "frame_indices": [5, 3, 4],
                "values": [-0.2, -0.6, -0.4],
                "returns": [999, 999, 999],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_curves.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            curves = load_episode_curves(path)
        self.assertEqual(curves[0].frame_indices, (3, 4, 5))
        self.assertEqual(curves[0].values, (-0.6, -0.4, -0.2))

    def test_future_points_are_never_visible(self) -> None:
        curve = EpisodeCurve(episode_id=1, frame_indices=(2, 3, 4), values=(-0.8, -0.5, -0.2))
        warmup = curve.frame_state(1)
        current = curve.frame_state(3)
        terminal = curve.frame_state(5)
        self.assertEqual(warmup.visible_count, 0)
        self.assertEqual(warmup.status, "warming_up")
        self.assertEqual(current.visible_count, 2)
        self.assertEqual(current.exact_point_index, 1)
        self.assertEqual(terminal.visible_count, 3)
        self.assertIsNone(terminal.exact_point_index)
        self.assertEqual(terminal.status, "terminal_hold")

    def test_duplicate_frames_are_rejected(self) -> None:
        payload = [{"episode_id": 2, "frame_indices": [3, 3], "values": [0.1, 0.2]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CurveDataError, "duplicate frame_index"):
                load_episode_curves(path)

    def test_nonfinite_values_are_rejected(self) -> None:
        with self.assertRaises(CurveDataError):
            EpisodeCurve(episode_id=2, frame_indices=(3,), values=(math.nan,))

    def test_selection_preserves_requested_order(self) -> None:
        curves = [
            EpisodeCurve(episode_id=1, frame_indices=(2,), values=(-0.2,)),
            EpisodeCurve(episode_id=2, frame_indices=(2,), values=(-0.3,)),
        ]
        selected = select_curves(curves, [2, 1])
        self.assertEqual([curve.episode_id for curve in selected], [2, 1])

    def test_constant_curve_gets_nonzero_domain(self) -> None:
        curve = EpisodeCurve(episode_id=1, frame_indices=(2, 3), values=(-0.4, -0.4))
        lower, upper = value_domain([curve])
        self.assertLess(lower, -0.4)
        self.assertGreater(upper, -0.4)

    def test_reversed_explicit_domain_is_rejected(self) -> None:
        curve = EpisodeCurve(episode_id=1, frame_indices=(2,), values=(-0.4,))
        with self.assertRaisesRegex(CurveDataError, "smaller than y_max"):
            value_domain([curve], requested_min=1.0, requested_max=-1.0)


if __name__ == "__main__":
    unittest.main()
