from __future__ import annotations

import unittest

from PIL import Image

from episode_value_video.curves import EpisodeCurve, value_domain
from episode_value_video.overlay import EpisodeOverlayRenderer


class OverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.curve = EpisodeCurve(
            episode_id=5,
            frame_indices=tuple(range(3, 11)),
            values=tuple(-0.8 + index * 0.05 for index in range(8)),
        )
        self.renderer = EpisodeOverlayRenderer(
            size=(640, 360),
            curve=self.curve,
            timeline_first_frame=0,
            timeline_last_frame=11,
            y_domain=value_domain([self.curve]),
            fps=20,
            camera_key="observation.images.front",
        )

    def test_x_mapping_uses_actual_episode_frame_domain(self) -> None:
        chart = self.renderer.layout.chart
        self.assertEqual(self.renderer.x_for_frame(0), chart[0])
        self.assertEqual(self.renderer.x_for_frame(11), chart[2])
        self.assertLess(self.renderer.x_for_frame(3), self.renderer.x_for_frame(4))

    def test_render_preserves_frame_size(self) -> None:
        background = Image.new("RGB", (640, 360), (30, 40, 50))
        output = self.renderer.render(
            background,
            frame_index=7,
            ordinal=7,
            total_frames=12,
        )
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, background.size)

    def test_visible_segments_stop_at_current_frame(self) -> None:
        state = self.curve.frame_state(6)
        segments = self.renderer._visible_segments(state.visible_count)
        flattened = [point for segment in segments for point in segment]
        self.assertEqual(len(flattened), 4)
        last_expected = self.renderer.point_xy(6, self.curve.value_at(6))  # type: ignore[arg-type]
        self.assertEqual(flattened[-1], last_expected)


if __name__ == "__main__":
    unittest.main()
