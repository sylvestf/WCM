from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from episode_value_video.curves import EpisodeCurve
from episode_value_video.render import RenderOptions, render_one_episode, resolve_playback_fps
from episode_value_video.sources import EpisodeFrameSource


class _FrameSource(EpisodeFrameSource):
    episode_id = 9
    fps = 12.0
    frame_count = 8
    first_frame_index = 0
    last_frame_index = 7
    camera_key = "camera"
    description = "synthetic"
    alignment_basis = "unit-test"
    expected_first_curve_frame = 2

    def frames(self):
        for frame_index in range(self.frame_count):
            image = Image.new("RGB", (320, 180), (frame_index * 10, 30, 50))
            yield frame_index, image


class _FakeWriter:
    last_fps = None

    def __init__(self, output_path, *, width, height, backend, **kwargs):
        self.output_path = Path(output_path)
        self.width = width + width % 2
        self.height = height + height % 2
        self.backend = "fake"
        self.fps = kwargs["fps"]
        type(self).last_fps = self.fps
        self.frame_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def write(self, frame):
        self.frame_count += 1

    def close(self):
        pass

    def commit(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_bytes(b"synthetic-video")
        return self.output_path

    def abort(self):
        pass


class _TinyFrameSource(_FrameSource):
    def frames(self):
        for frame_index in range(self.frame_count):
            yield frame_index, Image.new("RGB", (224, 224), (20, 30, 40))


class RenderPipelineTests(unittest.TestCase):
    def test_render_writes_alignment_report_and_preview(self) -> None:
        curve = EpisodeCurve(
            episode_id=9,
            frame_indices=(2, 3, 4, 5, 6),
            values=(-0.9, -0.8, -0.7, -0.6, -0.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("episode_value_video.render.VideoWriter", _FakeWriter):
                result = render_one_episode(
                    curve,
                    _FrameSource(),
                    options=RenderOptions(output_dir=Path(directory)),
                )
            self.assertTrue(Path(result.output_video).is_file())
            self.assertTrue(Path(result.preview_image).is_file())
            self.assertTrue(Path(result.alignment_report).is_file())
            self.assertTrue(result.mapping_verified)
            self.assertEqual(result.rendered_frame_count, 8)
            self.assertEqual(result.source_fps, 12.0)
            self.assertEqual(result.output_fps, 12.0)
            self.assertEqual(result.playback_speed, 1.0)
            self.assertEqual(_FakeWriter.last_fps, 12.0)

    def test_tiny_source_is_upscaled_for_hud_safe_output(self) -> None:
        curve = EpisodeCurve(
            episode_id=9,
            frame_indices=(2, 3, 4, 5, 6),
            values=(-0.9, -0.8, -0.7, -0.6, -0.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("episode_value_video.render.VideoWriter", _FakeWriter):
                result = render_one_episode(
                    curve,
                    _TinyFrameSource(),
                    options=RenderOptions(output_dir=Path(directory)),
                )
        self.assertEqual((result.source_width, result.source_height), (224, 224))
        self.assertGreaterEqual(result.output_width, 320)
        self.assertGreater(result.upscale_factor, 1.0)

    def test_speed_multiplier_changes_only_output_timing(self) -> None:
        curve = EpisodeCurve(
            episode_id=9,
            frame_indices=(2, 3, 4, 5, 6),
            values=(-0.9, -0.8, -0.7, -0.6, -0.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("episode_value_video.render.VideoWriter", _FakeWriter):
                result = render_one_episode(
                    curve,
                    _FrameSource(),
                    options=RenderOptions(output_dir=Path(directory), speed=2.0),
                )
        self.assertEqual(result.rendered_frame_count, 8)
        self.assertEqual(result.source_fps, 12.0)
        self.assertEqual(result.output_fps, 24.0)
        self.assertEqual(result.playback_speed, 2.0)
        self.assertAlmostEqual(result.source_duration_seconds, 8 / 12)
        self.assertAlmostEqual(result.output_duration_seconds, 8 / 24)
        self.assertEqual(_FakeWriter.last_fps, 24.0)

    def test_explicit_output_fps_computes_effective_speed(self) -> None:
        output_fps, speed = resolve_playback_fps(
            12.0,
            RenderOptions(output_dir=Path("unused"), output_fps=30.0),
        )
        self.assertEqual(output_fps, 30.0)
        self.assertEqual(speed, 2.5)

    def test_half_speed_and_ambiguous_programmatic_options(self) -> None:
        output_fps, speed = resolve_playback_fps(
            12.0,
            RenderOptions(output_dir=Path("unused"), speed=0.5),
        )
        self.assertEqual(output_fps, 6.0)
        self.assertEqual(speed, 0.5)
        with self.assertRaisesRegex(ValueError, "either output_fps"):
            resolve_playback_fps(
                12.0,
                RenderOptions(output_dir=Path("unused"), speed=2.0, output_fps=30.0),
            )


if __name__ == "__main__":
    unittest.main()
