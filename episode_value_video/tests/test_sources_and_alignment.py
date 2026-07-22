from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from episode_value_video.curves import EpisodeCurve
from episode_value_video.render import AlignmentError, _preflight_alignment
from episode_value_video.sources import (
    DatasetEpisodeFrameSource,
    EpisodeFrameSource,
    SourceResolutionError,
    VideoMapRepository,
)
from episode_value_video.video_io import VideoProbe


class _DummySource(EpisodeFrameSource):
    episode_id = 1
    fps = 10.0
    frame_count = 8
    first_frame_index = 0
    last_frame_index = 7
    camera_key = None
    description = "dummy"
    alignment_basis = "test"
    expected_first_curve_frame = 2

    def frames(self):
        return iter(())


class SourceAndAlignmentTests(unittest.TestCase):
    def test_current_evaluator_contract_passes(self) -> None:
        curve = EpisodeCurve(
            episode_id=1,
            frame_indices=(2, 3, 4, 5, 6),
            values=(-0.9, -0.8, -0.7, -0.6, -0.5),
        )
        warnings = _preflight_alignment(curve, _DummySource(), allow_mismatch=False)
        self.assertEqual(warnings, [])

    def test_incomplete_curve_fails_strict_alignment(self) -> None:
        curve = EpisodeCurve(
            episode_id=1,
            frame_indices=(2, 3, 4),
            values=(-0.9, -0.8, -0.7),
        )
        with self.assertRaisesRegex(AlignmentError, "incomplete debug subset"):
            _preflight_alignment(curve, _DummySource(), allow_mismatch=False)

    def test_dataset_source_yields_actual_frame_indices(self) -> None:
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        dataset = [
            {
                "episode_index": 3,
                "frame_index": frame_index,
                "camera": image,
            }
            for frame_index in range(5, 8)
        ]
        source = DatasetEpisodeFrameSource(
            dataset=dataset,
            row_start=0,
            row_end=3,
            episode_id=3,
            fps=10,
            first_frame_index=5,
            last_frame_index=7,
            camera_key="camera",
            description="fake",
        )
        frames = list(source.frames())
        self.assertEqual([frame for frame, _ in frames], [5, 6, 7])

    def test_dataset_source_rejects_frame_gap(self) -> None:
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        dataset = [
            {"episode_index": 3, "frame_index": 5, "camera": image},
            {"episode_index": 3, "frame_index": 7, "camera": image},
        ]
        source = DatasetEpisodeFrameSource(
            dataset=dataset,
            row_start=0,
            row_end=2,
            episode_id=3,
            fps=10,
            first_frame_index=5,
            last_frame_index=7,
            camera_key="camera",
            description="fake",
        )
        with self.assertRaises(SourceResolutionError):
            list(source.frames())

    def test_video_map_segment_infers_length_after_seek(self) -> None:
        curve = EpisodeCurve(
            episode_id=1,
            frame_indices=(2, 3, 4, 5, 6),
            values=(-0.9, -0.8, -0.7, -0.6, -0.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "shard.mp4"
            video.write_bytes(b"placeholder")
            mapping = root / "videos.json"
            mapping.write_text(
                json.dumps({"1": {"path": "shard.mp4", "start_seconds": 2.0}}),
                encoding="utf-8",
            )
            probe = VideoProbe(width=320, height=180, fps=10.0, frame_count=100, backend="fake")
            with patch("episode_value_video.sources.probe_video", return_value=probe):
                repository = VideoMapRepository(
                    mapping,
                    history_size=3,
                    backend="ffmpeg",
                    ffmpeg="ffmpeg",
                )
                source = repository.open_episode(curve)
        self.assertEqual(source.frame_count, 8)
        self.assertEqual(source.decode_frame_limit, 8)
        self.assertTrue(source.frame_count_inferred)


if __name__ == "__main__":
    unittest.main()
