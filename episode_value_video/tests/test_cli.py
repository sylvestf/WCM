from __future__ import annotations

import contextlib
import io
import unittest

from episode_value_video.cli import build_parser, main


class CliTimingTests(unittest.TestCase):
    def _base(self) -> list[str]:
        return [
            "render",
            "--curves",
            "curves.json",
            "--video-template",
            "episode-{episode_id}.mp4",
            "--output-dir",
            "videos",
        ]

    def test_speed_multiplier_parses(self) -> None:
        args = build_parser().parse_args([*self._base(), "--speed", "2.0"])
        self.assertEqual(args.speed, 2.0)
        self.assertIsNone(args.output_fps)

    def test_legacy_fps_alias_is_source_fps_not_output_fps(self) -> None:
        args = build_parser().parse_args(
            [*self._base(), "--fps", "10", "--output-fps", "30"]
        )
        self.assertEqual(args.source_fps, 10.0)
        self.assertEqual(args.output_fps, 30.0)
        self.assertEqual(args.speed, 1.0)

    def test_speed_and_output_fps_are_mutually_exclusive(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [*self._base(), "--speed", "2", "--output-fps", "30"]
                )

    def test_invalid_timing_values_fail_before_file_access(self) -> None:
        cases = [
            ("--speed", "0", "--speed"),
            ("--speed", "nan", "--speed"),
            ("--speed", "inf", "--speed"),
            ("--output-fps", "-1", "--output-fps"),
            ("--output-fps", "nan", "--output-fps"),
            ("--source-fps", "0", "--source-fps"),
        ]
        for flag, value, message in cases:
            with self.subTest(flag=flag, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    main([*self._base(), flag, value])


if __name__ == "__main__":
    unittest.main()
