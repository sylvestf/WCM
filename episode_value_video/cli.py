from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .curves import load_episode_curves, select_curves
from .pipeline import evaluation_settings_from_args, run_existing_evaluation
from .render import RenderOptions, render_episodes
from .sources import (
    EpisodeSourceRepository,
    LeRobotDatasetRepository,
    LeRobotShardRepository,
    VideoMapRepository,
    VideoTemplateRepository,
)
from .video_io import available_backend


def _add_curve_selection(parser: argparse.ArgumentParser, *, pipeline: bool = False) -> None:
    if not pipeline:
        inputs = parser.add_mutually_exclusive_group(required=True)
        inputs.add_argument(
            "--curves",
            help="Path to the existing evaluator's episode_curves.json.",
        )
        inputs.add_argument(
            "--eval-output",
            help="Evaluation output directory containing episode_curves/episode_curves.json.",
        )
    parser.add_argument(
        "--episode-id",
        type=int,
        action="append",
        help="Render only this episode id; repeat for multiple episodes.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Render at most this many selected/sorted episodes.",
    )


def _add_data_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        help="Override checkpoint config.data.root for LeRobotDataset loading.",
    )
    parser.add_argument("--repo-id", help="Override checkpoint dataset repo_id.")
    parser.add_argument("--revision", help="Override checkpoint dataset revision.")
    parser.add_argument(
        "--camera-key",
        help="Background camera feature. Defaults to the checkpoint's first image_key.",
    )
    parser.add_argument(
        "--source-fps",
        "--fps",
        dest="source_fps",
        type=float,
        help=(
            "Override the input/source FPS used for metadata and timestamp alignment. "
            "The legacy --fps spelling is retained as an alias; use --speed or "
            "--output-fps to change playback speed."
        ),
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Dataset frame_index corresponding to video ordinal 0 (video/shard modes).",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        help=(
            "Evaluator history_size for non-checkpoint video sources. Enables independent "
            "validation of the first predicted frame."
        ),
    )


def _add_video_source_group(
    parser: argparse.ArgumentParser,
    *,
    include_checkpoint: bool,
    required: bool,
) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    if include_checkpoint:
        group.add_argument(
            "--checkpoint",
            help=(
                "Checkpoint/deploy.pt used to reopen the exact LeRobotDataset and read original "
                "frames by episode_index/frame_index (recommended)."
            ),
        )
    group.add_argument(
        "--lerobot-root",
        help="LeRobot v3 root; resolve video shards through meta/episodes timestamps.",
    )
    group.add_argument(
        "--video-template",
        help="Episode video template, e.g. /videos/episode-{episode_id:06d}.mp4.",
    )
    group.add_argument(
        "--video-map",
        help="JSON mapping episode ids to video paths and optional fps/frame_offset metadata.",
    )


def _add_visual_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True, help="Directory for MP4 videos and manifests.")
    playback = parser.add_mutually_exclusive_group()
    playback.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help=(
            "Playback-speed multiplier without dropping/duplicating frames: 2.0 is 2x, "
            "0.5 is half speed (default: 1.0)."
        ),
    )
    playback.add_argument(
        "--output-fps",
        type=float,
        help=(
            "Encode at this FPS. Effective speed is output_fps/source_fps; use --speed "
            "when the same multiplier should apply to episodes with different source FPS."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pyav", "ffmpeg"],
        default="auto",
        help="Video decode/encode backend (default: prefer PyAV, then ffmpeg).",
    )
    parser.add_argument("--ffmpeg", help="Explicit ffmpeg executable path.")
    parser.add_argument("--codec", default="h264", help="Video encoder codec (default: h264).")
    parser.add_argument("--crf", type=int, default=18, help="Encoder CRF quality, 0-51 (default: 18).")
    parser.add_argument("--preset", default="medium", help="Encoder speed preset (default: medium).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing episode MP4 files.")
    parser.add_argument(
        "--allow-frame-mismatch",
        action="store_true",
        help="Best-effort render incomplete/debug curves; manifest will mark mapping unverified.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not write one JPEG preview per episode.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["episode", "global"],
        default="episode",
        help="Fixed y-axis per episode or shared across all rendered episodes.",
    )
    parser.add_argument("--y-min", type=float, help="Optional fixed value-axis minimum.")
    parser.add_argument("--y-max", type=float, help="Optional fixed value-axis maximum.")
    parser.add_argument("--accent", default="#61E4FF", help="Curve color as #RRGGBB.")
    parser.add_argument("--title", default="WORLD CRITIC", help="Small top-left video brand label.")
    parser.add_argument("--font", help="Optional TTF/OTF font file.")
    parser.add_argument(
        "--debug-alignment",
        action="store_true",
        help="Burn video ordinal -> frame_index mapping into each frame.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m episode_value_video",
        description=(
            "Render original episode frames with a frame-perfect, predicted-value-only curve. "
            "Existing world_critic files are never modified."
        ),
    )
    parser.add_argument("--version", action="version", version="episode-value-video 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser(
        "render",
        help="Render videos from an existing episode_curves.json artifact.",
    )
    _add_curve_selection(render_parser)
    _add_video_source_group(render_parser, include_checkpoint=True, required=True)
    _add_data_overrides(render_parser)
    _add_visual_options(render_parser)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the unchanged evaluator first, then render episode videos.",
    )
    pipeline_parser.add_argument("--checkpoint", required=True)
    pipeline_parser.add_argument("--eval-output-dir", required=True)
    pipeline_parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    pipeline_parser.add_argument("--eval-batch-size", type=int, default=64)
    pipeline_parser.add_argument("--eval-num-workers", type=int, default=8)
    pipeline_parser.add_argument("--nproc-per-node", type=int, default=1)
    pipeline_parser.add_argument("--expected-world-size", type=int)
    pipeline_parser.add_argument(
        "--max-batches",
        type=int,
        help="Debug only: incomplete episode curves are rejected unless mismatch is allowed.",
    )
    pipeline_parser.add_argument(
        "--max-eval-curve-episodes",
        type=int,
        help=(
            "Limit curve artifacts written by the evaluator before render selection. "
            "Usually leave unset, especially with --episode-id."
        ),
    )
    pipeline_parser.add_argument("--log-every-batches", type=int, default=20)
    _add_curve_selection(pipeline_parser, pipeline=True)
    _add_video_source_group(pipeline_parser, include_checkpoint=False, required=False)
    _add_data_overrides(pipeline_parser)
    _add_visual_options(pipeline_parser)
    return parser


def _curve_path(args: argparse.Namespace) -> Path:
    if getattr(args, "curves", None):
        return Path(args.curves).expanduser().resolve()
    eval_output = Path(args.eval_output).expanduser().resolve()
    return eval_output / "episode_curves" / "episode_curves.json"


def _repository(args: argparse.Namespace, *, pipeline_checkpoint: str | None = None) -> EpisodeSourceRepository:
    checkpoint = getattr(args, "checkpoint", None) or pipeline_checkpoint
    if getattr(args, "lerobot_root", None):
        return LeRobotShardRepository(
            args.lerobot_root,
            camera_key=args.camera_key,
            frame_offset=args.frame_offset,
            fps=args.source_fps,
            backend=args.backend,
            ffmpeg=args.ffmpeg,
            history_size=args.history_size,
        )
    if getattr(args, "video_template", None):
        return VideoTemplateRepository(
            args.video_template,
            frame_offset=args.frame_offset,
            fps=args.source_fps,
            backend=args.backend,
            ffmpeg=args.ffmpeg,
            history_size=args.history_size,
        )
    if getattr(args, "video_map", None):
        return VideoMapRepository(
            args.video_map,
            default_frame_offset=args.frame_offset,
            fps=args.source_fps,
            backend=args.backend,
            ffmpeg=args.ffmpeg,
            history_size=args.history_size,
        )
    if checkpoint:
        return LeRobotDatasetRepository(
            checkpoint,
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            revision=args.revision,
            camera_key=args.camera_key,
            fps=args.source_fps,
        )
    raise ValueError("No episode frame source was selected.")


def _render_options(args: argparse.Namespace) -> RenderOptions:
    return RenderOptions(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        speed=args.speed,
        output_fps=args.output_fps,
        backend=args.backend,
        ffmpeg=args.ffmpeg,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        overwrite=args.overwrite,
        allow_frame_mismatch=args.allow_frame_mismatch,
        write_preview=not args.no_preview,
        scale_mode=args.scale_mode,
        y_min=args.y_min,
        y_max=args.y_max,
        accent=args.accent,
        title=args.title,
        font_path=args.font,
        debug_alignment=args.debug_alignment,
    )


def _execute_render(args: argparse.Namespace, curve_path: Path) -> dict[str, Any]:
    curves = select_curves(
        load_episode_curves(curve_path),
        episode_ids=args.episode_id,
        max_episodes=args.max_episodes,
    )
    repository = _repository(args, pipeline_checkpoint=getattr(args, "checkpoint", None))
    return render_episodes(
        curves,
        repository,
        options=_render_options(args),
        curve_artifact=curve_path,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.source_fps is not None and (
        not math.isfinite(args.source_fps) or args.source_fps <= 0
    ):
        raise ValueError("--source-fps/--fps must be a finite positive number.")
    if args.output_fps is not None and (
        not math.isfinite(args.output_fps) or args.output_fps <= 0
    ):
        raise ValueError("--output-fps must be a finite positive number.")
    if not math.isfinite(args.speed) or args.speed <= 0:
        raise ValueError("--speed must be a finite positive number.")
    if args.history_size is not None and args.history_size < 1:
        raise ValueError("--history-size must be positive.")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive.")
    if getattr(args, "max_eval_curve_episodes", None) is not None and args.max_eval_curve_episodes < 1:
        raise ValueError("--max-eval-curve-episodes must be positive.")
    if args.command == "pipeline":
        # Fail before a potentially long distributed evaluation when the
        # second (video) stage cannot encode in this environment.
        available_backend(args.backend, ffmpeg=args.ffmpeg)
        curve_path = run_existing_evaluation(**evaluation_settings_from_args(args))
    else:
        curve_path = _curve_path(args)
    result = _execute_render(args, curve_path)
    print(json.dumps({"ok": True, **result}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
