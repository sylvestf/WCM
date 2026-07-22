from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from .curves import EpisodeCurve, value_domain
from .overlay import EpisodeOverlayRenderer
from .sources import EpisodeFrameSource, EpisodeSourceRepository
from .video_io import VideoWriter


class AlignmentError(RuntimeError):
    """Raised when source frames and evaluator curve keys do not agree."""


@dataclass(slots=True)
class RenderOptions:
    output_dir: Path
    speed: float = 1.0
    output_fps: float | None = None
    backend: str = "auto"
    ffmpeg: str | Path | None = None
    codec: str = "h264"
    crf: int = 18
    preset: str = "medium"
    overwrite: bool = False
    allow_frame_mismatch: bool = False
    write_preview: bool = True
    scale_mode: str = "episode"
    y_min: float | None = None
    y_max: float | None = None
    accent: str = "#61E4FF"
    title: str = "WORLD CRITIC"
    font_path: str | Path | None = None
    debug_alignment: bool = False


@dataclass(slots=True)
class EpisodeRenderResult:
    episode_id: int
    output_video: str
    preview_image: str | None
    alignment_report: str
    source: str
    camera_key: str | None
    source_fps: float
    output_fps: float
    playback_speed: float
    source_duration_seconds: float
    output_duration_seconds: float
    source_frame_count: int
    rendered_frame_count: int
    first_source_frame: int
    last_source_frame: int
    first_curve_frame: int
    last_curve_frame: int
    curve_point_count: int
    warmup_frame_count: int
    terminal_without_new_estimate: bool
    alignment_basis: str
    expected_first_curve_frame: int | None
    first_curve_contract_verified: bool
    frame_count_inferred: bool
    mapping_verified: bool
    warnings: list[str]
    output_width: int
    output_height: int
    source_width: int
    source_height: int
    upscale_factor: float
    encoder_backend: str


def _curve_has_gaps(curve: EpisodeCurve) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left, right in zip(curve.frame_indices, curve.frame_indices[1:])
        if right != left + 1
    ]


def resolve_playback_fps(source_fps: float, options: RenderOptions) -> tuple[float, float]:
    """Return ``(output_fps, effective_speed)`` without changing frame identity.

    Every source frame is still written exactly once. Changing the encoder FPS
    therefore changes wall-clock playback duration while preserving the exact
    ordinal/frame_index/value mapping.
    """

    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Source FPS must be finite and positive, got {source_fps!r}.")
    if not math.isfinite(options.speed) or options.speed <= 0:
        raise ValueError(f"Playback speed must be finite and positive, got {options.speed!r}.")
    if options.output_fps is not None:
        if not math.isclose(options.speed, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Specify either output_fps or a non-default speed, not both.")
        output_fps = float(options.output_fps)
        if not math.isfinite(output_fps) or output_fps <= 0:
            raise ValueError(f"Output FPS must be finite and positive, got {output_fps!r}.")
        return output_fps, output_fps / source_fps
    output_fps = source_fps * options.speed
    if not math.isfinite(output_fps) or output_fps <= 0:
        raise ValueError(
            f"source_fps * speed produced an invalid output FPS: {source_fps} * {options.speed}."
        )
    return output_fps, options.speed


def _preflight_alignment(
    curve: EpisodeCurve,
    source: EpisodeFrameSource,
    *,
    allow_mismatch: bool,
) -> list[str]:
    warnings: list[str] = []
    if curve.first_frame < source.first_frame_index or curve.last_frame > source.last_frame_index:
        raise AlignmentError(
            f"episode_id={curve.episode_id} curve span [{curve.first_frame}, {curve.last_frame}] "
            f"falls outside source span [{source.first_frame_index}, {source.last_frame_index}]."
        )

    gaps = _curve_has_gaps(curve)
    if gaps:
        message = (
            f"episode_id={curve.episode_id} curve has {len(gaps)} frame gap(s), e.g. {gaps[:5]}. "
            "This usually means evaluation was truncated with --max-batches."
        )
        if not allow_mismatch:
            raise AlignmentError(message)
        warnings.append(message)

    expected_last_curve = source.last_frame_index - 1
    if curve.last_frame != expected_last_curve:
        message = (
            f"episode_id={curve.episode_id} last prediction is frame {curve.last_frame}, but the "
            f"current evaluator contract expects source penultimate frame {expected_last_curve}. "
            "The curve may be an incomplete debug subset."
        )
        if not allow_mismatch:
            raise AlignmentError(message)
        warnings.append(message)

    if curve.first_frame < source.first_frame_index:
        raise AlignmentError(
            f"episode_id={curve.episode_id} begins before its source video."
        )
    if source.expected_first_curve_frame is None:
        warnings.append(
            "history_size was unavailable, so the first curve frame could not be "
            "independently distinguished from a truncated-prefix artifact."
        )
    elif curve.first_frame != source.expected_first_curve_frame:
        message = (
            f"episode_id={curve.episode_id} first prediction is frame {curve.first_frame}, "
            f"but source/history_size expects {source.expected_first_curve_frame}."
        )
        if not allow_mismatch:
            raise AlignmentError(message)
        warnings.append(message)
    return warnings


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def render_one_episode(
    curve: EpisodeCurve,
    source: EpisodeFrameSource,
    *,
    options: RenderOptions,
    y_domain: tuple[float, float] | None = None,
    log: Callable[[str], None] = print,
) -> EpisodeRenderResult:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    warnings = _preflight_alignment(
        curve,
        source,
        allow_mismatch=options.allow_frame_mismatch,
    )
    domain = y_domain or value_domain(
        [curve], requested_min=options.y_min, requested_max=options.y_max
    )
    output_fps, playback_speed = resolve_playback_fps(source.fps, options)
    output_path = options.output_dir / f"episode-{curve.episode_id:06d}.mp4"
    preview_path = options.output_dir / f"episode-{curve.episode_id:06d}-preview.jpg"
    report_path = options.output_dir / f"episode-{curve.episode_id:06d}-alignment.json"

    iterator = source.frames()
    try:
        first_frame_index, first_frame = next(iterator)
    except StopIteration as exc:
        raise AlignmentError(f"episode_id={curve.episode_id} source produced no frames.") from exc
    if first_frame_index != source.first_frame_index:
        raise AlignmentError(
            f"episode_id={curve.episode_id} source declared first frame "
            f"{source.first_frame_index}, but yielded {first_frame_index}."
        )
    source_width, source_height = first_frame.size
    upscale_factor = max(1.0, 320 / source_width, 180 / source_height)
    width = max(2, round(source_width * upscale_factor))
    height = max(2, round(source_height * upscale_factor))
    display_size = (width, height)
    renderer = EpisodeOverlayRenderer(
        size=(width, height),
        curve=curve,
        timeline_first_frame=source.first_frame_index,
        timeline_last_frame=source.last_frame_index,
        y_domain=domain,
        fps=source.fps,
        camera_key=source.camera_key,
        title=options.title,
        accent=options.accent,
        font_path=options.font_path,
        debug_alignment=options.debug_alignment,
    )

    preview_target = curve.frame_indices[len(curve.frame_indices) // 2]
    preview_frame: Image.Image | None = None
    last_rendered_frame: Image.Image | None = None
    rendered_count = 0
    last_frame_index: int | None = None
    log(
        f"[value-video] episode={curve.episode_id} frames={source.frame_count} "
        f"curve_points={len(curve.values)} source_fps={source.fps:g} "
        f"output_fps={output_fps:g} speed={playback_speed:g}x source={source.description}"
    )
    writer = VideoWriter(
        output_path,
        width=width,
        height=height,
        fps=output_fps,
        backend=options.backend,
        ffmpeg=options.ffmpeg,
        codec=options.codec,
        crf=options.crf,
        preset=options.preset,
        overwrite=options.overwrite,
    )
    try:
        with writer:
            for ordinal, (frame_index, frame) in enumerate(
                chain([(first_frame_index, first_frame)], iterator)
            ):
                expected_frame = source.first_frame_index + ordinal
                if frame_index != expected_frame:
                    raise AlignmentError(
                        f"episode_id={curve.episode_id} source frame sequence changed at ordinal "
                        f"{ordinal}: expected frame_index={expected_frame}, got {frame_index}."
                    )
                display_frame = (
                    frame
                    if frame.size == display_size
                    else frame.resize(display_size, Image.Resampling.LANCZOS)
                )
                rendered = renderer.render(
                    display_frame,
                    frame_index=frame_index,
                    ordinal=ordinal,
                    total_frames=source.frame_count,
                )
                writer.write(rendered)
                last_rendered_frame = rendered
                if frame_index == preview_target:
                    preview_frame = rendered.copy()
                rendered_count += 1
                last_frame_index = frame_index

            if rendered_count != source.frame_count:
                message = (
                    f"episode_id={curve.episode_id} decoded {rendered_count} frames but source "
                    f"metadata/curve expected {source.frame_count}."
                )
                if not options.allow_frame_mismatch:
                    raise AlignmentError(message)
                warnings.append(message)
            if last_frame_index != source.first_frame_index + rendered_count - 1:
                raise AlignmentError(
                    f"episode_id={curve.episode_id} final frame index is inconsistent with decoded count."
                )
        committed = writer.commit()
    except Exception:
        writer.abort()
        raise

    preview_output: str | None = None
    if options.write_preview:
        if preview_frame is None:
            # This only occurs for a tolerated partial/misaligned source.
            if last_rendered_frame is None:
                raise AlignmentError(f"episode_id={curve.episode_id} produced no previewable frame.")
            preview_frame = last_rendered_frame
        preview_frame.save(preview_path, quality=92, optimize=True)
        preview_output = str(preview_path.resolve())

    result = EpisodeRenderResult(
        episode_id=curve.episode_id,
        output_video=str(committed.resolve()),
        preview_image=preview_output,
        alignment_report=str(report_path.resolve()),
        source=source.description,
        camera_key=source.camera_key,
        source_fps=source.fps,
        output_fps=output_fps,
        playback_speed=playback_speed,
        source_duration_seconds=rendered_count / source.fps,
        output_duration_seconds=rendered_count / output_fps,
        source_frame_count=source.frame_count,
        rendered_frame_count=rendered_count,
        first_source_frame=source.first_frame_index,
        last_source_frame=(
            source.first_frame_index + rendered_count - 1
            if rendered_count
            else source.first_frame_index
        ),
        first_curve_frame=curve.first_frame,
        last_curve_frame=curve.last_frame,
        curve_point_count=len(curve.values),
        warmup_frame_count=max(0, curve.first_frame - source.first_frame_index),
        terminal_without_new_estimate=curve.last_frame < source.last_frame_index,
        alignment_basis=source.alignment_basis,
        expected_first_curve_frame=source.expected_first_curve_frame,
        first_curve_contract_verified=(
            source.expected_first_curve_frame is not None
            and curve.first_frame == source.expected_first_curve_frame
        ),
        frame_count_inferred=source.frame_count_inferred,
        mapping_verified=not warnings,
        warnings=warnings,
        output_width=writer.width,
        output_height=writer.height,
        source_width=source_width,
        source_height=source_height,
        upscale_factor=upscale_factor,
        encoder_backend=writer.backend,
    )
    _write_json(report_path, asdict(result))
    log(
        f"[value-video] wrote episode={curve.episode_id} video={committed} "
        f"mapping_verified={result.mapping_verified}"
    )
    return result


def render_episodes(
    curves: Iterable[EpisodeCurve],
    repository: EpisodeSourceRepository,
    *,
    options: RenderOptions,
    curve_artifact: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    selected = list(curves)
    if not selected:
        raise ValueError("No episode curves were selected for rendering.")
    if options.scale_mode not in {"episode", "global"}:
        raise ValueError("scale_mode must be 'episode' or 'global'.")
    shared_domain = (
        value_domain(
            selected,
            requested_min=options.y_min,
            requested_max=options.y_max,
        )
        if options.scale_mode == "global"
        else None
    )
    results: list[EpisodeRenderResult] = []
    for curve in selected:
        source = repository.open_episode(curve)
        results.append(
            render_one_episode(
                curve,
                source,
                options=options,
                y_domain=shared_domain,
                log=log,
            )
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "curve_artifact": (
            str(Path(curve_artifact).expanduser().resolve())
            if curve_artifact is not None
            else None
        ),
        "output_directory": str(options.output_dir.expanduser().resolve()),
        "num_episodes": len(results),
        "style": {
            "title": options.title,
            "accent": options.accent,
            "timing_mode": (
                "fixed_output_fps" if options.output_fps is not None else "speed_multiplier"
            ),
            "timing_strategy": "one_encoded_frame_per_source_frame",
            "requested_speed": options.speed if options.output_fps is None else None,
            "requested_output_fps": options.output_fps,
            "scale_mode": options.scale_mode,
            "y_min": options.y_min,
            "y_max": options.y_max,
            "axis_domain_source": (
                "explicit_cli"
                if options.y_min is not None or options.y_max is not None
                else "predicted_values_for_selected_scope"
            ),
            "predicted_value_only": True,
            "future_points_hidden": True,
            "exact_current_point_only": True,
        },
        "episodes": [asdict(result) for result in results],
    }
    manifest_path = options.output_dir / "episode_value_videos.json"
    manifest["manifest"] = str(manifest_path.resolve())
    _write_json(manifest_path, manifest)
    return manifest
