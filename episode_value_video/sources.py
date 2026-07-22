from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from PIL import Image

from .curves import EpisodeCurve
from .images import image_to_pil
from .video_io import iter_video_frames, probe_video


class SourceResolutionError(RuntimeError):
    """Raised when an episode cannot be mapped to its original frames."""


def _scalar(value: Any, *, field: str) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise SourceResolutionError(f"{field} must be scalar, got shape {array.shape}.")
    result = array.reshape(-1)[0]
    return result.item() if hasattr(result, "item") else result


def _positive_fps(value: Any) -> float | None:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if math.isfinite(fps) and fps > 0 else None


class EpisodeFrameSource(ABC):
    episode_id: int
    fps: float
    frame_count: int
    first_frame_index: int
    last_frame_index: int
    camera_key: str | None
    description: str
    alignment_basis: str
    frame_count_inferred: bool = False
    expected_first_curve_frame: int | None = None

    @abstractmethod
    def frames(self) -> Iterator[tuple[int, Image.Image]]:
        """Yield ``(actual_frame_index, original_RGB_frame)`` pairs."""


class EpisodeSourceRepository(ABC):
    @abstractmethod
    def open_episode(self, curve: EpisodeCurve) -> EpisodeFrameSource:
        pass


@dataclass(slots=True)
class DatasetEpisodeFrameSource(EpisodeFrameSource):
    dataset: Any
    row_start: int
    row_end: int
    episode_id: int
    fps: float
    first_frame_index: int
    last_frame_index: int
    camera_key: str
    description: str
    alignment_basis: str = "LeRobot row episode_index + frame_index"
    frame_count_inferred: bool = False
    expected_first_curve_frame: int | None = None

    @property
    def frame_count(self) -> int:
        return self.row_end - self.row_start

    def frames(self) -> Iterator[tuple[int, Image.Image]]:
        expected_frame = self.first_frame_index
        for row in range(self.row_start, self.row_end):
            sample = self.dataset[row]
            episode_id = int(_scalar(sample["episode_index"], field="episode_index"))
            if episode_id != self.episode_id:
                raise SourceResolutionError(
                    f"Dataset row {row} changed episode: expected {self.episode_id}, got {episode_id}."
                )
            frame_index = int(_scalar(sample["frame_index"], field="frame_index"))
            if frame_index != expected_frame:
                raise SourceResolutionError(
                    f"episode_id={self.episode_id} has non-consecutive source frames: "
                    f"expected {expected_frame}, got {frame_index} at row {row}."
                )
            if self.camera_key not in sample:
                raise SourceResolutionError(
                    f"Camera key {self.camera_key!r} is absent from dataset row {row}."
                )
            yield frame_index, image_to_pil(sample[self.camera_key])
            expected_frame += 1


class LeRobotDatasetRepository(EpisodeSourceRepository):
    """Read exact original frames through the same LeRobot dataset as evaluation."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        dataset_root: str | Path | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
        camera_key: str | None = None,
        fps: float | None = None,
    ) -> None:
        try:
            from world_critic.checkpoint import inspect_checkpoint_config
            from world_critic.config import apply_runtime_overrides
            from world_critic.data import episode_ids_from_dataset, load_lerobot_dataset
            from world_critic.training import config_from_checkpoint_payload
        except ImportError as exc:
            raise ImportError(
                "Checkpoint-backed frame loading must be run from WCM_v2 with the world_critic "
                "environment installed."
            ) from exc

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        config_dict = inspect_checkpoint_config(checkpoint_path)
        train_config = apply_runtime_overrides(
            config_from_checkpoint_payload({"config": config_dict})
        )
        if dataset_root is not None:
            train_config.data.root = str(Path(dataset_root).expanduser().resolve())
        if repo_id is not None:
            train_config.data.repo_id = repo_id
        if revision is not None:
            train_config.data.revision = revision
        self.data_config = train_config.data
        self.history_size = int(self.data_config.history_size)
        self.dataset = load_lerobot_dataset(self.data_config)

        configured_cameras = list(self.data_config.image_keys)
        self.camera_key = camera_key or (configured_cameras[0] if configured_cameras else None)
        if self.camera_key is None:
            raise SourceResolutionError("Checkpoint config contains no data.image_keys camera.")
        features = set(getattr(self.dataset, "features", {}))
        if self.camera_key not in features:
            raise SourceResolutionError(
                f"Camera key {self.camera_key!r} is absent from the LeRobot dataset. "
                f"Available features include: {sorted(features)[:30]}"
            )

        episode_by_row = np.asarray(episode_ids_from_dataset(self.dataset), dtype=np.int64).reshape(-1)
        if episode_by_row.size == 0:
            raise SourceResolutionError("LeRobot dataset contains no rows.")
        boundaries = np.flatnonzero(episode_by_row[1:] != episode_by_row[:-1]) + 1
        starts = np.concatenate(([0], boundaries)).astype(np.int64, copy=False)
        ends = np.concatenate((boundaries, [episode_by_row.size])).astype(np.int64, copy=False)
        self.ranges: dict[int, tuple[int, int]] = {}
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            episode_id = int(episode_by_row[start])
            if episode_id in self.ranges:
                raise SourceResolutionError(
                    f"episode_index={episode_id} is not contiguous in the LeRobot dataset."
                )
            self.ranges[episode_id] = (int(start), int(end))

        self.fps = _positive_fps(fps) or self._discover_fps(dataset_root)
        if self.fps is None:
            raise SourceResolutionError(
                "Could not determine dataset FPS. Pass --fps explicitly or ensure meta/info.json "
                "contains a positive 'fps'."
            )
        self.description = (
            f"LeRobotDataset(repo_id={self.data_config.repo_id!r}, "
            f"root={self.data_config.root!r}, camera={self.camera_key!r})"
        )

    def _discover_fps(self, explicit_root: str | Path | None) -> float | None:
        candidates = [
            getattr(self.dataset, "fps", None),
            getattr(getattr(self.dataset, "meta", None), "fps", None),
        ]
        meta = getattr(self.dataset, "meta", None)
        if isinstance(meta, Mapping):
            candidates.append(meta.get("fps"))
        info = getattr(meta, "info", None)
        if isinstance(info, Mapping):
            candidates.append(info.get("fps"))
        for candidate in candidates:
            resolved = _positive_fps(candidate)
            if resolved is not None:
                return resolved

        roots = [explicit_root, self.data_config.root]
        for root in roots:
            if root is None:
                continue
            info_path = Path(root).expanduser() / "meta" / "info.json"
            if not info_path.is_file():
                continue
            try:
                payload = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            resolved = _positive_fps(payload.get("fps"))
            if resolved is not None:
                return resolved
        return None

    def open_episode(self, curve: EpisodeCurve) -> EpisodeFrameSource:
        if curve.episode_id not in self.ranges:
            raise SourceResolutionError(
                f"episode_id={curve.episode_id} is absent from the checkpoint dataset."
            )
        row_start, row_end = self.ranges[curve.episode_id]
        first_sample = self.dataset[row_start]
        last_sample = self.dataset[row_end - 1]
        first_frame = int(_scalar(first_sample["frame_index"], field="frame_index"))
        last_frame = int(_scalar(last_sample["frame_index"], field="frame_index"))
        frame_count = row_end - row_start
        if last_frame - first_frame + 1 != frame_count:
            raise SourceResolutionError(
                f"episode_id={curve.episode_id} frame span [{first_frame}, {last_frame}] "
                f"does not match {frame_count} dataset rows."
            )
        return DatasetEpisodeFrameSource(
            dataset=self.dataset,
            row_start=row_start,
            row_end=row_end,
            episode_id=curve.episode_id,
            fps=self.fps,
            first_frame_index=first_frame,
            last_frame_index=last_frame,
            camera_key=self.camera_key,
            description=self.description,
            expected_first_curve_frame=first_frame + self.history_size - 1,
        )


@dataclass(slots=True)
class EncodedVideoEpisodeFrameSource(EpisodeFrameSource):
    path: Path
    episode_id: int
    fps: float
    frame_count: int
    first_frame_index: int
    last_frame_index: int
    camera_key: str | None
    description: str
    alignment_basis: str
    backend: str = "auto"
    ffmpeg: str | Path | None = None
    start_seconds: float = 0.0
    decode_frame_limit: int | None = None
    frame_count_inferred: bool = False
    expected_first_curve_frame: int | None = None

    def frames(self) -> Iterator[tuple[int, Image.Image]]:
        _, decoded = iter_video_frames(
            self.path,
            backend=self.backend,
            ffmpeg=self.ffmpeg,
            start_seconds=self.start_seconds,
            max_frames=self.decode_frame_limit,
        )
        for ordinal, frame in enumerate(decoded):
            yield self.first_frame_index + ordinal, frame


class VideoTemplateRepository(EpisodeSourceRepository):
    def __init__(
        self,
        template: str,
        *,
        frame_offset: int = 0,
        fps: float | None = None,
        backend: str = "auto",
        ffmpeg: str | Path | None = None,
        history_size: int | None = None,
    ) -> None:
        self.template = template
        self.frame_offset = int(frame_offset)
        self.fps_override = _positive_fps(fps)
        self.backend = backend
        self.ffmpeg = ffmpeg
        if history_size is not None and history_size < 1:
            raise ValueError("history_size must be positive when provided.")
        self.history_size = history_size

    def open_episode(self, curve: EpisodeCurve) -> EpisodeFrameSource:
        try:
            formatted = self.template.format(episode_id=curve.episode_id)
        except (KeyError, ValueError) as exc:
            raise SourceResolutionError(
                "Invalid --video-template. Use a Python field such as "
                "'/videos/episode-{episode_id:06d}.mp4'."
            ) from exc
        path = Path(formatted).expanduser().resolve()
        probe = probe_video(path, backend=self.backend, ffmpeg=self.ffmpeg)
        fps = self.fps_override or probe.fps
        if fps is None:
            raise SourceResolutionError(
                f"Could not determine FPS for {path}; pass --fps explicitly."
            )
        if probe.frame_count is not None:
            frame_count = probe.frame_count
            inferred = False
        else:
            # The WCM endpoint contract ends at the penultimate episode frame.
            # This is validated against the decoded count before output commit.
            frame_count = curve.last_frame - self.frame_offset + 2
            inferred = True
        if frame_count < 1:
            raise SourceResolutionError(
                f"Inferred a non-positive frame count for episode_id={curve.episode_id}. "
                "Check --frame-offset."
            )
        return EncodedVideoEpisodeFrameSource(
            path=path,
            episode_id=curve.episode_id,
            fps=fps,
            frame_count=frame_count,
            first_frame_index=self.frame_offset,
            last_frame_index=self.frame_offset + frame_count - 1,
            camera_key=None,
            description=str(path),
            alignment_basis="decoded video ordinal + explicit frame offset",
            backend=self.backend,
            ffmpeg=self.ffmpeg,
            frame_count_inferred=inferred,
            expected_first_curve_frame=(
                self.frame_offset + self.history_size - 1
                if self.history_size is not None
                else None
            ),
        )


class VideoMapRepository(EpisodeSourceRepository):
    def __init__(
        self,
        path: str | Path,
        *,
        default_frame_offset: int = 0,
        fps: float | None = None,
        backend: str = "auto",
        ffmpeg: str | Path | None = None,
        history_size: int | None = None,
    ) -> None:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SourceResolutionError("--video-map must contain a JSON object keyed by episode id.")
        self.base = source.parent
        self.entries = payload
        self.default_frame_offset = int(default_frame_offset)
        self.fps_override = _positive_fps(fps)
        self.backend = backend
        self.ffmpeg = ffmpeg
        if history_size is not None and history_size < 1:
            raise ValueError("history_size must be positive when provided.")
        self.history_size = history_size

    def open_episode(self, curve: EpisodeCurve) -> EpisodeFrameSource:
        raw = self.entries.get(str(curve.episode_id), self.entries.get(curve.episode_id))
        if raw is None:
            raise SourceResolutionError(
                f"Video map has no entry for episode_id={curve.episode_id}."
            )
        if isinstance(raw, str):
            entry = {"path": raw}
        elif isinstance(raw, dict):
            entry = raw
        else:
            raise SourceResolutionError(
                f"Video map entry for episode_id={curve.episode_id} must be a path or object."
            )
        raw_path = Path(str(entry.get("path", ""))).expanduser()
        if not raw_path.is_absolute():
            raw_path = self.base / raw_path
        path = raw_path.resolve()
        probe = probe_video(path, backend=self.backend, ffmpeg=self.ffmpeg)
        frame_offset = int(entry.get("frame_offset", self.default_frame_offset))
        fps = self.fps_override or _positive_fps(entry.get("fps")) or probe.fps
        if fps is None:
            raise SourceResolutionError(f"Could not determine FPS for {path}.")
        explicit_count = entry.get("frame_count")
        entry_history_size = entry.get("history_size", self.history_size)
        if entry_history_size is not None:
            entry_history_size = int(entry_history_size)
            if entry_history_size < 1:
                raise SourceResolutionError("Video map history_size must be positive.")
        start_seconds = float(entry.get("start_seconds", 0.0))
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise SourceResolutionError("Video map start_seconds must be finite and non-negative.")
        if explicit_count is not None:
            frame_count = int(explicit_count)
            inferred = False
        elif start_seconds > 0:
            frame_count = curve.last_frame - frame_offset + 2
            inferred = True
        elif probe.frame_count is not None:
            frame_count = probe.frame_count
            inferred = False
        else:
            frame_count = curve.last_frame - frame_offset + 2
            inferred = True
        return EncodedVideoEpisodeFrameSource(
            path=path,
            episode_id=curve.episode_id,
            fps=fps,
            frame_count=frame_count,
            first_frame_index=frame_offset,
            last_frame_index=frame_offset + frame_count - 1,
            camera_key=entry.get("camera_key"),
            description=str(path),
            alignment_basis="video map ordinal + frame_offset",
            backend=self.backend,
            ffmpeg=self.ffmpeg,
            start_seconds=start_seconds,
            decode_frame_limit=(
                frame_count
                if explicit_count is not None or start_seconds > 0
                else None
            ),
            frame_count_inferred=inferred,
            expected_first_curve_frame=(
                frame_offset + entry_history_size - 1
                if entry_history_size is not None
                else None
            ),
        )


class LeRobotShardRepository(EpisodeSourceRepository):
    """Resolve LeRobot v3 video shards through meta/episodes timestamps."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        camera_key: str | None = None,
        frame_offset: int = 0,
        fps: float | None = None,
        backend: str = "auto",
        ffmpeg: str | Path | None = None,
        history_size: int | None = None,
    ) -> None:
        self.root = Path(dataset_root).expanduser().resolve()
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot info file does not exist: {info_path}")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        features = self.info.get("features", {})
        video_keys = [
            key
            for key, feature in features.items()
            if isinstance(feature, dict) and str(feature.get("dtype", "")).lower() == "video"
        ]
        if not video_keys:
            raise SourceResolutionError(
                "The LeRobot dataset has no video features. Use --checkpoint mode so image "
                "features can be decoded row by row."
            )
        self.camera_key = camera_key or video_keys[0]
        if self.camera_key not in video_keys:
            raise SourceResolutionError(
                f"Camera {self.camera_key!r} is not a video feature. Available: {video_keys}"
            )
        self.fps = _positive_fps(fps) or _positive_fps(self.info.get("fps"))
        if self.fps is None:
            raise SourceResolutionError("LeRobot meta/info.json has no valid FPS; pass --fps.")
        self.video_pattern = self.info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
        self.frame_offset = int(frame_offset)
        self.backend = backend
        self.ffmpeg = ffmpeg
        if history_size is not None and history_size < 1:
            raise ValueError("history_size must be positive when provided.")
        self.history_size = history_size
        self.rows = self._load_episode_rows()

    def _load_episode_rows(self) -> dict[int, dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "LeRobot shard mode requires pyarrow (normally installed with datasets/LeRobot). "
                "Use --checkpoint mode if the dataset is already loadable there."
            ) from exc
        episode_dir = self.root / "meta" / "episodes"
        files = sorted(episode_dir.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata parquet files found under {episode_dir}")
        rows: dict[int, dict[str, Any]] = {}
        for path in files:
            table = pq.read_table(path)
            names = table.schema.names
            if "episode_index" not in names:
                raise SourceResolutionError(f"{path} has no episode_index column.")
            columns = {name: table.column(name).to_pylist() for name in names}
            for row_index, raw_episode in enumerate(columns["episode_index"]):
                episode_id = int(raw_episode)
                if episode_id in rows:
                    raise SourceResolutionError(
                        f"Duplicate episode_index={episode_id} in meta/episodes."
                    )
                rows[episode_id] = {name: values[row_index] for name, values in columns.items()}
        return rows

    def open_episode(self, curve: EpisodeCurve) -> EpisodeFrameSource:
        row = self.rows.get(curve.episode_id)
        if row is None:
            raise SourceResolutionError(
                f"episode_id={curve.episode_id} is absent from LeRobot meta/episodes."
            )
        prefix = f"videos/{self.camera_key}"
        required = [
            "length",
            f"{prefix}/chunk_index",
            f"{prefix}/file_index",
            f"{prefix}/from_timestamp",
        ]
        missing = [name for name in required if name not in row]
        if missing:
            raise SourceResolutionError(
                f"Episode metadata is missing camera fields for {self.camera_key!r}: {missing}"
            )
        frame_count = int(row["length"])
        chunk_index = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        start_seconds = float(row[f"{prefix}/from_timestamp"])
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise SourceResolutionError(
                f"episode_id={curve.episode_id} has invalid from_timestamp={start_seconds!r}."
            )
        relative = self.video_pattern.format(
            video_key=self.camera_key,
            chunk_index=chunk_index,
            file_index=file_index,
            episode_index=curve.episode_id,
        )
        path = (self.root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resolved LeRobot video shard does not exist: {path}")

        to_key = f"{prefix}/to_timestamp"
        if to_key in row:
            to_seconds = float(row[to_key])
            if not math.isfinite(to_seconds) or to_seconds < start_seconds:
                raise SourceResolutionError(
                    f"episode_id={curve.episode_id} has invalid to_timestamp={to_seconds!r}."
                )
            duration = to_seconds - start_seconds
            expected = frame_count / self.fps
            tolerance = max(1.5 / self.fps, 1e-3)
            if abs(duration - expected) > tolerance:
                raise SourceResolutionError(
                    f"episode_id={curve.episode_id} metadata duration {duration:.6f}s "
                    f"does not match length/fps {expected:.6f}s."
                )
        first_frame = self.frame_offset
        return EncodedVideoEpisodeFrameSource(
            path=path,
            episode_id=curve.episode_id,
            fps=self.fps,
            frame_count=frame_count,
            first_frame_index=first_frame,
            last_frame_index=first_frame + frame_count - 1,
            camera_key=self.camera_key,
            description=f"{path} [{start_seconds:.6f}s, {frame_count} frames]",
            alignment_basis="LeRobot meta/episodes video timestamp + episode-local ordinal",
            backend=self.backend,
            ffmpeg=self.ffmpeg,
            start_seconds=start_seconds,
            decode_frame_limit=frame_count,
            frame_count_inferred=False,
            expected_first_curve_frame=(
                first_frame + self.history_size - 1
                if self.history_size is not None
                else None
            ),
        )
