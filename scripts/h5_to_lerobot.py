from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import h5py
import numpy as np


LEROBOT_CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNKS_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_MB = 100
DEFAULT_VIDEO_FILE_SIZE_MB = 200


@dataclass(frozen=True)
class H5Layout:
    total_frames: int
    episode_lengths: np.ndarray
    episode_offsets: np.ndarray
    episode_success: np.ndarray
    action_dim: int
    state_dim: int
    image_shape: tuple[int, int, int]
    present_keys: tuple[str, ...]

    @property
    def total_episodes(self) -> int:
        return int(self.episode_lengths.size)


@dataclass(frozen=True)
class EpisodeShard:
    ordinal: int
    episode_start: int
    episode_end: int
    frame_start: int
    frame_end: int

    @property
    def chunk_index(self) -> int:
        return self.ordinal // DEFAULT_CHUNKS_SIZE

    @property
    def file_index(self) -> int:
        return self.ordinal % DEFAULT_CHUNKS_SIZE

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start


@dataclass(frozen=True)
class VideoJob:
    input_h5: str
    output_path: str
    pixels_key: str
    shard: EpisodeShard
    fps: int
    codec: str
    pixel_format: str
    encoder_options: dict[str, str]
    read_batch_size: int
    h5_cache_bytes: int


@dataclass(frozen=True)
class VideoResult:
    ordinal: int
    output_path: str
    frame_count: int
    byte_size: int
    info: dict[str, Any]


def _strict_integer_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"H5 field {name!r} must be one-dimensional, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.floating) or not np.all(np.isfinite(array)):
            raise ValueError(f"H5 field {name!r} must contain finite integers, got dtype={array.dtype}.")
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError(f"H5 field {name!r} contains non-integer values.")
        array = rounded
    return array.astype(np.int64, copy=False)


def inspect_h5_layout(path: str | Path) -> H5Layout:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input H5 file does not exist: {source}")

    required = {"action", "state", "pixels", "ep_len", "ep_offset", "ep_success"}
    with h5py.File(source, "r") as h5_file:
        present = tuple(sorted(h5_file.keys()))
        missing = sorted(required - set(present))
        if missing:
            raise KeyError(f"Input H5 is missing required fields: {missing}")

        actions = h5_file["action"]
        states = h5_file["state"]
        pixels = h5_file["pixels"]
        if actions.ndim != 2 or actions.shape[1] < 1:
            raise ValueError(f"'action' must have shape [N, A], got {actions.shape}.")
        if states.ndim != 2 or states.shape[1] < 1:
            raise ValueError(f"'state' must have shape [N, S], got {states.shape}.")
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError(f"'pixels' must have RGB shape [N, H, W, 3], got {pixels.shape}.")
        if pixels.dtype != np.uint8:
            raise ValueError(
                f"'pixels' must be uint8 for the fast video path, got dtype={pixels.dtype}."
            )

        total_frames = int(actions.shape[0])
        if total_frames < 1:
            raise ValueError("The input H5 contains no frames.")
        for key, dataset in (("state", states), ("pixels", pixels)):
            if int(dataset.shape[0]) != total_frames:
                raise ValueError(
                    f"H5 field {key!r} has {dataset.shape[0]} rows, expected {total_frames}."
                )

        lengths = _strict_integer_array(h5_file["ep_len"][:], name="ep_len")
        offsets = _strict_integer_array(h5_file["ep_offset"][:], name="ep_offset")
        success = _strict_integer_array(h5_file["ep_success"][:], name="ep_success")

        if lengths.size < 1:
            raise ValueError("The input H5 contains no episodes.")
        if offsets.size != lengths.size or success.size != lengths.size:
            raise ValueError(
                "ep_len, ep_offset, and ep_success must have identical lengths; "
                f"got {lengths.size}, {offsets.size}, and {success.size}."
            )
        if np.any(lengths <= 0):
            bad = np.flatnonzero(lengths <= 0)[:10].tolist()
            raise ValueError(f"Episode lengths must be positive; invalid episode ids: {bad}")
        if np.any((success != 0) & (success != 1)):
            bad = np.unique(success[(success != 0) & (success != 1)])[:10].tolist()
            raise ValueError(f"ep_success must contain only 0 or 1, got {bad}.")

        expected_offsets = np.empty_like(lengths)
        expected_offsets[0] = 0
        if lengths.size > 1:
            expected_offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
        if not np.array_equal(offsets, expected_offsets):
            mismatch = int(np.flatnonzero(offsets != expected_offsets)[0])
            raise ValueError(
                "Episodes must cover the flat arrays contiguously in episode order. "
                f"At episode {mismatch}, ep_offset={int(offsets[mismatch])}, "
                f"expected {int(expected_offsets[mismatch])}."
            )
        if int(lengths.sum(dtype=np.int64)) != total_frames:
            raise ValueError(
                f"sum(ep_len)={int(lengths.sum())} does not match the frame count {total_frames}."
            )

        if "step_idx" in h5_file and int(h5_file["step_idx"].shape[0]) != total_frames:
            raise ValueError(
                f"Optional 'step_idx' has {h5_file['step_idx'].shape[0]} rows, expected {total_frames}."
            )

        height, width, channels = map(int, pixels.shape[1:])
        return H5Layout(
            total_frames=total_frames,
            episode_lengths=lengths,
            episode_offsets=offsets,
            episode_success=success.astype(np.int8),
            action_dim=int(actions.shape[1]),
            state_dim=int(states.shape[1]),
            image_shape=(height, width, channels),
            present_keys=present,
        )


def compute_pi06_returns(
    episode_lengths: np.ndarray,
    episode_success: np.ndarray,
    *,
    failure_penalty: float | None,
    step_scale: float,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the pi0.6-style undiscounted return from episode success labels."""

    lengths = np.asarray(episode_lengths, dtype=np.int64).reshape(-1)
    success = np.asarray(episode_success, dtype=np.int8).reshape(-1)
    if lengths.size == 0 or lengths.size != success.size:
        raise ValueError("episode_lengths and episode_success must be non-empty and equally sized.")
    if np.any(lengths <= 0):
        raise ValueError("episode_lengths must be positive.")
    if np.any((success != 0) & (success != 1)):
        raise ValueError("episode_success must contain only 0 or 1.")
    if not np.isfinite(step_scale) or step_scale <= 0:
        raise ValueError("step_scale must be a finite positive number.")
    if failure_penalty is not None and (not np.isfinite(failure_penalty) or failure_penalty <= 0):
        raise ValueError("failure_penalty must be a finite positive number when supplied.")
    if normalization not in {"task_max", "global_minmax", "none"}:
        raise ValueError("normalization must be task_max, global_minmax, or none.")

    task_max_steps = float(int(lengths.max())) * float(step_scale)
    resolved_failure_penalty = (
        float(failure_penalty) if failure_penalty is not None else task_max_steps
    )
    total_frames = int(lengths.sum(dtype=np.int64))
    raw = np.empty(total_frames, dtype=np.float32)

    offset = 0
    for length, is_success in zip(lengths, success, strict=True):
        length_int = int(length)
        terminal_reward = 0.0 if bool(is_success) else -resolved_failure_penalty
        raw[offset : offset + length_int] = terminal_reward - float(step_scale) * np.arange(
            length_int - 1, -1, -1, dtype=np.float32
        )
        offset += length_int

    if normalization == "task_max":
        normalized = np.clip(raw / task_max_steps, -1.0, 0.0).astype(np.float32, copy=False)
    elif normalization == "global_minmax":
        minimum = float(raw.min())
        maximum = float(raw.max())
        if maximum - minimum < 1e-8:
            normalized = np.zeros_like(raw)
        else:
            normalized = (-1.0 + 2.0 * (raw - minimum) / (maximum - minimum)).astype(
                np.float32, copy=False
            )
    else:
        normalized = raw.copy()
    return raw, normalized, resolved_failure_penalty


def build_episode_shards(episode_lengths: np.ndarray, max_frames: int) -> list[EpisodeShard]:
    lengths = np.asarray(episode_lengths, dtype=np.int64).reshape(-1)
    if max_frames < 1:
        raise ValueError("max_frames must be positive.")
    if lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError("episode_lengths must be a non-empty array of positive values.")

    offsets = np.empty_like(lengths)
    offsets[0] = 0
    if lengths.size > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)

    shards: list[EpisodeShard] = []
    start_episode = 0
    accumulated = 0
    for episode, length in enumerate(lengths):
        length_int = int(length)
        if accumulated > 0 and accumulated + length_int > max_frames:
            frame_start = int(offsets[start_episode])
            frame_end = int(offsets[episode])
            shards.append(
                EpisodeShard(
                    ordinal=len(shards),
                    episode_start=start_episode,
                    episode_end=episode,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )
            )
            start_episode = episode
            accumulated = 0
        accumulated += length_int

    final_frame_start = int(offsets[start_episode])
    final_frame_end = int(lengths.sum(dtype=np.int64))
    shards.append(
        EpisodeShard(
            ordinal=len(shards),
            episode_start=start_episode,
            episode_end=int(lengths.size),
            frame_start=final_frame_start,
            frame_end=final_frame_end,
        )
    )
    return shards


def _parse_encoder_options(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        key = key.strip()
        if not separator or not key or not value.strip():
            raise ValueError(f"Invalid --encoder-option {item!r}; expected KEY=VALUE.")
        result[key] = value.strip()
    return result


def _encoder_options(args: argparse.Namespace, encoder_threads: int) -> dict[str, str]:
    codec = args.video_codec.lower()
    hardware_codec = any(
        token in codec for token in ("nvenc", "qsv", "vaapi", "videotoolbox")
    )
    options: dict[str, str] = {}
    if args.video_gop > 0:
        options["g"] = str(args.video_gop)

    if "nvenc" in codec:
        options.update({"rc": "constqp", "qp": str(args.video_crf), "preset": "p1"})
    elif "qsv" in codec:
        options["global_quality"] = str(args.video_crf)
    elif "vaapi" in codec:
        options["qp"] = str(args.video_crf)
    elif "videotoolbox" in codec:
        options["q:v"] = str(max(1, min(100, int(100 - args.video_crf * 2))))
    elif "svtav1" in codec:
        preset = args.video_preset if args.video_preset.isdigit() else "12"
        options.update({"crf": str(args.video_crf), "preset": preset})
    else:
        options.update(
            {
                "crf": str(args.video_crf),
                "preset": args.video_preset,
                "bf": "0",
                "sc_threshold": "0",
            }
        )

    if encoder_threads > 0 and not hardware_codec:
        options["threads"] = str(encoder_threads)
    options.update(_parse_encoder_options(args.encoder_option))
    return options


def _video_info(video_path: Path, av_module: Any) -> dict[str, Any]:
    with av_module.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        codec = getattr(getattr(stream, "codec", None), "canonical_name", None)
        if codec is None:
            codec = getattr(stream.codec_context, "name", "unknown")
        frame_rate = stream.base_rate or stream.average_rate
        return {
            "video.height": int(stream.height),
            "video.width": int(stream.width),
            "video.codec": str(codec),
            "video.pix_fmt": str(stream.pix_fmt),
            "video.is_depth_map": False,
            "video.fps": int(round(float(frame_rate))),
            "video.channels": 3,
            "has_audio": False,
        }


def _encode_video_job(job: VideoJob) -> VideoResult:
    try:
        import av
    except ImportError as exc:
        raise ImportError("Fast H5 conversion requires PyAV (installed with lerobot>=0.5.1).") from exc

    output = Path(job.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded_frames = 0
    time_base = Fraction(1, job.fps)

    with h5py.File(
        job.input_h5,
        "r",
        rdcc_nbytes=job.h5_cache_bytes,
        rdcc_nslots=1_000_003,
    ) as h5_file:
        pixels = h5_file[job.pixels_key]
        height, width, channels = map(int, pixels.shape[1:])
        if channels != 3:
            raise ValueError(f"Expected RGB pixels, got {pixels.shape}.")
        if job.pixel_format == "yuv420p" and (height % 2 or width % 2):
            raise ValueError(
                f"yuv420p requires even image dimensions, got height={height}, width={width}."
            )

        with av.open(str(output), "w", options={"movflags": "faststart"}) as container:
            stream = container.add_stream(job.codec, rate=job.fps, options=job.encoder_options)
            stream.width = width
            stream.height = height
            stream.pix_fmt = job.pixel_format
            stream.time_base = time_base

            batch_capacity = min(job.read_batch_size, job.shard.frame_count)
            buffer = np.empty((batch_capacity, height, width, channels), dtype=np.uint8)
            position = job.shard.frame_start
            while position < job.shard.frame_end:
                stop = min(position + batch_capacity, job.shard.frame_end)
                count = stop - position
                view = buffer[:count]
                pixels.read_direct(
                    view,
                    source_sel=np.s_[position:stop, :, :, :],
                    dest_sel=np.s_[:count, :, :, :],
                )
                for image in view:
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    frame.pts = encoded_frames
                    frame.time_base = time_base
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    encoded_frames += 1
                position = stop

            for packet in stream.encode():
                container.mux(packet)

    if encoded_frames != job.shard.frame_count:
        raise RuntimeError(
            f"Video shard {job.shard.ordinal} encoded {encoded_frames} frames, "
            f"expected {job.shard.frame_count}."
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Video encoder did not produce a valid file: {output}")
    return VideoResult(
        ordinal=job.shard.ordinal,
        output_path=str(output),
        frame_count=encoded_frames,
        byte_size=output.stat().st_size,
        info=_video_info(output, av),
    )


def _fixed_size_list_array(array: np.ndarray, pa_module: Any) -> Any:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    values = pa_module.array(contiguous.reshape(-1), type=pa_module.float32(), from_pandas=False)
    return pa_module.FixedSizeListArray.from_arrays(values, list_size=contiguous.shape[1])


def _write_data_shards(
    *,
    input_h5: Path,
    root: Path,
    layout: H5Layout,
    shards: list[EpisodeShard],
    raw_returns: np.ndarray,
    normalized_returns: np.ndarray,
    fps: int,
    action_key: str,
    state_key: str,
    compression: str | None,
    h5_cache_bytes: int,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Fast H5 conversion requires pyarrow (installed through the LeRobot datasets dependency)."
        ) from exc

    episode_index = np.repeat(
        np.arange(layout.total_episodes, dtype=np.int64), layout.episode_lengths
    )
    absolute_index = np.arange(layout.total_frames, dtype=np.int64)
    frame_index = absolute_index - layout.episode_offsets[episode_index]
    timestamp = (frame_index / float(fps)).astype(np.float32)
    task_index = np.zeros(layout.total_frames, dtype=np.int64)
    success_rows = np.repeat(layout.episode_success, layout.episode_lengths).astype(np.int8)

    with h5py.File(
        input_h5,
        "r",
        rdcc_nbytes=h5_cache_bytes,
        rdcc_nslots=1_000_003,
    ) as h5_file:
        for shard in shards:
            row_slice = slice(shard.frame_start, shard.frame_end)
            actions = np.asarray(h5_file["action"][row_slice], dtype=np.float32)
            states = np.asarray(h5_file["state"][row_slice], dtype=np.float32)
            if not np.all(np.isfinite(actions)):
                raise ValueError(f"Non-finite action values found in data shard {shard.ordinal}.")
            if not np.all(np.isfinite(states)):
                raise ValueError(f"Non-finite state values found in data shard {shard.ordinal}.")

            arrays = [
                _fixed_size_list_array(actions, pa),
                _fixed_size_list_array(states, pa),
                pa.array(normalized_returns[row_slice], type=pa.float32(), from_pandas=False),
                pa.array(raw_returns[row_slice], type=pa.float32(), from_pandas=False),
                pa.array(success_rows[row_slice], type=pa.int8(), from_pandas=False),
                pa.array(timestamp[row_slice], type=pa.float32(), from_pandas=False),
                pa.array(frame_index[row_slice], type=pa.int64(), from_pandas=False),
                pa.array(episode_index[row_slice], type=pa.int64(), from_pandas=False),
                pa.array(absolute_index[row_slice], type=pa.int64(), from_pandas=False),
                pa.array(task_index[row_slice], type=pa.int64(), from_pandas=False),
            ]
            names = [
                action_key,
                state_key,
                "return",
                "return_raw",
                "episode_success",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ]
            table = pa.Table.from_arrays(arrays, names=names)
            path = (
                root
                / "data"
                / f"chunk-{shard.chunk_index:03d}"
                / f"file-{shard.file_index:03d}.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table,
                path,
                compression=compression,
                use_dictionary=True,
                row_group_size=min(65_536, shard.frame_count),
            )


def _episode_file_indices(
    total_episodes: int, shards: list[EpisodeShard]
) -> tuple[np.ndarray, np.ndarray]:
    chunk = np.empty(total_episodes, dtype=np.int64)
    file = np.empty(total_episodes, dtype=np.int64)
    for shard in shards:
        chunk[shard.episode_start : shard.episode_end] = shard.chunk_index
        file[shard.episode_start : shard.episode_end] = shard.file_index
    return chunk, file


def _write_metadata(
    *,
    root: Path,
    layout: H5Layout,
    data_shards: list[EpisodeShard],
    video_shards: list[EpisodeShard],
    task: str,
    fps: int,
    robot_type: str | None,
    image_key: str,
    action_key: str,
    state_key: str,
    video_info: dict[str, Any],
    compression: str | None,
) -> None:
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Fast H5 conversion requires pandas and pyarrow (installed through LeRobot)."
        ) from exc

    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    tasks = pd.DataFrame(
        {"task_index": np.asarray([0], dtype=np.int64)},
        index=pd.Index([task], name="task"),
    )
    tasks.to_parquet(meta / "tasks.parquet", compression=compression)

    data_chunk, data_file = _episode_file_indices(layout.total_episodes, data_shards)
    video_chunk, video_file = _episode_file_indices(layout.total_episodes, video_shards)
    video_from = np.empty(layout.total_episodes, dtype=np.float64)
    video_to = np.empty(layout.total_episodes, dtype=np.float64)
    for shard in video_shards:
        for episode in range(shard.episode_start, shard.episode_end):
            local_start = int(layout.episode_offsets[episode]) - shard.frame_start
            video_from[episode] = local_start / float(fps)
            video_to[episode] = (
                local_start + int(layout.episode_lengths[episode])
            ) / float(fps)

    episode_indices = np.arange(layout.total_episodes, dtype=np.int64)
    episode_table = pa.Table.from_arrays(
        [
            pa.array(episode_indices, type=pa.int64()),
            pa.array([[task] for _ in range(layout.total_episodes)], type=pa.list_(pa.string())),
            pa.array(layout.episode_lengths, type=pa.int64()),
            pa.array(data_chunk, type=pa.int64()),
            pa.array(data_file, type=pa.int64()),
            pa.array(layout.episode_offsets, type=pa.int64()),
            pa.array(layout.episode_offsets + layout.episode_lengths, type=pa.int64()),
            pa.array(video_chunk, type=pa.int64()),
            pa.array(video_file, type=pa.int64()),
            pa.array(video_from, type=pa.float64()),
            pa.array(video_to, type=pa.float64()),
            pa.array(np.zeros(layout.total_episodes, dtype=np.int64), type=pa.int64()),
            pa.array(np.zeros(layout.total_episodes, dtype=np.int64), type=pa.int64()),
        ],
        names=[
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            f"videos/{image_key}/chunk_index",
            f"videos/{image_key}/file_index",
            f"videos/{image_key}/from_timestamp",
            f"videos/{image_key}/to_timestamp",
            "meta/episodes/chunk_index",
            "meta/episodes/file_index",
        ],
    )
    episodes_path = meta / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        episode_table,
        episodes_path,
        compression=compression,
        use_dictionary=True,
    )

    height, width, channels = layout.image_shape
    features = {
        image_key: {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channels"],
            "info": video_info,
        },
        state_key: {
            "dtype": "float32",
            "shape": [layout.state_dim],
            "names": None,
        },
        action_key: {
            "dtype": "float32",
            "shape": [layout.action_dim],
            "names": None,
        },
        "return": {"dtype": "float32", "shape": [1], "names": None},
        "return_raw": {"dtype": "float32", "shape": [1], "names": None},
        "episode_success": {"dtype": "int8", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": layout.total_episodes,
        "total_frames": layout.total_frames,
        "total_tasks": 1,
        "chunks_size": DEFAULT_CHUNKS_SIZE,
        "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_MB,
        "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_MB,
        "fps": fps,
        "splits": {"train": f"0:{layout.total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        "features": features,
    }
    (meta / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _verify_structure(
    *,
    root: Path,
    layout: H5Layout,
    video_results: list[VideoResult],
) -> None:
    try:
        import pandas as pd
        import pyarrow.dataset as pa_dataset
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Verification requires pandas and pyarrow.") from exc

    data_rows = pa_dataset.dataset(root / "data", format="parquet").count_rows()
    if data_rows != layout.total_frames:
        raise RuntimeError(f"Generated data has {data_rows} rows, expected {layout.total_frames}.")
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if pq.read_metadata(episodes_path).num_rows != layout.total_episodes:
        raise RuntimeError("Generated episode metadata row count is incorrect.")
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    if len(tasks) != 1 or "task_index" not in tasks.columns:
        raise RuntimeError("Generated task metadata is invalid.")
    if sum(result.frame_count for result in video_results) != layout.total_frames:
        raise RuntimeError("Generated videos do not cover every H5 frame exactly once.")
    if any(result.byte_size <= 0 for result in video_results):
        raise RuntimeError("At least one generated video file is empty.")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _verify_with_lerobot(
    *,
    root: Path,
    repo_id: str,
    layout: H5Layout,
    normalized_returns: np.ndarray,
    image_key: str,
    action_key: str,
    state_key: str,
) -> None:
    verification_cache = root / ".verification_cache"
    previous_cache = os.environ.get("HF_DATASETS_CACHE")
    os.environ["HF_DATASETS_CACHE"] = str(verification_cache)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        if previous_cache is None:
            os.environ.pop("HF_DATASETS_CACHE", None)
        else:
            os.environ["HF_DATASETS_CACHE"] = previous_cache
        raise ImportError(
            "LeRobot verification requires lerobot>=0.5.1,<0.6. "
            "Pass --skip-lerobot-verification only if verification will be run later in the WCM environment."
        ) from exc
    dataset = None
    try:
        dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
        if len(dataset) != layout.total_frames:
            raise RuntimeError(f"LeRobot reopened {len(dataset)} rows, expected {layout.total_frames}.")

        candidates = {0, layout.total_frames - 1}
        for target in (0, 1):
            matching = np.flatnonzero(layout.episode_success == target)
            if matching.size:
                episode = int(matching[0])
                candidates.add(int(layout.episode_offsets[episode]))
                candidates.add(
                    int(layout.episode_offsets[episode] + layout.episode_lengths[episode] - 1)
                )

        height, width, _ = layout.image_shape
        for row in sorted(candidates):
            sample = dataset[row]
            expected = {
                image_key,
                action_key,
                state_key,
                "return",
                "return_raw",
                "episode_success",
                "episode_index",
                "frame_index",
                "task_index",
            }
            missing = expected - set(sample)
            if missing:
                raise RuntimeError(f"LeRobot sample {row} is missing fields: {sorted(missing)}")
            image = _as_numpy(sample[image_key])
            if image.ndim != 3:
                raise RuntimeError(f"Decoded image at row {row} has invalid shape {image.shape}.")
            decoded_hw = image.shape[1:] if image.shape[0] in (1, 3, 4) else image.shape[:2]
            if tuple(decoded_hw) != (height, width):
                raise RuntimeError(
                    f"Decoded image at row {row} has spatial shape {decoded_hw}, expected {(height, width)}."
                )
            if _as_numpy(sample[action_key]).reshape(-1).size != layout.action_dim:
                raise RuntimeError(f"Action dimension mismatch at row {row}.")
            if _as_numpy(sample[state_key]).reshape(-1).size != layout.state_dim:
                raise RuntimeError(f"State dimension mismatch at row {row}.")
            actual_return = float(_as_numpy(sample["return"]).reshape(-1)[0])
            if not np.isclose(actual_return, float(normalized_returns[row]), rtol=0, atol=1e-6):
                raise RuntimeError(
                    f"Return mismatch at row {row}: got {actual_return}, expected {normalized_returns[row]}."
                )

    finally:
        dataset = None
        gc.collect()
        shutil.rmtree(verification_cache, ignore_errors=True)
        if previous_cache is None:
            os.environ.pop("HF_DATASETS_CACHE", None)
        else:
            os.environ["HF_DATASETS_CACHE"] = previous_cache


def _resolve_parallelism(args: argparse.Namespace, num_video_shards: int) -> tuple[int, int]:
    cpu_count = max(1, os.cpu_count() or 1)
    if args.video_workers < 0:
        raise ValueError("--video-workers cannot be negative.")
    if args.encoder_threads < 0:
        raise ValueError("--encoder-threads cannot be negative.")
    if args.video_workers == 0:
        hardware_codec = any(
            token in args.video_codec.lower()
            for token in ("nvenc", "qsv", "vaapi", "videotoolbox")
        )
        desired = 2 if hardware_codec else max(1, cpu_count // 4)
        workers = min(num_video_shards, 8, desired)
    else:
        workers = min(num_video_shards, args.video_workers)
    workers = max(1, workers)
    # 224x224 encoders stop scaling well at high thread counts. Keeping each
    # process at <=8 threads leaves CPU time for parallel HDF5 decompression.
    threads = args.encoder_threads or max(1, min(8, cpu_count // workers))
    return workers, threads


def _validate_feature_key(key: str, *, argument: str) -> str:
    key = key.strip()
    if not key:
        raise ValueError(f"{argument} cannot be empty.")
    if "/" in key:
        raise ValueError(f"{argument} cannot contain '/': {key!r}")
    return key


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Convert a flat episode-indexed H5 file directly into a WCM-compatible LeRobot v3 dataset. "
            "The source H5 value/value_raw fields are ignored; pi0.6-style returns are recomputed from ep_success."
        )
    )
    result.add_argument("--input-h5", required=True, help="Source H5 file; it is opened read-only.")
    result.add_argument(
        "--output-dir",
        required=True,
        help="New LeRobot v3 dataset directory. It must not already exist.",
    )
    result.add_argument(
        "--repo-id",
        required=True,
        help="LeRobot repo id stored in the WCM config, for example local/maniskill-pi06.",
    )
    result.add_argument(
        "--task",
        required=True,
        help="Natural-language instruction for this single-task H5 dataset.",
    )
    result.add_argument("--fps", type=int, default=10, help="Stored frame rate; default 10.")
    result.add_argument("--robot-type", default="maniskill")
    result.add_argument("--image-key", default="observation.images.front")
    result.add_argument("--state-key", default="observation.state")
    result.add_argument("--action-key", default="action")
    result.add_argument(
        "--failure-penalty",
        type=float,
        help="Positive C_fail. Default: maximum episode length times --step-scale.",
    )
    result.add_argument(
        "--step-scale",
        type=float,
        default=1.0,
        help="Control-step cost represented by one stored frame; default 1.",
    )
    result.add_argument(
        "--normalization",
        choices=["task_max", "global_minmax", "none"],
        default="task_max",
        help="WCM target scaling; canonical default is task_max.",
    )
    result.add_argument(
        "--max-video-frames",
        type=int,
        default=5000,
        help="Maximum frames per MP4, without splitting an episode; default 5000.",
    )
    result.add_argument(
        "--max-data-frames",
        type=int,
        default=250000,
        help="Maximum rows per data Parquet, without splitting an episode; default 250000.",
    )
    result.add_argument(
        "--video-workers",
        type=int,
        default=0,
        help="Parallel video encoder processes; 0 selects an automatic value up to 8.",
    )
    result.add_argument(
        "--encoder-threads",
        type=int,
        default=0,
        help="Threads per video encoder process; 0 divides available CPUs automatically.",
    )
    result.add_argument(
        "--video-codec",
        default="h264",
        help="PyAV/FFmpeg encoder name. h264 is the fast portable default; h264_nvenc can use NVIDIA GPUs.",
    )
    result.add_argument("--pixel-format", default="yuv420p")
    result.add_argument("--video-crf", type=int, default=23)
    result.add_argument(
        "--video-preset",
        default="ultrafast",
        help="Software encoder preset; ultrafast prioritizes conversion throughput.",
    )
    result.add_argument(
        "--video-gop",
        type=int,
        default=2,
        help="Keyframe interval. LeRobot's random-access-friendly default is 2.",
    )
    result.add_argument(
        "--encoder-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra FFmpeg encoder option; repeat as needed and override defaults.",
    )
    result.add_argument(
        "--read-batch-size",
        type=int,
        default=256,
        help="H5 frames read into memory per encoder batch; default 256.",
    )
    result.add_argument(
        "--h5-cache-mb",
        type=int,
        default=64,
        help="HDF5 raw chunk cache per process in MiB; default 64.",
    )
    result.add_argument(
        "--parquet-compression",
        choices=["snappy", "zstd", "none"],
        default="snappy",
    )
    result.add_argument(
        "--skip-lerobot-verification",
        action="store_true",
        help="Skip reopening and decoding sample rows with LeRobot. Structural checks still run.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    started = time.perf_counter()

    if args.fps < 1:
        raise ValueError("--fps must be positive.")
    if args.max_video_frames < 1 or args.max_data_frames < 1:
        raise ValueError("--max-video-frames and --max-data-frames must be positive.")
    if args.read_batch_size < 1 or args.h5_cache_mb < 1:
        raise ValueError("--read-batch-size and --h5-cache-mb must be positive.")
    if args.video_crf < 0 or args.video_gop < 0:
        raise ValueError("--video-crf and --video-gop cannot be negative.")

    input_h5 = Path(args.input_h5).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Output directory already exists: {output}. Use a new path; the converter never overwrites data."
        )
    if not args.repo_id.strip():
        raise ValueError("--repo-id cannot be empty.")
    task = args.task.strip()
    if not task:
        raise ValueError("--task cannot be empty.")
    image_key = _validate_feature_key(args.image_key, argument="--image-key")
    state_key = _validate_feature_key(args.state_key, argument="--state-key")
    action_key = _validate_feature_key(args.action_key, argument="--action-key")
    if len({image_key, state_key, action_key}) != 3:
        raise ValueError("--image-key, --state-key, and --action-key must be distinct.")

    compression = None if args.parquet_compression == "none" else args.parquet_compression
    h5_cache_bytes = args.h5_cache_mb * 1024 * 1024
    layout = inspect_h5_layout(input_h5)
    raw_returns, normalized_returns, resolved_failure_penalty = compute_pi06_returns(
        layout.episode_lengths,
        layout.episode_success,
        failure_penalty=args.failure_penalty,
        step_scale=args.step_scale,
        normalization=args.normalization,
    )
    video_shards = build_episode_shards(layout.episode_lengths, args.max_video_frames)
    data_shards = build_episode_shards(layout.episode_lengths, args.max_data_frames)
    workers, encoder_threads = _resolve_parallelism(args, len(video_shards))
    encoder_options = _encoder_options(args, encoder_threads)

    ignored = [key for key in ("value", "value_raw") if key in layout.present_keys]
    print(
        f"[h5] episodes={layout.total_episodes}, frames={layout.total_frames}, "
        f"success={int(layout.episode_success.sum())}, "
        f"failure={int((layout.episode_success == 0).sum())}",
        flush=True,
    )
    print(
        f"[returns] ignored H5 fields={ignored or 'none'}, normalization={args.normalization}, "
        f"failure_penalty={resolved_failure_penalty:g}",
        flush=True,
    )
    print(
        f"[video] shards={len(video_shards)}, workers={workers}, "
        f"encoder_threads={encoder_threads}, codec={args.video_codec}, options={encoder_options}",
        flush=True,
    )

    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True)
    executor: concurrent.futures.ProcessPoolExecutor | None = None
    try:
        video_jobs = []
        for shard in video_shards:
            video_path = (
                staging
                / "videos"
                / image_key
                / f"chunk-{shard.chunk_index:03d}"
                / f"file-{shard.file_index:03d}.mp4"
            )
            video_jobs.append(
                VideoJob(
                    input_h5=str(input_h5),
                    output_path=str(video_path),
                    pixels_key="pixels",
                    shard=shard,
                    fps=args.fps,
                    codec=args.video_codec,
                    pixel_format=args.pixel_format,
                    encoder_options=encoder_options,
                    read_batch_size=args.read_batch_size,
                    h5_cache_bytes=h5_cache_bytes,
                )
            )

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        futures = {executor.submit(_encode_video_job, job): job for job in video_jobs}

        _write_data_shards(
            input_h5=input_h5,
            root=staging,
            layout=layout,
            shards=data_shards,
            raw_returns=raw_returns,
            normalized_returns=normalized_returns,
            fps=args.fps,
            action_key=action_key,
            state_key=state_key,
            compression=compression,
            h5_cache_bytes=h5_cache_bytes,
        )
        print(f"[data] wrote {len(data_shards)} Parquet shard(s)", flush=True)

        video_results: list[VideoResult] = []
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            result = future.result()
            video_results.append(result)
            print(
                f"[video] finished {job.shard.ordinal + 1}/{len(video_jobs)}: "
                f"frames={result.frame_count}, size={result.byte_size / (1024**2):.1f} MiB",
                flush=True,
            )
        video_results.sort(key=lambda item: item.ordinal)
        executor.shutdown(wait=True, cancel_futures=False)
        executor = None

        first_video_info = video_results[0].info
        if any(result.info != first_video_info for result in video_results[1:]):
            raise RuntimeError("Generated video shards do not share identical codec/shape/fps metadata.")
        _write_metadata(
            root=staging,
            layout=layout,
            data_shards=data_shards,
            video_shards=video_shards,
            task=task,
            fps=args.fps,
            robot_type=args.robot_type or None,
            image_key=image_key,
            action_key=action_key,
            state_key=state_key,
            video_info=first_video_info,
            compression=compression,
        )
        _verify_structure(root=staging, layout=layout, video_results=video_results)
        if not args.skip_lerobot_verification:
            print("[verify] reopening sample rows with LeRobot and PyAV...", flush=True)
            _verify_with_lerobot(
                root=staging,
                repo_id=args.repo_id.strip(),
                layout=layout,
                normalized_returns=normalized_returns,
                image_key=image_key,
                action_key=action_key,
                state_key=state_key,
            )

        elapsed = time.perf_counter() - started
        manifest = {
            "schema_version": 1,
            "source_h5": str(input_h5),
            "source_keys": list(layout.present_keys),
            "ignored_source_value_fields": ignored,
            "output_repo_id": args.repo_id.strip(),
            "task": task,
            "counts": {
                "episodes": layout.total_episodes,
                "frames": layout.total_frames,
                "successful_episodes": int(layout.episode_success.sum()),
                "failed_episodes": int((layout.episode_success == 0).sum()),
            },
            "return_definition": {
                "source": "ep_success",
                "nonterminal_reward": -float(args.step_scale),
                "success_terminal_reward": 0.0,
                "failure_terminal_reward": -resolved_failure_penalty,
                "failure_penalty": resolved_failure_penalty,
                "discount": 1.0,
                "normalization": args.normalization,
                "task_max_episode_length": int(layout.episode_lengths.max()),
                "normalized_range": [
                    float(normalized_returns.min()),
                    float(normalized_returns.max()),
                ],
                "raw_range": [float(raw_returns.min()), float(raw_returns.max())],
            },
            "writer": {
                "format": LEROBOT_CODEBASE_VERSION,
                "strategy": "direct_parquet_parallel_episode_aligned_video",
                "fps": args.fps,
                "video_codec": args.video_codec,
                "pixel_format": args.pixel_format,
                "encoder_options": encoder_options,
                "video_workers": workers,
                "encoder_threads_per_worker": encoder_threads,
                "video_shards": len(video_shards),
                "data_shards": len(data_shards),
                "lerobot_verified": not args.skip_lerobot_verification,
                "elapsed_seconds": elapsed,
            },
        }
        (staging / "meta" / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output)
        print(
            f"[done] wrote {output} in {elapsed:.1f}s "
            f"({layout.total_frames / max(elapsed, 1e-9):.1f} frames/s)",
            flush=True,
        )
    except BaseException:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
