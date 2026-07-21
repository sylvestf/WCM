from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig, ModelConfig


@dataclass(frozen=True)
class EpisodeSplit:
    train: list[int]
    val: list[int]


@dataclass
class CombinedProcessor:
    """Pickle-safe image/token preprocessing bundle for DataLoader workers.

    This must live at module scope.  A locally-defined class works with Linux
    ``fork`` workers but cannot be pickled by Windows/macOS ``spawn`` workers,
    which would make the checked-in launcher fail as soon as
    ``num_workers > 0``.
    """

    image_processor: Any
    tokenizer: Any


def build_episode_split(
    episode_ids: Iterable[int],
    val_fraction: float,
    seed: int,
) -> EpisodeSplit:
    ids = sorted({int(value) for value in episode_ids})
    if not ids:
        raise ValueError("No episodes are available for splitting.")
    generator = random.Random(seed)
    generator.shuffle(ids)
    if len(ids) == 1 or val_fraction <= 0:
        return EpisodeSplit(train=sorted(ids), val=[])
    val_count = max(1, min(len(ids) - 1, round(len(ids) * val_fraction)))
    return EpisodeSplit(train=sorted(ids[val_count:]), val=sorted(ids[:val_count]))


def save_episode_split(split: EpisodeSplit, path: str | Path, seed: int, val_fraction: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "val_fraction": val_fraction,
        "train": split.train,
        "val": split.val,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_episode_split(path: str | Path) -> EpisodeSplit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported split manifest: {path}")
    split = EpisodeSplit(
        train=[int(value) for value in payload["train"]],
        val=[int(value) for value in payload["val"]],
    )
    overlap = set(split.train) & set(split.val)
    if overlap:
        raise ValueError(f"Train/val episode leakage in {path}: {sorted(overlap)[:10]}")
    return split


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_image_tensor(value: Any) -> torch.Tensor:
    """Canonicalize LeRobot image cells from tensor/array/PIL-like values."""
    if torch.is_tensor(value):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, RuntimeError):
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"Image feature must be rank-3, got shape {array.shape}.")
        return torch.from_numpy(array)


def _as_scalar(value: Any, *, name: str) -> Any:
    """Read one scalar without silently truncating vector-valued HF/Torch cells."""
    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}.")
    scalar = array.reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def _column_values(dataset: Any, key: str) -> list[Any]:
    """Return a materialized column across LeRobot/Hugging Face versions."""
    table = getattr(dataset, "hf_dataset", None)
    column_names = set(getattr(table, "column_names", ()))
    if table is not None and key in column_names:
        return list(table[key])
    return [dataset[index][key] for index in range(len(dataset))]


def _scalar_column(dataset: Any, key: str, dtype: Any) -> np.ndarray:
    # LeRobot wraps a Hugging Face Dataset.  For scalar metadata columns such
    # as episode_index/frame_index, converting the underlying Arrow column
    # directly avoids materializing millions of Python scalar objects first.
    table = getattr(dataset, "hf_dataset", None)
    column_names = set(getattr(table, "column_names", ()))
    if table is not None and key in column_names:
        arrow_table = getattr(table, "data", None)
        if arrow_table is not None and hasattr(arrow_table, "column"):
            try:
                values = arrow_table.column(key).to_numpy(zero_copy_only=False)
                return np.asarray(values, dtype=dtype)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                # Fall back to the version-agnostic scalar path below for
                # Arrow extension columns or older datasets releases.
                pass
    return np.asarray(
        [_as_scalar(value, name=key) for value in _column_values(dataset, key)],
        dtype=dtype,
    )


def _feature_shape(feature: Any) -> tuple[int, ...]:
    if isinstance(feature, Mapping):
        shape = feature.get("shape", ())
    else:
        shape = getattr(feature, "shape", ())
    return tuple(shape or ())


def _clean_task_text(value: Any, *, source: str) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{source} must contain one task string, got {len(value)} values.")
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{source} resolved to an empty language instruction.")
    return text


def infer_feature_dim(dataset: Any, key: str) -> int:
    feature = dataset.features.get(key)
    if feature is not None:
        shape = _feature_shape(feature)
        if shape:
            return int(math.prod(shape))
    sample = dataset[0][key]
    return int(_to_numpy(sample).size)


def fit_action_normalization(dataset: Any, config: DataConfig, train_episode_ids: Iterable[int]) -> None:
    if not config.normalize_action:
        config.action_mean = None
        config.action_std = None
        return
    if config.action_mean is not None and config.action_std is not None:
        return
    episode_ids = episode_ids_from_dataset(dataset)
    train_mask = np.isin(episode_ids, np.asarray(list(train_episode_ids), dtype=np.int64))
    if not train_mask.any():
        raise ValueError("Cannot fit action normalization: training split has no rows.")
    actions = np.stack(
        [_to_numpy(value).reshape(-1) for value in _column_values(dataset, config.action_key)]
    ).astype(np.float64)
    actions = actions.reshape(len(actions), -1)[train_mask]
    finite_rows = np.isfinite(actions).all(axis=1)
    actions = actions[finite_rows]
    if len(actions) == 0:
        raise ValueError("No finite training actions are available for normalization.")
    mean = actions.mean(axis=0)
    std = actions.std(axis=0)
    std = np.maximum(std, config.normalization_epsilon)
    config.action_mean = mean.astype(np.float32).tolist()
    config.action_std = std.astype(np.float32).tolist()


def episode_ids_from_dataset(dataset: Any) -> np.ndarray:
    return _scalar_column(dataset, "episode_index", np.int64).reshape(-1)


def task_for_sample(dataset: Any, sample: dict[str, Any]) -> str:
    if "task" in sample:
        try:
            return _clean_task_text(sample["task"], source="sample['task']")
        except ValueError:
            if "task_index" not in sample:
                raise
    if "task_index" not in sample:
        raise KeyError("LeRobot sample has neither 'task' nor 'task_index'.")
    task_index = int(_as_scalar(sample["task_index"], name="task_index"))
    tasks = getattr(dataset.meta, "tasks", None)
    if tasks is None:
        raise RuntimeError("Dataset metadata does not expose task strings.")

    # LeRobot 0.5.x stores a DataFrame indexed by task text with task_index as a
    # column. Match the actual id first; row position is only a compatibility
    # fallback for older/hand-authored metadata.
    if hasattr(tasks, "columns") and "task_index" in tasks.columns:
        match = tasks[tasks["task_index"] == task_index]
        if len(match) > 1:
            raise ValueError(f"Task metadata contains multiple rows for task_index={task_index}.")
        if len(match) == 1:
            value = match.iloc[0]["task"] if "task" in tasks.columns else match.index[0]
            return _clean_task_text(value, source=f"task metadata for task_index={task_index}")
    if isinstance(tasks, dict):
        if task_index in tasks:
            return _clean_task_text(tasks[task_index], source=f"task metadata for task_index={task_index}")
        if str(task_index) in tasks:
            return _clean_task_text(tasks[str(task_index)], source=f"task metadata for task_index={task_index}")
        raise KeyError(f"Cannot resolve task_index={task_index} in task metadata.")
    if isinstance(tasks, (list, tuple)):
        try:
            value = tasks[task_index]
        except IndexError as exc:
            raise KeyError(f"Cannot resolve task_index={task_index} in task metadata.") from exc
        return _clean_task_text(value, source=f"task metadata for task_index={task_index}")
    if hasattr(tasks, "iloc"):
        try:
            row = tasks.iloc[task_index]
        except (IndexError, KeyError) as exc:
            raise KeyError(f"Cannot resolve task_index={task_index} in task metadata.") from exc
        value = row["task"] if hasattr(tasks, "columns") and "task" in tasks.columns else row.name
        return _clean_task_text(value, source=f"task metadata row {task_index}")
    raise TypeError(f"Unsupported task metadata type: {type(tasks).__name__}.")


class LeRobotWorldCriticDataset(Dataset):
    """Episode-safe temporal windows over an official LeRobot v3 dataset."""

    def __init__(
        self,
        dataset: Any,
        config: DataConfig,
        episode_ids: Iterable[int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.window = config.history_size + config.prediction_horizon
        if config.history_size < 1:
            raise ValueError("history_size must be positive.")
        if config.prediction_horizon != 1:
            raise NotImplementedError("The current model implements an exact one-step action-conditioned target.")
        if not config.image_keys:
            raise ValueError("At least one image feature must be configured in data.image_keys.")
        available_features = set(dataset.features)
        required = {config.action_key, config.return_key, *config.image_keys}
        missing = sorted(required - available_features)
        if missing:
            raise KeyError(f"LeRobot dataset is missing required features: {missing}")

        indexing_started = time.monotonic()
        print("[dataset] loading episode_index column...", flush=True)
        episode_by_row = episode_ids_from_dataset(dataset)
        self.episode_by_row = episode_by_row
        print(
            "[dataset] episode_index column loaded: "
            f"rows={episode_by_row.size}, elapsed_s={time.monotonic() - indexing_started:.1f}",
            flush=True,
        )
        if episode_by_row.size == 0:
            raise ValueError("Dataset contains no rows.")

        # The table is required to be episode-contiguous.  Build all episode
        # ranges in one pass over the episode column.  The previous
        # implementation called ``np.flatnonzero(episode_by_row == id)`` for
        # every episode, which scans a 2.2M-row dataset once per episode and
        # can turn index construction into billions of comparisons.
        boundaries = np.flatnonzero(episode_by_row[1:] != episode_by_row[:-1]) + 1
        range_starts = np.concatenate(
            (np.asarray([0], dtype=np.int64), boundaries.astype(np.int64, copy=False))
        )
        range_ends = np.concatenate(
            (boundaries.astype(np.int64, copy=False), np.asarray([episode_by_row.size], dtype=np.int64))
        )
        episode_ranges: dict[int, tuple[int, int]] = {}
        for start, end in zip(range_starts.tolist(), range_ends.tolist(), strict=True):
            episode_id = int(episode_by_row[start])
            if episode_id in episode_ranges:
                raise ValueError(f"episode_index={episode_id} is not contiguous in the LeRobot table.")
            episode_ranges[episode_id] = (int(start), int(end))

        available_episodes = set(episode_ranges)
        allowed = set(map(int, episode_ids)) if episode_ids is not None else available_episodes
        unknown = allowed - available_episodes
        if unknown:
            raise ValueError(f"Requested episodes are absent from this dataset: {sorted(unknown)[:10]}")
        print("[dataset] loading frame_index column...", flush=True)
        frame_by_row = _scalar_column(dataset, "frame_index", np.int64).reshape(-1)
        print(
            "[dataset] frame_index column loaded: "
            f"rows={frame_by_row.size}, elapsed_s={time.monotonic() - indexing_started:.1f}",
            flush=True,
        )
        if frame_by_row.shape != episode_by_row.shape:
            raise ValueError(
                "episode_index and frame_index columns have different lengths: "
                f"{episode_by_row.shape} vs {frame_by_row.shape}"
            )

        # Keep one int64 start row per temporal window instead of one small
        # NumPy array per window.  The latter carries substantial Python/NumPy
        # object overhead for large datasets; the complete row sequence is
        # reconstructed in __getitem__ from the start and fixed window size.
        window_start_chunks: list[np.ndarray] = []
        allowed_sorted = sorted(allowed)
        print(
            "[dataset] building temporal index: "
            f"rows={episode_by_row.size}, episodes={len(allowed_sorted)}, window={self.window}",
            flush=True,
        )
        skipped_short = 0
        for episode_position, episode_id in enumerate(allowed_sorted, start=1):
            row_start, row_end = episode_ranges[episode_id]
            episode_frames = frame_by_row[row_start:row_end]
            if episode_frames.size > 1 and not np.all(np.diff(episode_frames) == 1):
                raise ValueError(
                    f"episode_index={episode_id} has non-consecutive frame_index values: "
                    f"{episode_frames[:10].tolist()}"
                )
            if episode_frames.size < self.window:
                skipped_short += 1
                continue

            window_start_chunks.append(
                np.arange(
                    row_start,
                    row_end - self.window + 1,
                    dtype=np.int64,
                )
            )
            if episode_position == 1 or episode_position % 100 == 0 or episode_position == len(allowed_sorted):
                current_windows = sum(chunk.size for chunk in window_start_chunks)
                print(
                    "[dataset] temporal index progress: "
                    f"episodes={episode_position}/{len(allowed_sorted)}, "
                    f"windows={current_windows}, elapsed_s={time.monotonic() - indexing_started:.1f}",
                    flush=True,
                )

        self.window_starts = (
            np.concatenate(window_start_chunks)
            if window_start_chunks
            else np.empty(0, dtype=np.int64)
        )
        # Keep the old attribute name as a compatibility alias for callers
        # that only inspect its length.  Entries now contain compact start
        # rows rather than materialized window arrays.
        self.indices = self.window_starts
        print(
            "[dataset] temporal index ready: "
            f"windows={len(self.window_starts)}, skipped_short_episodes={skipped_short}, "
            f"elapsed_s={time.monotonic() - indexing_started:.1f}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self.window_starts[index])
        rows = np.arange(start, start + self.window, dtype=np.int64)
        samples = [self.dataset[int(row)] for row in rows]
        current = samples[:-1]
        expected_episode = int(self.episode_by_row[int(rows[0])])
        sample_episodes = [
            int(_as_scalar(sample["episode_index"], name="episode_index")) for sample in samples
        ]
        if any(episode != expected_episode for episode in sample_episodes):
            raise ValueError(
                f"Window rows {rows.tolist()} disagree with their episode_index column: {sample_episodes}."
            )
        sample_frames = [int(_as_scalar(sample["frame_index"], name="frame_index")) for sample in samples]
        if len(sample_frames) > 1 and any(
            right != left + 1 for left, right in zip(sample_frames, sample_frames[1:])
        ):
            raise ValueError(f"Window has non-consecutive frame_index values: {sample_frames}.")
        instruction = task_for_sample(self.dataset, current[0])
        if any(task_for_sample(self.dataset, sample) != instruction for sample in samples):
            raise ValueError(
                f"Window {int(rows[0])}:{int(rows[-1])} changes task instruction within an episode."
            )
        images = [
            [_to_image_tensor(sample[key]) for key in self.config.image_keys]
            for sample in samples
        ]

        actions = torch.stack(
            [torch.as_tensor(sample[self.config.action_key], dtype=torch.float32).reshape(-1) for sample in current]
        )
        if not torch.isfinite(actions).all():
            raise ValueError(f"Window {int(rows[0])}:{int(rows[-1])} contains non-finite actions.")
        if self.config.normalize_action:
            if self.config.action_mean is None or self.config.action_std is None:
                raise RuntimeError("Action normalization is enabled but train statistics are missing.")
            mean = torch.as_tensor(self.config.action_mean, dtype=actions.dtype)
            std = torch.as_tensor(self.config.action_std, dtype=actions.dtype)
            if mean.numel() != actions.size(-1):
                raise ValueError(
                    f"Action normalization dimension {mean.numel()} != action dimension {actions.size(-1)}."
                )
            actions = (actions - mean) / std
        return_targets = None
        has_returns = [self.config.return_key in sample for sample in current]
        if any(has_returns) and not all(has_returns):
            raise ValueError(f"Return field {self.config.return_key!r} is missing from part of a window.")
        if all(has_returns):
            return_values = []
            for sample in current:
                value = torch.as_tensor(sample[self.config.return_key], dtype=torch.float32)
                if value.numel() != 1:
                    raise ValueError(
                        f"Return field {self.config.return_key!r} must be scalar, got shape {tuple(value.shape)}."
                    )
                return_values.append(value.reshape(1))
            return_targets = torch.stack(return_values)
            if not torch.isfinite(return_targets).all():
                raise ValueError(f"Window {int(rows[0])}:{int(rows[-1])} contains non-finite returns.")
        else:
            raise KeyError(self.config.return_key)

        episode_id = expected_episode
        frame_indices = torch.as_tensor(sample_frames[:-1], dtype=torch.long)
        batch: dict[str, Any] = {
            "images": images,
            "actions": actions,
            "instruction": instruction,
            "valid_mask": torch.ones(len(current), dtype=torch.bool),
            "episode_id": episode_id,
            "frame_indices": frame_indices,
            "sample_id": f"{episode_id}:{int(frame_indices[0])}",
        }
        if return_targets is not None:
            batch["return_targets"] = return_targets
        if self.config.state_key is not None:
            has_states = [self.config.state_key in sample for sample in samples[1:]]
            if any(has_states) and not all(has_states):
                raise ValueError(f"State field {self.config.state_key!r} is missing from part of a window.")
        if self.config.state_key is not None and all(has_states):
            batch["next_state_vector"] = torch.stack(
                [
                    torch.as_tensor(sample[self.config.state_key], dtype=torch.float32).reshape(-1)
                    for sample in samples[1:]
                ]
            )
            if not torch.isfinite(batch["next_state_vector"]).all():
                raise ValueError(f"Window {int(rows[0])}:{int(rows[-1])} contains non-finite state targets.")
        return batch


class WorldCriticCollator:
    def __init__(self, processor: Any, image_size: int, max_text_length: int) -> None:
        self.image_processor = getattr(processor, "image_processor", processor)
        self.tokenizer = getattr(processor, "tokenizer", None)
        if self.tokenizer is None:
            raise TypeError("Processor must expose a tokenizer for language instructions.")
        self.image_size = image_size
        self.max_text_length = max_text_length

    @torch.no_grad()
    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty sample list.")
        batch = len(samples)
        time = len(samples[0]["images"])
        if time < 2:
            raise ValueError("Each sample must contain current and next observation images.")
        views = len(samples[0]["images"][0])
        if views < 1:
            raise ValueError("Each timestep must contain at least one camera image.")
        if any(
            len(sample["images"]) != time
            or any(len(step_views) != views for step_views in sample["images"])
            for sample in samples
        ):
            raise ValueError("All samples in a batch must contain the same timestep and camera counts.")
        flat_images = [
            image
            for sample in samples
            for step_views in sample["images"]
            for image in step_views
        ]
        image_list = []
        do_rescale = None
        for image in flat_images:
            if not torch.is_tensor(image):
                image = torch.as_tensor(image)
            if image.ndim != 3:
                raise ValueError(f"Expected a CHW or HWC image tensor, got {image.shape}")
            channels_first = image.shape[0] in (1, 3, 4)
            channels_last = image.shape[-1] in (1, 3, 4)
            if channels_first == channels_last:
                raise ValueError(
                    f"Cannot unambiguously infer CHW versus HWC channel layout for image {tuple(image.shape)}."
                )
            if channels_first:
                image = image.permute(1, 2, 0)
            array = image.cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Images must contain only finite values.")
            image_is_float = np.issubdtype(array.dtype, np.floating)
            if image_is_float:
                minimum = float(array.min())
                maximum = float(array.max())
                if minimum < -1e-6:
                    raise ValueError(
                        "Image processor expects non-negative raw pixels; already-normalized images are unsupported."
                    )
                if maximum <= 1.0 + 1e-6:
                    current_do_rescale = False
                elif maximum <= 255.0 + 1e-6:
                    current_do_rescale = True
                else:
                    raise ValueError(f"Float image range [{minimum}, {maximum}] exceeds [0,255].")
            elif np.issubdtype(array.dtype, np.integer):
                minimum = int(array.min())
                maximum = int(array.max())
                if minimum < 0 or maximum > 255:
                    raise ValueError(f"Integer image range [{minimum}, {maximum}] exceeds [0,255].")
                current_do_rescale = True
            else:
                raise TypeError(f"Unsupported image dtype {array.dtype}.")
            image_list.append(array)
            if do_rescale is None:
                do_rescale = current_do_rescale
            elif do_rescale != current_do_rescale:
                raise ValueError("A batch mixes [0,1] float images and uint8/[0,255] images.")
        processor_kwargs = dict(
            images=image_list,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
        )
        if do_rescale is False:
            processor_kwargs["do_rescale"] = False
        processed = self.image_processor(**processor_kwargs)["pixel_values"]
        processed = processed.view(batch, time, views, *processed.shape[1:])
        tokenized = self.tokenizer(
            [sample["instruction"] for sample in samples],
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        output = {
            "images": processed,
            "actions": torch.stack([sample["actions"] for sample in samples]),
            "instruction_input_ids": tokenized["input_ids"],
            "instruction_attention_mask": tokenized["attention_mask"],
            "valid_mask": torch.stack([sample["valid_mask"] for sample in samples]),
            "episode_id": torch.as_tensor([sample["episode_id"] for sample in samples], dtype=torch.long),
            "frame_indices": torch.stack([sample["frame_indices"] for sample in samples]),
            "sample_id": [sample["sample_id"] for sample in samples],
        }
        has_returns = ["return_targets" in sample for sample in samples]
        if any(has_returns) and not all(has_returns):
            raise ValueError("A batch mixes samples with and without return_targets.")
        if all(has_returns):
            output["return_targets"] = torch.stack([sample["return_targets"] for sample in samples])
        has_states = ["next_state_vector" in sample for sample in samples]
        if any(has_states) and not all(has_states):
            raise ValueError("A batch mixes samples with and without next_state_vector targets.")
        if all(has_states):
            output["next_state_vector"] = torch.stack([sample["next_state_vector"] for sample in samples])
        return output


def load_lerobot_dataset(config: DataConfig) -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("LeRobot support requires lerobot>=0.5.1,<0.6.") from exc
    dataset = LeRobotDataset(
        repo_id=config.repo_id,
        root=config.root,
        revision=config.revision,
    )
    _preflight_video_decoder(dataset, config)
    return dataset


def _feature_dtype(feature: Any) -> str:
    if isinstance(feature, Mapping):
        value = feature.get("dtype", "")
    else:
        value = getattr(feature, "dtype", "")
    return str(value).strip().lower()


def _preflight_video_decoder(dataset: Any, config: DataConfig) -> None:
    """Load TorchCodec in the parent process so ABI errors are actionable."""
    features = getattr(dataset, "features", {})
    has_video = any(_feature_dtype(features.get(key)) == "video" for key in config.image_keys)
    raw_backend = getattr(dataset, "_video_backend", None)
    if raw_backend is None:
        raw_backend = getattr(dataset, "video_backend", None)
    backend = str(raw_backend or "torchcodec").lower()
    if not has_video or backend != "torchcodec":
        return
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as exc:
        try:
            torchcodec_version = metadata.version("torchcodec")
        except metadata.PackageNotFoundError:
            torchcodec_version = "not installed"
        try:
            import torch

            torch_version = torch.__version__
        except Exception:
            torch_version = "unknown"
        raise RuntimeError(
            "LeRobot is using TorchCodec for video features, but its native decoder could not be loaded. "
            f"Detected torch={torch_version}, torchcodec={torchcodec_version}. WCM_v2 requires "
            "torch 2.7.x with torchcodec 0.5.x. Reinstall the resolved environment from WCM_v2 with "
            '`uv pip install --refresh --reinstall-package torchcodec -e ".[all]"` and verify that FFmpeg shared libraries are available '
            "(`ffmpeg -version` must work; on Linux the libav*.so files must be discoverable)."
        ) from exc


def build_processor(model_config: ModelConfig) -> Any:
    try:
        from transformers import AutoImageProcessor, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Data preprocessing requires transformers.") from exc

    return CombinedProcessor(
        image_processor=AutoImageProcessor.from_pretrained(model_config.vision.model_name),
        tokenizer=AutoTokenizer.from_pretrained(model_config.language.model_name),
    )


def build_datasets(config: DataConfig, manifest_path: str | Path | None = None):
    dataset = load_lerobot_dataset(config)
    episode_ids = episode_ids_from_dataset(dataset)
    if config.split_manifest:
        split = load_episode_split(config.split_manifest)
    elif manifest_path is not None and Path(manifest_path).exists():
        split = load_episode_split(manifest_path)
    else:
        split = build_episode_split(episode_ids, config.val_fraction, config.split_seed)
        if manifest_path is not None:
            save_episode_split(split, manifest_path, config.split_seed, config.val_fraction)
    available = set(map(int, np.unique(episode_ids).tolist()))
    requested = set(split.train) | set(split.val)
    unknown = requested - available
    if unknown:
        raise ValueError(f"Split manifest references episodes absent from this dataset: {sorted(unknown)[:10]}")
    missing = available - requested
    if missing:
        raise ValueError(f"Split manifest does not cover all dataset episodes: {sorted(missing)[:10]}")
    fit_action_normalization(dataset, config, split.train)
    validate_action_normalization(dataset, config)
    train = LeRobotWorldCriticDataset(dataset, config, split.train)
    val = LeRobotWorldCriticDataset(dataset, config, split.val) if split.val else None
    return dataset, train, val, split


def validate_action_normalization(dataset: Any, config: DataConfig) -> None:
    if not config.normalize_action:
        return
    if config.action_mean is None or config.action_std is None:
        raise ValueError("Checkpoint/data config is missing action normalization statistics.")
    action_dim = infer_feature_dim(dataset, config.action_key)
    if len(config.action_mean) != action_dim or len(config.action_std) != action_dim:
        raise ValueError(
            f"Checkpoint action statistics have dimension {len(config.action_mean)}; dataset action_dim={action_dim}."
        )
    if any(not np.isfinite(value) for value in [*config.action_mean, *config.action_std]):
        raise ValueError("Action normalization statistics contain non-finite values.")
    if any(value <= 0 for value in config.action_std):
        raise ValueError("Action normalization std values must be positive.")


def canonicalize_return_target(value: torch.Tensor) -> torch.Tensor:
    """Convert LeRobot [B,T] or legacy [B,T,1] returns to [B,T,1]."""
    value = value.float()
    if value.ndim == 2:
        return value.unsqueeze(-1)
    if value.ndim == 3 and value.size(-1) == 1:
        return value
    raise ValueError(f"Return target must be [B,T] or [B,T,1], got {value.shape}")
