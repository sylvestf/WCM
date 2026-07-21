from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Convert a local LeRobot v1.6 copy to v2.0.")
    result.add_argument("--source", required=True)
    result.add_argument("--output", required=True)
    task_source = result.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--single-task")
    task_source.add_argument("--tasks-col")
    task_source.add_argument("--tasks-path")
    return result


def _load_official_module() -> Any:
    """Load the v1.6 converter from the historical package layout.

    The converter moved/was removed in later LeRobot releases.  We deliberately
    keep the import local so the training (v3) environment can still import this
    wrapper and report an actionable error when a legacy environment is needed.
    """
    try:
        from lerobot.datasets.v2 import convert_dataset_v1_to_v2 as official

        return official
    except ImportError as first_error:
        try:
            from lerobot.scripts import convert_dataset_v1_to_v2 as official

            return official
        except ImportError as second_error:
            raise ImportError(
                "The v1.6 converter is not shipped by this LeRobot environment. "
                "Run this wrapper with a historical LeRobot release containing "
                "lerobot.datasets.v2.convert_dataset_v1_to_v2 (for example v0.3.x)."
            ) from second_error


def _official_symbol(official: Any, name: str, *, default: Any = None) -> Any:
    """Resolve a converter symbol across historical LeRobot module layouts."""

    value = getattr(official, name, None)
    if value is not None:
        return value
    # In a few development snapshots constants/helpers were imported into the
    # converter only inside ``main``.  The public utils module remains the
    # stable source of the v2 paths and JSON helpers.
    try:
        from lerobot.datasets import utils as dataset_utils

        value = getattr(dataset_utils, name, None)
    except ImportError:
        value = None
    if value is None:
        try:
            from lerobot.datasets import video_utils

            value = getattr(video_utils, name, None)
        except ImportError:
            value = None
    return default if value is None else value


def _is_feature_type(feature: Any, datasets_module: Any, *names: str) -> bool:
    """Return whether ``feature`` belongs to one of the datasets feature types.

    The ``datasets`` package has moved a few feature classes between releases
    (and some releases expose them only through ``datasets.features``).  The
    v1.6 converter is intentionally run in a historical environment, so do not
    rely on one exact import path.  Falling back to the class name also keeps
    this helper usable with old local mirrors whose feature objects have been
    deserialised without the original class module.
    """

    feature_class_name = feature.__class__.__name__
    features_module = getattr(datasets_module, "features", datasets_module)
    for name in names:
        for module in (datasets_module, features_module):
            candidate = getattr(module, name, None)
            if candidate is not None:
                try:
                    if isinstance(feature, candidate):
                        return True
                except TypeError:
                    # A few optional extension features expose a non-class
                    # descriptor under the public name.  The class-name
                    # fallback below is still safe in that case.
                    pass
        if feature_class_name == name:
            return True
    return False


def _sample_shape(value: Any) -> tuple[int, ...]:
    """Best-effort shape extraction for a decoded v1 feature value."""

    if value is None:
        return ()
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            pass
    # PIL images expose height/width but not always a useful numpy shape when
    # backed by a lazy decoder.
    if hasattr(value, "height") and hasattr(value, "width"):
        channels = 1
        mode = getattr(value, "mode", None)
        if mode in {"LA"}:
            channels = 2
        elif mode in {"RGB"}:
            channels = 3
        elif mode in {"RGBA"}:
            channels = 4
        return int(value.height), int(value.width), channels
    try:
        return tuple(int(dim) for dim in np.asarray(value).shape)
    except (TypeError, ValueError):
        return ()


def _feature_dtype(feature: Any, sample: Any = None) -> Any:
    """Get a serialisable dtype from old/new ``datasets`` feature objects."""

    dtype = getattr(feature, "dtype", None)
    if dtype is None:
        nested = getattr(feature, "feature", None)
        dtype = getattr(nested, "dtype", None)
    if dtype is None and sample is not None:
        try:
            dtype = np.asarray(sample).dtype.name
        except (TypeError, ValueError):
            dtype = None
    if dtype is None:
        return "float32"
    return str(dtype)


def _infer_sequence_spec(feature: Any, sample: Any) -> tuple[Any, tuple[int, ...], int | None]:
    """Infer dtype/shape for a fixed- or variable-length Sequence feature.

    ``datasets.Sequence(length=None)`` is legal and is used by a number of
    early LeRobot exports.  The official converter assumed ``length`` was an
    integer; infer it from the first row instead and reject only genuinely
    ragged data that cannot be represented by the v2 metadata contract.
    """

    nested = getattr(feature, "feature", None)
    length = getattr(feature, "length", None)
    sample_shape = _sample_shape(sample)
    # Hugging Face datasets historically used ``-1`` as the sentinel for a
    # variable-length Sequence; newer releases use ``None``.  Treat both as
    # unknown and infer the concrete rectangular length from the data.
    if length is None or (isinstance(length, (int, np.integer)) and int(length) < 0):
        if not sample_shape:
            raise ValueError("Cannot infer the length of a variable-length v1 Sequence from an empty sample.")
        length = sample_shape[0]
    try:
        length = int(length)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported v1 Sequence length {length!r}.") from exc
    if length < 0:
        raise ValueError(f"v1 Sequence length must be non-negative, got {length}.")

    nested_sample = None
    if length and sample is not None:
        try:
            nested_sample = sample[0]
        except (IndexError, KeyError, TypeError):
            try:
                nested_sample = np.asarray(sample)[0]
            except (IndexError, TypeError, ValueError):
                nested_sample = None
    nested_dtype = _feature_dtype(nested, nested_sample)
    nested_shape = tuple(sample_shape[1:]) if len(sample_shape) > 1 else ()
    declared_shape = getattr(nested, "shape", None)
    if declared_shape is not None:
        try:
            nested_shape = tuple(int(dim) for dim in declared_shape)
        except (TypeError, ValueError):
            pass
    return nested_dtype, (length, *nested_shape), length


def _infer_features(official: Any, dataset: Any) -> dict[str, dict[str, Any]]:
    """Infer v1 feature metadata without requiring a robot config.

    Older official code unconditionally dereferences ``robot_config.type`` even
    though robot configuration is optional at the CLI.  Generic motor names are
    sufficient for conversion and preserve the actual dtype/shape information.
    """
    # Depending on the historical release, the converter exposes either the
    # imported ``datasets`` module or only its helper functions.  Importing the
    # package directly keeps this wrapper compatible with both layouts.
    try:
        datasets_module = official.datasets
    except AttributeError:
        try:
            import datasets as datasets_module
        except ImportError as exc:
            raise ImportError("The historical LeRobot environment must provide the 'datasets' package.") from exc
    image_channels = getattr(official, "get_image_pixel_channels", None)
    result: dict[str, dict[str, Any]] = {}
    for key, feature in dataset.features.items():
        sample = None
        if len(dataset):
            try:
                sample = dataset[0][key]
            except (IndexError, KeyError, TypeError):
                sample = None

        if _is_feature_type(feature, datasets_module, "Value"):
            dtype, shape, names = _feature_dtype(feature, sample), (1,), None
        elif _is_feature_type(feature, datasets_module, "Sequence", "LargeList", "List"):
            declared_length = getattr(feature, "length", None)
            variable_length = declared_length is None or (
                isinstance(declared_length, (int, np.integer)) and int(declared_length) < 0
            )
            if variable_length and len(dataset) > 1:
                # A v2 feature shape must be rectangular.  Variable-length
                # sequences are accepted only when all rows happen to share
                # one concrete length; otherwise fail before writing a
                # misleading metadata declaration.
                observed_lengths: set[int] = set()
                for row_index in range(len(dataset)):
                    try:
                        row_shape = _sample_shape(dataset[row_index][key])
                    except (IndexError, KeyError, TypeError):
                        continue
                    if row_shape:
                        observed_lengths.add(row_shape[0])
                if len(observed_lengths) > 1:
                    raise ValueError(
                        f"v1 Sequence feature {key!r} is ragged with lengths "
                        f"{sorted(observed_lengths)}; v2 metadata requires a fixed shape."
                    )
            dtype, shape, length = _infer_sequence_spec(feature, sample)
            names = {"motors": [f"motor_{i}" for i in range(length)]} if length is not None else None
        elif _is_feature_type(feature, datasets_module, "Image"):
            if sample is None:
                raise ValueError(f"Cannot infer image shape for v1 feature {key!r} from an empty dataset.")
            if image_channels is not None:
                try:
                    channels = int(image_channels(sample))
                except (AttributeError, TypeError, ValueError):
                    channels = _sample_shape(sample)[-1] if _sample_shape(sample) else 3
            else:
                mode = getattr(sample, "mode", None)
                channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}.get(mode)
                if channels is None:
                    sample_shape = _sample_shape(sample)
                    channels = sample_shape[-1] if len(sample_shape) >= 3 else 1
            sample_shape = _sample_shape(sample)
            if len(sample_shape) >= 2:
                shape = (sample_shape[0], sample_shape[1], channels)
            else:
                raise ValueError(f"Cannot infer image shape for v1 feature {key!r}: sample has shape {sample_shape}.")
            dtype, names = "image", ["height", "width", "channels"]
        elif (
            getattr(feature, "_type", None) in {"VideoFrame", "Video"}
            or feature.__class__.__name__ in {"VideoFrame", "Video"}
        ):
            dtype, shape, names = "video", None, ["height", "width", "channels"]
        elif _is_feature_type(feature, datasets_module, "Array2D", "Array3D", "Array4D", "Array5D") or (
            getattr(feature, "shape", None) is not None and getattr(feature, "dtype", None) is not None
        ):
            shape = getattr(feature, "shape", None) or _sample_shape(sample)
            if not shape:
                raise ValueError(f"Cannot infer array shape for v1 feature {key!r}.")
            try:
                shape = tuple(int(dim) for dim in shape)
            except (TypeError, ValueError):
                shape = _sample_shape(sample)
            if not shape or any(dim < 0 for dim in shape):
                raise ValueError(f"Cannot infer concrete array shape for v1 feature {key!r}: {shape!r}.")
            dtype, names = _feature_dtype(feature, sample), None
        else:
            raise TypeError(f"Unsupported v1 feature {key!r}: {feature!r}")
        result[key] = {"dtype": dtype, "shape": shape, "names": names}
    return result


def _episode_task_lists(tasks_by_episode: dict[int, Any]) -> dict[int, list[str]]:
    """Normalize task metadata to the v2 contract (list[str] per episode)."""
    normalized: dict[int, list[str]] = {}
    for episode, tasks in tasks_by_episode.items():
        if isinstance(tasks, str):
            values = [tasks]
        elif isinstance(tasks, (list, tuple)):
            values = list(tasks)
        else:
            raise ValueError(f"Task mapping for episode {episode} must be a string or list of strings.")
        if not values or any(not isinstance(task, str) or not task.strip() for task in values):
            raise ValueError(f"Task mapping for episode {episode} contains an empty/non-string instruction.")
        normalized[int(episode)] = values
    return normalized


def _single_task_mapping(tasks_by_episode: dict[int, Any]) -> dict[int, str]:
    """Prepare the mapping accepted by the historical official helper."""
    normalized = _episode_task_lists(tasks_by_episode)
    if any(len(tasks) != 1 for tasks in normalized.values()):
        raise ValueError(
            "v1.6 --tasks-path accepts one instruction per episode; use --tasks-col "
            "for episodes containing multiple instructions."
        )
    return {episode: tasks[0] for episode, tasks in normalized.items()}


def _find_v1_video(source: Path, filename: str) -> Path:
    """Find one v1.6 video file, including both released directory layouts.

    Early v1.6 exports put files directly below a ``videos*`` directory,
    while some local/HF mirrors already contain the three-level layout that
    the official converter recognizes (``videos*/.../.../<file>``).  Looking
    up the exact basename recursively keeps this wrapper compatible with both
    forms without depending on a particular camera-directory naming scheme.
    """
    video_roots = sorted(path for path in source.glob("videos*") if path.is_dir())
    matches = [
        candidate
        for root in video_roots
        for candidate in root.rglob(filename)
        if candidate.is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one v1.6 video {filename!r} below videos*; found {matches}."
        )
    return matches[0]


def main() -> None:
    args = parser().parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if source == output or source in output.parents:
        raise ValueError("The v2.0 output must be outside the v1.6 source tree.")
    output.parent.mkdir(parents=True, exist_ok=True)

    official = _load_official_module()

    v1_info_path = _official_symbol(official, "V1_INFO_PATH", default="meta_data/info.json")
    v1_video_file = _official_symbol(
        official,
        "V1_VIDEO_FILE",
        default="{video_key}_episode_{episode_index:06d}.mp4",
    )
    info_path = _official_symbol(official, "INFO_PATH", default="meta/info.json")
    episodes_path = _official_symbol(official, "EPISODES_PATH", default="meta/episodes.jsonl")
    tasks_path = _official_symbol(official, "TASKS_PATH", default="meta/tasks.jsonl")
    parquet_path = _official_symbol(
        official,
        "DEFAULT_PARQUET_PATH",
        default="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    video_path = _official_symbol(
        official,
        "DEFAULT_VIDEO_PATH",
        default="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    chunk_size = int(_official_symbol(official, "DEFAULT_CHUNK_SIZE", default=1000))
    load_json = _official_symbol(official, "load_json")
    write_json = _official_symbol(official, "write_json")
    write_jsonlines = _official_symbol(official, "write_jsonlines")
    split_parquet_by_episodes = _official_symbol(official, "split_parquet_by_episodes")
    convert_stats_to_json = _official_symbol(official, "convert_stats_to_json")
    add_task_index_by_episodes = _official_symbol(official, "add_task_index_by_episodes")
    add_task_index_from_tasks_col = _official_symbol(official, "add_task_index_from_tasks_col")
    required_symbols = [load_json, write_json, write_jsonlines, split_parquet_by_episodes, convert_stats_to_json]
    required_symbols.append(
        add_task_index_from_tasks_col
        if args.tasks_col is not None
        else add_task_index_by_episodes
    )
    if any(symbol is None for symbol in required_symbols):
        raise ImportError(
            "The historical v1.6 converter is missing one of its required JSON/parquet helpers. "
            "Use the LeRobot 0.3.x converter environment or provide a compatible wrapper."
        )
    datasets_module = getattr(official, "datasets", None)
    if datasets_module is None:
        try:
            import datasets as datasets_module
        except ImportError as exc:
            raise ImportError("The historical LeRobot environment must provide the 'datasets' package.") from exc
    load_dataset = getattr(datasets_module, "load_dataset", None)
    if load_dataset is None:
        raise ImportError("The historical datasets package has no load_dataset() helper.")

    source_info = source / v1_info_path
    if not source_info.is_file():
        raise FileNotFoundError(
            f"v1.6 metadata file is missing: {source_info}. Expected the official meta_data/info.json layout."
        )
    if not (source / "data").is_dir():
        raise FileNotFoundError(
            f"v1.6 parquet data directory is missing: {source / 'data'}."
        )
    metadata_v1 = load_json(source_info)
    dataset = load_dataset("parquet", data_dir=source / "data", split="train")
    episode_indices = sorted(dataset.unique("episode_index"))
    if not episode_indices:
        raise ValueError("v1.6 dataset contains no episodes; cannot create a v2.0 dataset.")
    # Always use the local generic inference.  Historical official helpers
    # disagree on whether ``robot_config`` is optional and several releases
    # unconditionally dereference it; conversion does not need robot-specific
    # names, so avoiding that call makes the wrapper version-stable.
    features = _infer_features(official, dataset)
    video_keys = [key for key, feature in features.items() if feature["dtype"] == "video"]
    if episode_indices != list(range(len(episode_indices))):
        raise ValueError("v1.6 episode_index must be contiguous and zero-based.")

    if args.single_task:
        tasks_by_episode = dict.fromkeys(episode_indices, args.single_task)
        dataset, tasks = add_task_index_by_episodes(dataset, _single_task_mapping(tasks_by_episode))
        tasks_by_episode = _episode_task_lists(tasks_by_episode)
    elif args.tasks_path:
        tasks_by_episode = {
            int(key): value for key, value in load_json(Path(args.tasks_path)).items()
        }
        dataset, tasks = add_task_index_by_episodes(dataset, _single_task_mapping(tasks_by_episode))
        tasks_by_episode = _episode_task_lists(tasks_by_episode)
    else:
        dataset, tasks, tasks_by_episode = add_task_index_from_tasks_col(dataset, args.tasks_col)
        tasks_by_episode = _episode_task_lists(tasks_by_episode)
        # The historical converter removes the instruction column from the
        # parquet table, but several releases leave its stale description in
        # ``info.json``.  A v2 reader treats every metadata feature as a real
        # data column, so retain only features that are actually written.
        features.pop(args.tasks_col, None)

    (output / Path(tasks_path).parent).mkdir(parents=True, exist_ok=True)
    write_jsonlines(
        [{"task_index": index, "task": task} for index, task in enumerate(tasks)],
        output / tasks_path,
    )
    features["task_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    total_episodes = len(episode_indices)
    total_chunks = (total_episodes + chunk_size - 1) // chunk_size

    if video_keys:
        dataset = dataset.remove_columns(video_keys)
        for episode_chunk in range(total_chunks):
            start = chunk_size * episode_chunk
            stop = min(chunk_size * (episode_chunk + 1), total_episodes)
            for video_key in video_keys:
                for episode_index in range(start, stop):
                    filename = v1_video_file.format(
                        video_key=video_key,
                        episode_index=episode_index,
                    )
                    target = output / video_path.format(
                        episode_chunk=episode_chunk,
                        video_key=video_key,
                        episode_index=episode_index,
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(_find_v1_video(source, filename), target)

        for video_key in video_keys:
            first = output / video_path.format(
                episode_chunk=0,
                video_key=video_key,
                episode_index=0,
            )
            get_video_info = _official_symbol(official, "get_video_info")
            if get_video_info is None:
                raise ImportError("The historical v1.6 converter has no get_video_info() helper.")
            video_info = get_video_info(first)
            features[video_key]["shape"] = (
                video_info.pop("video.height"),
                video_info.pop("video.width"),
                video_info.pop("video.channels"),
            )
            features[video_key]["video_info"] = video_info

    episode_lengths = split_parquet_by_episodes(
        dataset,
        total_episodes,
        total_chunks,
        output,
    )
    write_jsonlines(
        [
            {
                "episode_index": episode,
                "tasks": tasks_by_episode[episode],
                "length": episode_lengths[episode],
            }
            for episode in episode_indices
        ],
        output / episodes_path,
    )
    info = {
        "codebase_version": "v2.0",
        "robot_type": "unknown",
        "total_episodes": total_episodes,
        "total_frames": len(dataset),
        "total_tasks": len(tasks),
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": total_chunks,
        "chunks_size": chunk_size,
        "fps": metadata_v1["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": parquet_path,
        "video_path": video_path if video_keys else None,
        "features": features,
    }
    write_json(info, output / info_path)
    convert_stats_to_json(source, output)
    print(json.dumps({"source": str(source), "output": str(output), "version": "v2.0"}, indent=2))


if __name__ == "__main__":
    main()
