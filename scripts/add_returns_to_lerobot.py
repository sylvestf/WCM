from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TARGET_LEROBOT_VERSION = "v3.0"
SUPPORTED_SOURCE_VERSIONS = ("v1.6", "v2.0", "v2.1", TARGET_LEROBOT_VERSION)

# Success fields seen in released LeRobot datasets and common exporters.  The
# order is intentional: explicit ``--success-key`` wins, then terminal/next
# transition fields, then episode-level aliases, and finally the generic name.
AUTO_SUCCESS_KEYS = (
    "next.success",
    "next.is_success",
    "is_episode_successful",
    "episode.success",
    "episode_success",
    "is_success",
    "success",
)


def conversion_plan(source_version: str) -> list[tuple[str, str]]:
    source_version = normalize_version(source_version) or source_version
    chain = {
        "v1.6": ("v2.0", "v2.1", "v3.0"),
        "v2.0": ("v2.1", "v3.0"),
        "v2.1": ("v3.0",),
        "v3.0": (),
    }
    if source_version not in chain:
        raise ValueError(f"Unsupported LeRobot source version: {source_version!r}")
    result = []
    current = source_version
    for target in chain[source_version]:
        result.append((current, target))
        current = target
    return result


def normalize_version(value: Any) -> str | None:
    if value is None:
        return None
    aliases = {
        "1": "v1.6",
        "1.0": "v1.6",
        "1.6": "v1.6",
        "v1": "v1.6",
        "v1.0": "v1.6",
        "v1.6": "v1.6",
        "2": "v2.0",
        "2.0": "v2.0",
        "v2": "v2.0",
        "v2.0": "v2.0",
        "2.1": "v2.1",
        "v2.1": "v2.1",
        "3": TARGET_LEROBOT_VERSION,
        "3.0": TARGET_LEROBOT_VERSION,
        "v3": TARGET_LEROBOT_VERSION,
        TARGET_LEROBOT_VERSION: TARGET_LEROBOT_VERSION,
    }
    normalized = str(value).strip().lower()
    direct = aliases.get(normalized)
    if direct is not None:
        return direct
    # Metadata in hand-authored/older Hub mirrors occasionally carries a
    # patch component (for example ``v2.1.0``).  The dataset schema, rather
    # than the package patch version, is what selects the migration chain.
    match = re.fullmatch(r"v?(\d+)(?:\.(\d+))?(?:\.\d+)?", normalized)
    if match:
        major, minor = int(match.group(1)), int(match.group(2) or 0)
        if major == 1 and minor in (0, 6):
            return "v1.6"
        if major == 2 and minor == 0:
            return "v2.0"
        if major == 2 and minor == 1:
            return "v2.1"
        if major == 3 and minor == 0:
            return TARGET_LEROBOT_VERSION
    return None


def dataset_root(root: str | Path, repo_id: str) -> Path:
    """Resolve either a direct dataset root or a parent containing ``repo_id``."""
    root_path = Path(root).expanduser().resolve()
    nested = root_path.joinpath(*repo_id.split("/"))
    candidates = (nested, root_path)
    for candidate in candidates:
        if any(
            marker.exists()
            for marker in (
                candidate / "meta" / "info.json",
                candidate / "meta" / "episodes.jsonl",
                candidate / "meta" / "tasks.parquet",
                candidate / "meta_data",
            )
        ):
            return candidate
    return root_path


def detect_lerobot_version(root: str | Path) -> str:
    root = Path(root)

    meta = root / "meta"
    meta_data = root / "meta_data"

    # Prefer unambiguous on-disk schema markers over a stale
    # ``codebase_version`` string.  Converters in the wild sometimes retain
    # the input version in info.json while already writing the next layout;
    # trusting that stale declaration would run a migration twice.
    #
    # v3.0 (LeRobot 0.4.x/0.5.x) stores episode metadata in parquet shards
    # below ``meta/episodes`` and tasks in ``meta/tasks.parquet``.  A few
    # local mirrors use a directory of task shards, so keep that variant too.
    if (meta / "episodes").is_dir() and (
        (meta / "tasks.parquet").is_file() or (meta / "tasks").is_dir()
    ):
        return TARGET_LEROBOT_VERSION

    # v2.0/v2.1 use the legacy JSONL metadata.  The presence of per-episode
    # stats is the only schema-level distinction between the two generations.
    if (meta / "episodes.jsonl").is_file() and (meta / "tasks.jsonl").is_file():
        if (meta / "episodes_stats.jsonl").is_file():
            return "v2.1"
        return "v2.0"

    # v1.6's canonical metadata lives under ``meta_data``.  Check this before
    # reading a generic ``meta/info.json`` so a stale info file cannot make an
    # otherwise unmistakable v1 tree look like a later generation.  Some
    # mirrors omit the safetensors stats file, therefore info.json alone is a
    # sufficient marker when the legacy data directory is present.
    if (meta_data / "info.json").is_file() and (
        (meta_data / "stats.safetensors").is_file()
        or (root / "data").is_dir()
        or any(root.glob("videos*"))
    ):
        return "v1.6"

    # Metadata-only snapshots may contain no data files.  In that case honour
    # an explicit declared generation, but validate it through the same alias
    # table used by --source-version.
    for info_path in (meta / "info.json", meta_data / "info.json"):
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not parse LeRobot metadata file {info_path}.") from exc
            raw_declared = info.get("codebase_version")
            declared = normalize_version(raw_declared)
            if raw_declared is not None and declared is None:
                raise ValueError(
                    f"Unsupported LeRobot codebase_version={raw_declared!r} in {info_path}. "
                    "Expected v1.6, v2.0, v2.1, or v3.0."
                )
            if declared is not None:
                return declared

    # Early v1.6 exports sometimes contain only ``meta_data`` and a ``train``
    # directory (rather than a ``data`` directory).  Keep this conservative
    # fallback for those mirrors, but do not classify an arbitrary folder named
    # ``meta_data`` as a dataset unless it also has a known marker above.
    if meta_data.is_dir():
        return "v1.6"
    if (root / "train").exists() and not meta.exists():
        return "v1.6"
    raise ValueError(
        f"Cannot determine the LeRobot schema at {root}. Expected a standard v3 layout "
        "(meta/episodes and meta/tasks.parquet) or a recognized legacy layout."
    )


def legacy_migration_error(version: str, repo_id: str, root: str | Path) -> RuntimeError:
    repo = repo_id or "<repo-id>"
    local_root = str(Path(root).expanduser())
    stages = {
        "v1.6": "convert_dataset_v1_to_v2",
        "v2.0": "convert_dataset_v20_to_v21",
        "v2.1": "convert_dataset_v21_to_v30",
    }
    missing_stage = stages.get(version, "legacy converter")
    return RuntimeError(
        f"Detected LeRobot {version} at {local_root}, but the {missing_stage} stage is not available "
        "in the active Python environment. The orchestrator supports v1.6 -> v2.0 -> v2.1 -> v3.0 "
        "through the local wrappers in scripts/, and each wrapper must be launched with a LeRobot release "
        "that actually contains its historical converter. Never migrate the only copy: use a disposable "
        "copy (the default pipeline does this) or a Hub test branch, verify the converted data, then add "
        f"the return field to v3.0. For repo {repo!r}, the missing official function is {missing_stage}."
    )


def require_local_v3(root: str | Path, repo_id: str) -> tuple[Path, str]:
    resolved = dataset_root(root, repo_id)
    version = detect_lerobot_version(resolved)
    if version != TARGET_LEROBOT_VERSION:
        raise legacy_migration_error(version, repo_id, resolved)
    return resolved, version


def parse_converter_commands(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected FROM=COMMAND for --converter-command, got {item!r}.")
        source, command = item.split("=", 1)
        source = normalize_version(source)
        if source not in {"v1.6", "v2.0", "v2.1"}:
            raise ValueError(f"Unsupported converter stage source: {source!r}.")
        if not command.strip():
            raise ValueError(f"Converter command for {source} is empty.")
        result[source] = command
    return result


def default_converter_commands() -> dict[str, Any]:
    scripts = Path(__file__).resolve().parent
    python = sys.executable
    return {
        "v1.6": [
            python,
            str(scripts / "convert_lerobot_v16_to_v20_local.py"),
            "--source",
            "{source}",
            "--output",
            "{output}",
        ],
        "v2.0": [
            python,
            str(scripts / "convert_lerobot_v20_to_v21_local.py"),
            "--source",
            "{source}",
            "--output",
            "{output}",
            "--repo-id",
            "{repo_id}",
        ],
        "v2.1": [
            python,
            str(scripts / "convert_lerobot_v21_to_v30_local.py"),
            "--source",
            "{source}",
            "--output",
            "{output}",
            "--repo-id",
            "{repo_id}",
        ],
    }


def _conversion_workspace_name(repo_id: str) -> str:
    """Return a collision-resistant, filesystem-safe staging directory name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", repo_id).strip("-._") or "dataset"
    digest = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:10]
    return f".{slug}-{digest}-lerobot-conversion"


def migrate_to_v3(
    source_root: Path,
    source_version: str,
    repo_id: str,
    output_parent: Path,
    converter_commands: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]], Path]:
    """Orchestrate explicit converters on disposable copies until v3 is reached."""
    output_parent = output_parent.resolve()
    source_root = source_root.resolve()
    workspace = output_parent / _conversion_workspace_name(repo_id)
    if workspace == source_root or source_root in workspace.parents or workspace in source_root.parents:
        raise ValueError("Conversion workspace and source dataset must not contain one another.")
    if workspace.exists():
        raise FileExistsError(f"Conversion workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    current_root = workspace / "stage_0_source"
    records: list[dict[str, Any]] = []
    try:
        shutil.copytree(source_root, current_root, copy_function=shutil.copy2)

        for index, (from_version, to_version) in enumerate(conversion_plan(source_version), start=1):
            template = converter_commands.get(from_version)
            if template is None:
                raise legacy_migration_error(from_version, repo_id, current_root)
            stage_output = workspace / f"stage_{index}_{to_version.replace('.', '_')}"
            format_values = {
                "repo_id": repo_id,
                "source": str(current_root),
                "output": str(stage_output),
                "from_version": from_version,
                "to_version": to_version,
            }
            if isinstance(template, list):
                command = [_format_command_template(str(part), format_values) for part in template]
                subprocess.run(command, check=True)
                recorded_command: Any = command
            else:
                # String templates are retained for custom commands, but execute
                # through the platform shell only after the caller has explicitly
                # opted into that form.  Quote substituted paths on Windows/POSIX
                # so spaces in dataset roots cannot split the command.
                shell_values = {
                    key: _shell_quote(str(value)) for key, value in format_values.items()
                }
                command_text = _format_command_template(template, shell_values)
                subprocess.run(command_text, check=True, shell=True)
                recorded_command = command_text
            produced = dataset_root(stage_output, repo_id)
            detected = detect_lerobot_version(produced)
            if detected != to_version:
                raise RuntimeError(
                    f"Converter {from_version}->{to_version} produced {detected} at {produced}."
                )
            current_root = produced
            records.append(
                {
                    "source_version": from_version,
                    "target_version": to_version,
                    "command": recorded_command,
                }
            )
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    return current_root, records, workspace


def _shell_quote(value: str) -> str:
    """Quote a substituted custom-command value for the host shell."""
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


_TEMPLATE_FIELD = re.compile(r"\{(repo_id|source|output|from_version|to_version)\}")


def _format_command_template(value: str, substitutions: dict[str, str]) -> str:
    """Replace only documented placeholders, preserving braces in user text."""

    return _TEMPLATE_FIELD.sub(lambda match: substitutions[match.group(1)], value)


def _template_contains_any_flag(template: Any, flags: tuple[str, ...]) -> bool:
    """Detect whether a custom converter template already carries a CLI flag."""
    def matches(token: str) -> bool:
        return any(token == flag or token.startswith(flag + "=") for flag in flags)

    if isinstance(template, (list, tuple)):
        return any(matches(str(part)) for part in template)
    if isinstance(template, str):
        # ``shlex`` handles quoted Windows/POSIX command arguments better than
        # a substring search (e.g. --single-task-name must not match
        # --single-task).  Fall back to a conservative substring check when a
        # user-supplied command has shell syntax shlex cannot parse.
        try:
            tokens = shlex.split(template, posix=os.name != "nt")
        except ValueError:
            tokens = template.split()
        return any(matches(token) for token in tokens)
    return False


def _remove_conversion_workspace(path: Path | None) -> None:
    if path is not None:
        # LeRobot/datasets may keep memory-mapped parquet handles alive for a
        # short time after the last sample is read.  Collect before attempting
        # removal so Windows can delete the disposable staging tree as well as
        # POSIX systems.
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)


def _parse_source_version(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    result = normalize_version(normalized)
    if result is None or result not in SUPPORTED_SOURCE_VERSIONS:
        choices = ", ".join(("auto", *SUPPORTED_SOURCE_VERSIONS))
        raise argparse.ArgumentTypeError(
            f"unsupported LeRobot source version {value!r}; use one of {choices} "
            "(common aliases such as 1.6, v2, and 3 are accepted)."
        )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Create a new LeRobot v3 dataset with pi-star-0.6-style return fields."
    )
    result.add_argument("--repo-id", required=True, help="Source LeRobot repo id.")
    result.add_argument(
        "--root",
        help=(
            "Local source root (or a parent containing repo-id). Required for v1.6/v2.0/v2.1 "
            "because legacy conversion runs on a disposable local copy; when omitted, only a v3.0 "
            "Hub dataset is loaded."
        ),
    )
    result.add_argument("--output-dir", required=True, help="New dataset directory; source is never modified.")
    result.add_argument("--output-repo-id", help="Metadata repo id for the new dataset.")
    result.add_argument(
        "--source-version",
        type=_parse_source_version,
        default="auto",
        help=(
            "Source generation (auto, v1.6, v2.0, v2.1, or v3.0; common aliases are accepted). "
            "Default: detect from local metadata/layout."
        ),
    )
    result.add_argument(
        "--converter-command",
        action="append",
        default=[],
        metavar="FROM=COMMAND",
        help=(
            "Command template for a legacy stage. Use placeholders {repo_id}, {source}, {output}, "
            "{from_version}, and {to_version}; repeat for every required stage."
        ),
    )
    task_source = result.add_mutually_exclusive_group()
    task_source.add_argument("--single-task", help="Task text for a v1.6 single-task dataset.")
    task_source.add_argument("--tasks-col", help="Instruction column for a v1.6 dataset.")
    task_source.add_argument("--tasks-path", help="Per-episode task JSON for a v1.6 dataset.")
    result.add_argument("--success-key", help="Explicit per-frame success feature.")
    result.add_argument(
        "--success-reduction",
        choices=["last", "any", "constant"],
        default="any",
        help="How a per-frame success feature becomes an episode label.",
    )
    result.add_argument(
        "--success-labels",
        help="Optional JSON mapping episode_index to a JSON bool or integer 0/1.",
    )
    result.add_argument(
        "--failure-penalty",
        type=float,
        help="Positive C_fail in raw control steps. Default: each task's maximum episode length.",
    )
    result.add_argument(
        "--step-key",
        default="frame_index",
        help="Integer step field used to verify episode ordering.",
    )
    result.add_argument(
        "--step-scale",
        type=float,
        default=1.0,
        help="Control-step cost per stored frame (for example frameskip); default 1.",
    )
    result.add_argument(
        "--normalization",
        choices=["task_max", "global_minmax", "none"],
        default="task_max",
        help=(
            "Return target scaling: task_max (the canonical per-task scaling), "
            "global_minmax (legacy H5-compatible global [-1,1] scaling), or none."
        ),
    )
    result.add_argument("--skip-recompute-stats", action="store_true")
    result.add_argument("--num-workers", type=int, default=0)
    return result


def as_scalar(value: Any, *, name: str = "value") -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}.")
    scalar = array.reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def column_names(dataset: Any) -> set[str]:
    table = getattr(dataset, "hf_dataset", None)
    return set(getattr(table, "column_names", ()))


def load_column(dataset: Any, key: str, *, scalar: bool = False) -> np.ndarray:
    table = getattr(dataset, "hf_dataset", None)
    values = list(table[key]) if table is not None and key in column_names(dataset) else [
        dataset[index][key] for index in range(len(dataset))
    ]
    if scalar:
        return np.asarray([as_scalar(value, name=key) for value in values])
    arrays = [np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value) for value in values]
    try:
        return np.stack(arrays)
    except ValueError:
        return np.asarray(arrays, dtype=object)


def load_task_indices(dataset: Any) -> np.ndarray:
    """Load task ids, deriving stable ids from a scalar task column if needed.

    ``task_index`` is part of the canonical v3 contract, but hand-authored
    v2/v3 mirrors and a few early exporters only retain the instruction text.
    Deriving ids from first-seen task strings is lossless for return scaling and
    makes the augmentation script useful for those datasets without guessing
    episode success or task boundaries.
    """

    available = set(dataset.features) | column_names(dataset)
    if "task_index" in available:
        return load_column(dataset, "task_index", scalar=True).astype(np.int64).reshape(-1)
    if "task" not in available:
        raise KeyError(
            "The dataset has neither 'task_index' nor a scalar 'task' feature; "
            "a task id is required for task-normalized returns."
        )
    raw_tasks = load_column(dataset, "task", scalar=True).reshape(-1)
    mapping: dict[str, int] = {}
    indices: list[int] = []
    for value in raw_tasks:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        text = str(value).strip()
        if not text:
            raise ValueError("The scalar 'task' feature contains an empty instruction.")
        if text not in mapping:
            mapping[text] = len(mapping)
        indices.append(mapping[text])
    return np.asarray(indices, dtype=np.int64)


def resolve_success_key(dataset: Any, explicit: str | None) -> str | None:
    available = set(dataset.features) | column_names(dataset)
    if explicit:
        if explicit not in available:
            raise KeyError(f"--success-key={explicit!r} is not present in the dataset.")
        return explicit
    for candidate in AUTO_SUCCESS_KEYS:
        if candidate in available:
            return candidate
    return None


def strict_bool(value: Any, *, source: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) in (0, 1):
        return bool(value)
    # A few parquet exporters materialize boolean flags as float32/float64.
    # Accept only exact finite 0.0/1.0 values; arbitrary non-zero floats must
    # still fail instead of being silently coerced to True.
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value) in (0.0, 1.0):
        return bool(value)
    raise ValueError(f"{source} must be a JSON bool or integer 0/1, got {value!r}.")


def load_success_labels(path: str | Path) -> dict[int, bool]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--success-labels must contain a JSON object mapping episode_index to bool.")
    result: dict[int, bool] = {}
    for key, value in payload.items():
        try:
            episode = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid episode_index key in success labels: {key!r}.") from exc
        result[episode] = strict_bool(value, source=f"success label for episode_index={episode}")
    return result


def episode_success(values: np.ndarray, reduction: str) -> bool:
    flat = np.asarray(values).reshape(-1)
    labels = np.asarray(
        [strict_bool(as_scalar(value, name="success label"), source="success label") for value in flat],
        dtype=bool,
    )
    if labels.size == 0:
        raise ValueError("Cannot reduce an empty success label sequence.")
    if reduction == "last":
        return bool(labels[-1])
    if reduction == "any":
        return bool(labels.any())
    if not np.all(labels == labels[0]):
        raise ValueError("success-reduction=constant but labels change within an episode.")
    return bool(labels[0])


def compute_pi06_returns(
    episode_index: np.ndarray,
    task_index: np.ndarray,
    success_by_episode: dict[int, bool],
    failure_penalty: float | None,
    normalize: bool | None = None,
    step_index: np.ndarray | None = None,
    step_scale: float = 1.0,
    normalization: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    # ``normalize`` was the original small public helper's boolean argument.
    # Keep it source-compatible for callers/tests while exposing the explicit
    # modes used by the command-line tool.  In particular, ``global_minmax``
    # reproduces the old RLDS->H5 writer's final [-1, 1] transform without
    # changing the canonical task-normalized default.
    if normalization is None:
        if normalize is None:
            normalization = "task_max"
        else:
            normalization = "task_max" if bool(normalize) else "none"
    elif normalization not in {"task_max", "global_minmax", "none"}:
        raise ValueError(
            "normalization must be one of 'task_max', 'global_minmax', or 'none'."
        )
    elif normalize is not None:
        implied = normalization == "task_max"
        if bool(normalize) != implied:
            raise ValueError(
                "normalize and normalization specify conflicting return scaling modes."
            )

    episode_index = np.asarray(episode_index).reshape(-1)
    task_index = np.asarray(task_index).reshape(-1)
    if len(episode_index) == 0:
        raise ValueError("Cannot compute returns for an empty dataset.")
    if len(episode_index) != len(task_index):
        raise ValueError("episode_index and task_index lengths differ.")
    if step_index is not None:
        step_index = np.asarray(step_index).reshape(-1)
        if len(step_index) != len(episode_index):
            raise ValueError("step_index and episode_index lengths differ.")
    if not np.isfinite(step_scale) or step_scale <= 0:
        raise ValueError("step_scale must be a finite positive number.")
    if failure_penalty is not None and (not np.isfinite(failure_penalty) or failure_penalty <= 0):
        raise ValueError("failure_penalty must be a finite positive number when supplied.")

    rows_by_episode: dict[int, list[int]] = defaultdict(list)
    task_by_episode: dict[int, int] = {}
    for row, (episode, task) in enumerate(zip(episode_index, task_index, strict=True)):
        episode = int(episode)
        task = int(task)
        rows_by_episode[episode].append(row)
        if episode in task_by_episode and task_by_episode[episode] != task:
            raise ValueError(f"episode_index={episode} contains multiple task_index values.")
        task_by_episode[episode] = task

    task_max_length: dict[int, int] = defaultdict(int)
    for episode, rows in rows_by_episode.items():
        if len(rows) > 1 and not np.all(np.diff(rows) == 1):
            raise ValueError(f"episode_index={episode} is not contiguous in the dataset table.")
        if step_index is not None:
            episode_steps = step_index[np.asarray(rows)]
            if len(episode_steps) > 1 and not np.all(np.diff(episode_steps) == 1):
                raise ValueError(
                    f"episode_index={episode} has non-consecutive step values {episode_steps[:10].tolist()}; "
                    "return alignment is ambiguous."
                )
        task = task_by_episode[episode]
        task_max_length[task] = max(task_max_length[task], len(rows))

    unknown_labels = set(map(int, success_by_episode)) - set(rows_by_episode)
    if unknown_labels:
        raise ValueError(f"Success labels reference absent episodes: {sorted(unknown_labels)[:10]}.")

    raw = np.empty((len(episode_index), 1), dtype=np.float32)
    normalized = np.empty_like(raw)
    for episode, rows in rows_by_episode.items():
        if episode not in success_by_episode:
            raise KeyError(f"Missing success label for episode_index={episode}.")
        success = strict_bool(success_by_episode[episode], source=f"success label for episode_index={episode}")
        task = task_by_episode[episode]
        length = len(rows)
        scaled_task_length = float(task_max_length[task]) * step_scale
        terminal_reward = 0.0 if success else -float(
            failure_penalty if failure_penalty is not None else scaled_task_length
        )
        # pi*0.6 Eq. 5: r_T=0 on success, -C_fail on failure, and -step_scale
        # otherwise. G_t includes r_t, so a successful length-3 episode is [-2,-1,0].
        returns = terminal_reward - step_scale * np.arange(length - 1, -1, -1, dtype=np.float32)
        row_array = np.asarray(rows)
        raw[row_array] = returns[:, None]
        if normalization == "task_max":
            normalized[row_array] = np.clip(returns / scaled_task_length, -1.0, 0.0)[:, None]
        else:
            normalized[row_array] = returns[:, None]

    if normalization == "global_minmax":
        # This is intentionally a dataset-level transform: it is the exact
        # convention used by the legacy H5 writer (success/failure values are
        # mapped together, rather than independently per task).  Keep the raw
        # control-step targets untouched in ``raw`` for auditability.
        minimum = float(np.min(raw))
        maximum = float(np.max(raw))
        if maximum - minimum < 1e-8:
            normalized.fill(0.0)
        else:
            normalized[...] = -1.0 + 2.0 * (raw - minimum) / (maximum - minimum)
    return raw, normalized, dict(task_max_length)


def validate_paths(source_root: Path | None, output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Output directory {output} already exists. Use a completely new path; add_features creates it."
        )
    if source_root is not None:
        source = source_root.resolve()
        if output == source:
            raise ValueError("Output directory must differ from the source dataset root.")
        if source in output.parents:
            raise ValueError("Output directory cannot be inside the source dataset directory.")
    return output


def validate_augmented_dataset(source: Any, verified: Any, expected: dict[str, np.ndarray]) -> None:
    if len(verified) != len(source):
        raise RuntimeError(f"Augmentation changed row count: {len(source)} -> {len(verified)}")
    for key in ("episode_index", "frame_index"):
        before = load_column(source, key, scalar=True).reshape(-1)
        after = load_column(verified, key, scalar=True).reshape(-1)
        np.testing.assert_array_equal(after, before)
    np.testing.assert_allclose(
        load_column(verified, "return", scalar=True).astype(np.float32),
        expected["return"],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        load_column(verified, "return_raw", scalar=True).astype(np.float32),
        expected["return_raw"],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        load_column(verified, "episode_success", scalar=True).astype(bool),
        expected["episode_success"],
    )


def main() -> None:
    args = parser().parse_args()
    if not np.isfinite(args.step_scale) or args.step_scale <= 0:
        raise ValueError("--step-scale must be a finite positive number.")
    if args.failure_penalty is not None and (
        not np.isfinite(args.failure_penalty) or args.failure_penalty <= 0
    ):
        raise ValueError("--failure-penalty must be a finite positive number.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    source_root: Path | None = None
    detected_root: Path | None = None
    source_version_for_metadata = TARGET_LEROBOT_VERSION
    source_version = TARGET_LEROBOT_VERSION
    conversion_records: list[dict[str, Any]] = []
    conversion_workspace: Path | None = None
    if args.root is not None:
        detected_root = dataset_root(args.root, args.repo_id)
        detected_version = detect_lerobot_version(detected_root)
        source_version_for_metadata = detected_version
        source_version = detected_version if args.source_version == "auto" else args.source_version
        if args.source_version != "auto" and source_version != detected_version:
            raise ValueError(
                f"--source-version={source_version} disagrees with detected version {detected_version}."
            )
        # Validate the user-visible destination before creating any disposable
        # conversion workspace.  In particular, reject an output nested inside
        # the source before copytree can recurse into its own destination.
        requested_output = validate_paths(detected_root, args.output_dir)
        if source_version == TARGET_LEROBOT_VERSION:
            source_root = detected_root
        else:
            commands = default_converter_commands()
            custom_commands = parse_converter_commands(args.converter_command)
            commands.update(custom_commands)
            if source_version == "v1.6":
                task_args = [
                    ("--single-task", args.single_task),
                    ("--tasks-col", args.tasks_col),
                    ("--tasks-path", args.tasks_path),
                ]
                selected = [(flag, value) for flag, value in task_args if value is not None]
                template = commands["v1.6"]
                embedded_task_flag = _template_contains_any_flag(
                    template,
                    tuple(flag for flag, _ in task_args),
                )
                # The built-in wrapper needs one task source.  A custom
                # command may carry its own task flag in the command template;
                # in that case requiring a duplicated CLI flag is surprising
                # and would produce malformed historical-converter invocations.
                if len(selected) > 1:
                    raise ValueError(
                        "LeRobot v1.6 conversion accepts at most one of --single-task, "
                        "--tasks-col, or --tasks-path."
                    )
                if selected and embedded_task_flag:
                    raise ValueError(
                        "The custom v1.6 converter command already contains a task-source flag; "
                        "do not also pass --single-task, --tasks-col, or --tasks-path."
                    )
                if len(selected) == 0 and not embedded_task_flag:
                    raise ValueError(
                        "LeRobot v1.6 conversion requires exactly one of "
                        "--single-task, --tasks-col, or --tasks-path, unless a custom "
                        "--converter-command already includes its own task flag."
                    )
                if len(selected) == 1 and not embedded_task_flag:
                    flag, value = selected[0]
                    if isinstance(template, list):
                        template.extend([flag, str(value)])
                    else:
                        template += f" {flag} {_shell_quote(str(value))}"
                    commands["v1.6"] = template
            source_root, conversion_records, conversion_workspace = migrate_to_v3(
                detected_root,
                source_version,
                args.repo_id,
                requested_output.parent,
                commands,
            )
            # Keep the disposable canonical copy alive while LeRobot reads it,
            # but guarantee cleanup both on success and on an uncaught error.
            atexit.register(_remove_conversion_workspace, conversion_workspace)
    elif args.source_version not in {"auto", TARGET_LEROBOT_VERSION}:
        raise ValueError("Legacy conversion requires --root so a disposable source copy can be created.")
    output_path = validate_paths(source_root, args.output_dir)

    try:
        from lerobot.datasets.dataset_tools import add_features, recompute_stats
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("Return augmentation requires lerobot>=0.5.1,<0.6 and Python>=3.12.") from exc

    try:
        source = LeRobotDataset(repo_id=args.repo_id, root=source_root)
    except Exception as exc:
        hint = (
            " The current tool accepts only canonical LeRobot v3.0. If this is a v1.6/v2.0/v2.1 Hub "
            "dataset, migrate and verify it on a copy or test branch first."
        ) if args.root is None else ""
        raise RuntimeError(f"Could not load source dataset {args.repo_id!r}.{hint}") from exc

    # When ``--root`` is omitted, LeRobot resolves a revision-safe Hub cache
    # directory internally.  Re-check against that concrete root now so an
    # output path accidentally pointing at the cache can never be modified in
    # place.
    output_path = validate_paths(source.root, args.output_dir)

    episode_index = load_column(source, "episode_index", scalar=True).astype(np.int64).reshape(-1)
    task_index = load_task_indices(source)
    step_index = load_column(source, args.step_key, scalar=True).astype(np.int64).reshape(-1)
    success_key = resolve_success_key(source, args.success_key)
    labels_from_file = load_success_labels(args.success_labels) if args.success_labels else {}

    success_by_episode: dict[int, bool] = {}
    if success_key is not None:
        success_values = load_column(source, success_key, scalar=True)
        for episode in np.unique(episode_index):
            rows = np.flatnonzero(episode_index == episode)
            success_by_episode[int(episode)] = episode_success(
                success_values[rows],
                args.success_reduction,
            )
    elif labels_from_file:
        success_by_episode = labels_from_file
    else:
        raise ValueError(
            "No trustworthy success label found. Pass --success-key or --success-labels; "
            "the script will not infer success from episode termination."
        )

    raw_returns, returns, task_max_length = compute_pi06_returns(
        episode_index,
        task_index,
        success_by_episode,
        args.failure_penalty,
        normalization=args.normalization,
        step_index=step_index,
        step_scale=args.step_scale,
    )
    for field in ("return", "return_raw"):
        if field in source.features or field in column_names(source):
            raise ValueError(
                f"Source already contains {field!r}. Choose a clean source dataset instead of overwriting it."
            )

    success_rows = np.asarray(
        [success_by_episode[int(ep)] for ep in episode_index], dtype=np.bool_
    )
    has_episode_success = "episode_success" in source.features or "episode_success" in column_names(source)
    if has_episode_success:
        existing_success = np.asarray(
            [
                strict_bool(value, source="episode_success")
                for value in load_column(source, "episode_success", scalar=True)
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(
            existing_success,
            success_rows,
            err_msg="Existing episode_success does not agree with the selected success labels.",
        )
    features = {
        "return": (returns[:, 0], {"dtype": "float32", "shape": (1,), "names": None}),
        "return_raw": (raw_returns[:, 0], {"dtype": "float32", "shape": (1,), "names": None}),
    }
    if not has_episode_success:
        # Keep this audit label numeric for compatibility with every v3
        # feature/Arrow backend; consumers can cast it to bool losslessly.
        features["episode_success"] = (
            success_rows.astype(np.int8),
            {"dtype": "int8", "shape": (1,), "names": None},
        )
    output_repo_id = args.output_repo_id or f"{args.repo_id}_with_return"
    augmented = add_features(
        dataset=source,
        features=features,
        output_dir=str(output_path),
        repo_id=output_repo_id,
    )
    if not args.skip_recompute_stats:
        recompute_stats(
            augmented,
            skip_image_video=True,
            chunk_size=50,
            num_workers=args.num_workers,
        )

    verified = LeRobotDataset(repo_id=output_repo_id, root=output_path)
    validate_augmented_dataset(
        source,
        verified,
        {
            "return": returns[:, 0],
            "return_raw": raw_returns[:, 0],
            "episode_success": success_rows,
        },
    )

    metadata = {
        "schema_version": 2,
        "source": {
            "repo_id": args.repo_id,
            "root": str(detected_root) if detected_root is not None else None,
            "detected_version": source_version_for_metadata,
        },
        "canonical_training_format": TARGET_LEROBOT_VERSION,
        "conversion_steps": conversion_records,
        "definition": {
            "nonterminal_reward": -float(args.step_scale),
            "success_terminal_reward": 0.0,
            "failure_terminal_reward": "-failure_penalty",
            "failure_penalty": (
                args.failure_penalty if args.failure_penalty is not None else "task_max_episode_length"
            ),
            "discount": 1.0,
            "normalization": args.normalization,
            "step_scale": args.step_scale,
        },
        "success_source": success_key or str(Path(args.success_labels).resolve()),
        "success_reduction": args.success_reduction if success_key is not None else "episode_json",
        "task_max_episode_length": {str(key): value for key, value in task_max_length.items()},
        "num_success_episodes": int(sum(success_by_episode.values())),
        "num_failure_episodes": int(len(success_by_episode) - sum(success_by_episode.values())),
    }
    if args.normalization == "global_minmax":
        # Persist the fitted affine transform so the target scale is fully
        # auditable/reproducible even if a downstream consumer only receives
        # the metadata and not the source table.
        metadata["definition"]["global_min"] = float(np.min(raw_returns))
        metadata["definition"]["global_max"] = float(np.max(raw_returns))
    metadata_path = output_path / "meta" / "return_definition.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    conversion_manifest = {
        "schema_version": 1,
        "source_version": source_version,
        "target_version": TARGET_LEROBOT_VERSION,
        "source_root": str(detected_root) if detected_root is not None else None,
        "canonical_root": str(source_root) if source_root is not None else None,
        "source_was_modified": False,
        "steps": conversion_records,
    }
    (metadata_path.parent / "conversion_manifest.json").write_text(
        json.dumps(conversion_manifest, indent=2), encoding="utf-8"
    )
    # Drop readers before deleting a legacy staging tree.  This matters on
    # Windows where memory-mapped parquet/video handles otherwise keep files
    # locked even after the final sample has been validated.
    del verified, augmented, source
    gc.collect()
    if conversion_workspace is not None:
        _remove_conversion_workspace(conversion_workspace)
        atexit.unregister(_remove_conversion_workspace)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
