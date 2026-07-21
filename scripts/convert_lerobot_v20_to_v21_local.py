from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Apply the official LeRobot v2.0->v2.1 stats migration to a local copy."
    )
    result.add_argument("--source", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--repo-id", default="converted_dataset")
    result.add_argument("--num-workers", type=int, default=4)
    return result


def _load_v21_components() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Load the v2.1 migration API from a matching historical LeRobot release."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
        from lerobot.datasets.v21.convert_stats import check_aggregate_stats, convert_stats

        return LeRobotDataset, EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info, (
            check_aggregate_stats,
            convert_stats,
        )
    except ImportError as first_error:
        raise ImportError(
            "The v2.0->v2.1 converter is not shipped by this LeRobot environment. "
            "Run this wrapper with a historical LeRobot release containing "
            "lerobot.datasets.v21.convert_stats (for example v0.3.x)."
        ) from first_error


def _call_supported(function: Any, **kwargs: Any) -> Any:
    """Call a historical helper while tolerating optional-argument drift."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name not in accepted
    ]
    if missing:
        raise TypeError(
            f"Unsupported historical helper signature {signature}; required arguments {missing} "
            "are not available in this wrapper."
        )
    return function(**accepted)


def _open_local_v20_dataset(dataset_cls: Any, repo_id: str, root: Path) -> Any:
    """Open the copied v2.0 tree without ever falling back to a Hub/cache root."""
    try:
        signature = inspect.signature(dataset_cls)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot inspect the historical LeRobotDataset constructor; refusing a potentially remote open."
        ) from exc
    parameters = signature.parameters
    if "root" not in parameters and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        raise RuntimeError(
            f"Historical LeRobotDataset constructor {signature} has no root= argument; "
            "use a release that can open a local v2.0 copy."
        )
    kwargs: dict[str, Any] = {"repo_id": repo_id, "root": root}
    if "revision" in parameters:
        kwargs["revision"] = "v2.0"
    return _call_supported(dataset_cls, **kwargs)


def main() -> None:
    args = parser().parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if source == output or source in output.parents:
        raise ValueError("The v2.1 output must be outside the v2.0 source tree.")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve/import the historical API before creating the disposable output
    # tree.  If the active environment lacks the converter, the wrapper should
    # fail without leaving a misleading half-copy behind.
    LeRobotDataset, EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info, stats_api = _load_v21_components()
    check_aggregate_stats, convert_stats = stats_api

    try:
        shutil.copytree(source, output, copy_function=shutil.copy2)

        required_metadata = (
            output / "meta" / "info.json",
            output / "meta" / "episodes.jsonl",
            output / "meta" / "tasks.jsonl",
            output / "meta" / "stats.json",
        )
        missing_metadata = [str(path) for path in required_metadata if not path.is_file()]
        if missing_metadata:
            raise ValueError(
                "The copied LeRobot v2.0 dataset is incomplete; refusing any Hub/cache fallback. "
                f"Missing metadata files: {missing_metadata}"
            )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    try:
        dataset = _open_local_v20_dataset(LeRobotDataset, args.repo_id, output)
    except Exception as exc:
        if output.exists():
            shutil.rmtree(output)
        raise RuntimeError(
            "Could not open the copied local v2.0 dataset. Use a historical LeRobot release whose "
            "LeRobotDataset supports the root= argument."
        ) from exc
    try:
        # The v2.0 input has aggregate stats only.  Remove any stale per-
        # episode file from a hand-authored copy before recomputing it.
        old_episode_stats = dataset.root / EPISODES_STATS_PATH
        if old_episode_stats.is_file():
            old_episode_stats.unlink()
        _call_supported(convert_stats, dataset=dataset, num_workers=args.num_workers)
        _call_supported(
            check_aggregate_stats,
            dataset=dataset,
            reference_stats=load_stats(dataset.root),
        )
        dataset.meta.info["codebase_version"] = "v2.1"
        write_info(dataset.meta.info, dataset.root)
        old_stats = dataset.root / STATS_PATH
        if old_stats.is_file():
            old_stats.unlink()
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise

    print(json.dumps({"source": str(source), "output": str(output), "version": "v2.1"}, indent=2))


if __name__ == "__main__":
    main()
