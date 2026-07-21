from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Any


def _is_v3_root(path: Path) -> bool:
    """Return whether ``path`` has the canonical v3 metadata layout."""
    meta = path / "meta"
    return meta.joinpath("episodes").is_dir() and (
        meta.joinpath("tasks.parquet").is_file() or meta.joinpath("tasks").is_dir()
    )


def _find_converted_v3_root(staging_parent: Path, preferred: Path) -> Path:
    """Locate the converter's output despite historical root-layout changes.

    The v2.1->v3 converter has changed whether it mutates the supplied root,
    nests the repository below ``root``, or emits a sibling directory.  Since
    this wrapper always operates on a disposable staging tree, searching that
    tree is safer than assuming one API-specific path.  Prefer the expected
    working root, then inspect shallow descendants and reject ambiguous output
    instead of silently selecting the wrong dataset.
    """
    candidates: list[Path] = []
    if _is_v3_root(preferred):
        candidates.append(preferred)
    # The converter never needs more than a few levels for its parent/repo
    # layouts.  Avoid following arbitrary deep data directories (videos,
    # parquet chunks) and keep the search deterministic.
    for candidate in sorted(staging_parent.rglob("meta")):
        if len(candidate.relative_to(staging_parent).parts) > 4:
            continue
        root = candidate.parent
        if _is_v3_root(root) and root not in candidates:
            candidates.append(root)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "Official v2.1->v3 converter completed but no canonical v3 dataset "
            f"was found below disposable staging root {staging_parent}."
        )
    raise RuntimeError(
        "Official v2.1->v3 converter produced multiple possible v3 roots: "
        + ", ".join(str(path) for path in candidates)
        + ". Refusing to choose an output implicitly."
    )


def _root_mode_from_converter_source(source: str) -> str:
    """Infer whether a historical converter expects ``root`` or its parent.

    LeRobot 0.4.x and 0.5.0 use the same import location
    (``lerobot.datasets.v30.convert_dataset_v21_to_v30``), but changed the
    meaning of ``root`` in 0.5.0.  Module-path based dispatch therefore is not
    sufficient.  Keep this small parser deliberately conservative: only the
    explicit ``Path(root) / repo_id`` pattern selects the parent mode; unknown
    implementations default to exact-root mode and are still validated by the
    output-layout checks below.
    """
    compact = "".join(source.split())
    if "Path(root)/repo_id" in compact or "Path(root)/local_name" in compact:
        return "parent"
    if "root=Path(root)/repo_id" in compact:
        return "parent"
    return "exact"


def _converter_root_mode(module: Any) -> str:
    """Return ``parent`` or ``exact`` for a historical converter module."""
    converter = getattr(module, "convert_dataset", None)
    if converter is None:
        converter = getattr(module, "convert_dataset_v21_to_v30", None)
    if converter is None:
        raise AttributeError(
            f"{module.__name__} does not expose convert_dataset or "
            "convert_dataset_v21_to_v30."
        )
    try:
        source = inspect.getsource(converter)
    except (OSError, TypeError):
        source = ""
    mode = _root_mode_from_converter_source(source)
    if source:
        return mode

    # Some frozen/compiled environments do not retain source.  Package
    # versions are only a fallback because downstream forks can change the
    # implementation without changing the version string.
    try:
        from importlib.metadata import version

        package_version = tuple(int(part) for part in version("lerobot").split(".")[:2])
    except Exception:
        package_version = None
    if package_version is not None and package_version < (0, 5):
        return "parent"
    return "exact"


def _invoke_converter(module: Any, *, repo_id: str, root: Path) -> None:
    """Call historical converter APIs while tolerating signature drift."""
    converter = getattr(module, "convert_dataset", None)
    if converter is None:
        converter = getattr(module, "convert_dataset_v21_to_v30", None)
    if converter is None:
        raise AttributeError(
            f"{module.__name__} does not expose convert_dataset or "
            "convert_dataset_v21_to_v30."
        )
    values = {
        "repo_id": repo_id,
        "root": root,
        "push_to_hub": False,
        "force_conversion": True,
    }
    try:
        signature = inspect.signature(converter)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        converter(**values)
        return
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        converter(**values)
        return
    accepted = {
        name: value
        for name, value in values.items()
        if name in signature.parameters
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in accepted
    ]
    if missing:
        raise TypeError(
            f"Unsupported historical converter signature {signature}; required arguments {missing} "
            "are not among the known v2.1->v3 arguments."
        )
    converter(**accepted)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the official LeRobot v2.1->v3.0 converter on a copy.")
    result.add_argument("--source", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--repo-id", required=True)
    return result


def _load_official_converter() -> tuple[Any, str]:
    """Load the converter across the v0.4/v0.5 package relocation.

    LeRobot v0.4.x/v0.5.0 expose ``datasets.v30`` and expect ``root`` to be a
    parent directory containing ``repo_id``.  LeRobot v0.5.1 moved the script to
    ``scripts`` and expects ``root`` to be the exact dataset directory.
    """
    try:
        import lerobot.datasets.v30.convert_dataset_v21_to_v30 as module

        return module, _converter_root_mode(module)
    except ImportError:
        try:
            import lerobot.scripts.convert_dataset_v21_to_v30 as module

            return module, "exact"
        except ImportError as exc:
            raise ImportError(
                "The v2.1->v3 converter is not available in this LeRobot environment. "
                "Use a historical LeRobot v0.4.x/v0.5.0 environment for "
                "lerobot.datasets.v30.convert_dataset_v21_to_v30, or v0.5.1+ for "
                "lerobot.scripts.convert_dataset_v21_to_v30."
            ) from exc


def main() -> None:
    args = parser().parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(output)
    if source == output or source in output.parents:
        raise ValueError("The v3.0 output must be outside the v2.1 source tree.")
    output.parent.mkdir(parents=True, exist_ok=True)

    converter_module, root_mode = _load_official_converter()
    staging_parent = output.parent / f".{output.name}_official_v30"
    staging_parent.mkdir(parents=True, exist_ok=False)

    # Copy before calling the official converter.  Both historical variants
    # rewrite/rename their input root, but only inside this disposable staging
    # directory; the caller's source tree is never touched.
    local_name = args.repo_id.replace("/", "__")
    if root_mode == "parent":
        work_root = staging_parent / local_name
        converter_root = staging_parent
    else:
        work_root = staging_parent / "dataset"
        converter_root = work_root
    try:
        shutil.copytree(source, work_root, copy_function=shutil.copy2)

        required_metadata = (
            work_root / "meta" / "info.json",
            work_root / "meta" / "episodes.jsonl",
            work_root / "meta" / "tasks.jsonl",
            work_root / "meta" / "episodes_stats.jsonl",
        )
        missing_metadata = [str(path) for path in required_metadata if not path.is_file()]
        if missing_metadata:
            raise ValueError(
                "The copied LeRobot v2.1 dataset is incomplete; refusing any Hub/cache fallback. "
                f"Missing metadata files: {missing_metadata}"
            )
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise

    try:
        _invoke_converter(converter_module, repo_id=local_name, root=converter_root)
        converted_root = _find_converted_v3_root(staging_parent, work_root)
        # ``shutil.move`` requires a non-existing destination and preserves the
        # converter's complete metadata/data tree.  If the historical API
        # emitted a nested repository root, move that root rather than the
        # original staging directory.
        shutil.move(str(converted_root), str(output))
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise
    finally:
        if staging_parent.exists():
            # Some parquet/video backends keep a short-lived read handle on
            # Windows.  Cleanup is best-effort here: the caller's converted
            # output has already been moved out of staging, and the outer
            # orchestrator also retries cleanup at process exit.
            shutil.rmtree(staging_parent, ignore_errors=True)

    print(json.dumps({"source": str(source), "output": str(output), "version": "v3.0"}, indent=2))


if __name__ == "__main__":
    main()
