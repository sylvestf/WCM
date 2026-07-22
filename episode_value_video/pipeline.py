from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_existing_evaluation(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "val",
    batch_size: int = 64,
    num_workers: int = 8,
    nproc_per_node: int = 1,
    expected_world_size: int | None = None,
    max_batches: int | None = None,
    max_curve_episodes: int | None = None,
    log_every_batches: int = 20,
    dataset_root: str | Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
) -> Path:
    """Run the unchanged WCM evaluator and return its curve JSON path."""

    if nproc_per_node < 1:
        raise ValueError("nproc_per_node must be positive.")
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    eval_args = [
        "-m",
        "world_critic.evaluate",
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_path),
        "--split",
        split,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--episode-curves",
        "--log-every-batches",
        str(log_every_batches),
    ]
    if expected_world_size is not None:
        eval_args.extend(["--expected-world-size", str(expected_world_size)])
    if max_batches is not None:
        eval_args.extend(["--max-batches", str(max_batches)])
    if max_curve_episodes is not None:
        eval_args.extend(["--max-curve-episodes", str(max_curve_episodes)])

    if nproc_per_node == 1:
        command = [sys.executable, *eval_args]
    else:
        torchrun = shutil.which("torchrun")
        if torchrun is None:
            raise FileNotFoundError(
                "nproc_per_node > 1 requires the torchrun executable from the WCM environment."
            )
        command = [
            torchrun,
            "--standalone",
            f"--nproc-per-node={nproc_per_node}",
            *eval_args,
        ]

    environment = os.environ.copy()
    if dataset_root is not None:
        environment["WCM_DATASET_ROOT"] = str(Path(dataset_root).expanduser().resolve())
    if repo_id is not None:
        environment["WCM_DATASET_REPO_ID"] = repo_id
    if revision is not None:
        environment["WCM_DATASET_REVISION"] = revision
    print("[value-video] running existing evaluator:", " ".join(command), flush=True)
    try:
        subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Existing WCM evaluation failed with exit code {exc.returncode}.") from exc

    curve_path = output_path / "episode_curves" / "episode_curves.json"
    if not curve_path.is_file():
        raise FileNotFoundError(
            f"Evaluation completed but did not create the expected curve artifact: {curve_path}"
        )
    return curve_path


def evaluation_settings_from_args(args: Any) -> dict[str, Any]:
    return {
        "checkpoint": args.checkpoint,
        "output_dir": args.eval_output_dir,
        "split": args.split,
        "batch_size": args.eval_batch_size,
        "num_workers": args.eval_num_workers,
        "nproc_per_node": args.nproc_per_node,
        "expected_world_size": args.expected_world_size,
        "max_batches": args.max_batches,
        "max_curve_episodes": args.max_eval_curve_episodes,
        "log_every_batches": args.log_every_batches,
        "dataset_root": args.dataset_root,
        "repo_id": args.repo_id,
        "revision": args.revision,
    }
