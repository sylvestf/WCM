from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import CHECKPOINT_SCHEMA_VERSION, collectively_validate, load_checkpoint_payload
from .data import (
    LeRobotWorldCriticDataset,
    WorldCriticCollator,
    build_processor,
    load_episode_split,
    load_lerobot_dataset,
    validate_action_normalization,
)
from .distributed import (
    DistributedEvalSampler,
    barrier,
    cleanup_distributed,
    initialize_distributed,
)
from .curves import write_episode_curve_artifacts
from .model import WorldCriticModel
from .training import config_from_checkpoint_payload, evaluate_loader, seed_everything
from .config import apply_runtime_overrides, validate_train_config


def _json_safe(value):
    """Keep summaries strict-JSON even when Pearson is undefined."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Distributed offline value/dynamics evaluation.")
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--split", choices=["train", "val", "all"], default="val")
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--expected-world-size", type=int)
    result.add_argument(
        "--max-batches",
        type=int,
        help="Per-rank debug limit; do not compare this mode across different world sizes.",
    )
    result.add_argument(
        "--episode-curves",
        "--plot-episode-curves",
        dest="episode_curves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write one value-vs-return curve per episode. Each overlapping window contributes "
            "only its last valid timestep, sorted by frame_index (default: enabled)."
        ),
    )
    result.add_argument(
        "--curve-output-dir",
        help="Optional directory for episode_curves.json/CSV and PNG plots (defaults under --output-dir).",
    )
    result.add_argument(
        "--max-curve-episodes",
        type=int,
        help="Optional limit on the number of sorted episode plots written; metrics still use the full split.",
    )
    result.add_argument(
        "--log-every-batches",
        type=int,
        default=20,
        help="Print an unbuffered fetch/forward timing line every N batches (default: 20).",
    )
    return result


def run() -> None:
    args = parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive when provided.")
    if args.max_curve_episodes is not None and args.max_curve_episodes < 1:
        raise ValueError("--max-curve-episodes must be positive when provided.")
    if args.log_every_batches < 1:
        raise ValueError("--log-every-batches must be positive.")
    ctx = initialize_distributed(args.expected_world_size)
    run_started = time.monotonic()

    def log(message: str, *, main_only: bool = False) -> None:
        if main_only and not ctx.is_main:
            return
        print(
            f"[eval][+{time.monotonic() - run_started:8.1f}s]"
            f"[rank={ctx.rank}][device={ctx.device}] {message}",
            flush=True,
        )

    try:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else None
        log(
            f"starting: cwd={Path.cwd()}, checkpoint={checkpoint_path}, "
            f"checkpoint_bytes={checkpoint_size}, output_dir={Path(args.output_dir).resolve()}, "
            f"split={args.split}, batch_size={args.batch_size}, num_workers={args.num_workers}, "
            f"max_batches={args.max_batches}, episode_curves={args.episode_curves}",
        )
        log("loading checkpoint payload...")
        payload = load_checkpoint_payload(args.checkpoint, ctx)
        log(
            "checkpoint payload loaded: "
            f"artifact_type={payload.get('artifact_type')!r}, "
            f"keys={sorted(payload.keys())}, epoch={payload.get('epoch')!r}, "
            f"global_step={payload.get('global_step')!r}",
        )

        def validate_artifact() -> None:
            if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("Unsupported checkpoint schema.")
            if payload.get("artifact_type") not in {"full_resume", "deploy"}:
                raise ValueError(f"Unsupported artifact type: {payload.get('artifact_type')!r}")
            if "model" not in payload or "config" not in payload:
                raise KeyError("Checkpoint artifact is missing model or config.")

        log("validating checkpoint metadata...")
        collectively_validate(ctx, "Evaluation checkpoint metadata validation", validate_artifact)
        log("checkpoint metadata validation complete")
        if ctx.is_main and payload.get("artifact_type") == "full_resume":
            log("WARNING: full-resume checkpoint; deploy.pt uses substantially less CPU memory.", main_only=True)
        train_config = None

        def build_and_validate_config() -> None:
            nonlocal train_config
            train_config = apply_runtime_overrides(config_from_checkpoint_payload(payload))
            validate_train_config(train_config)

        log("validating checkpoint config...")
        collectively_validate(ctx, "Evaluation config validation", build_and_validate_config)
        log(
            "config validation complete: "
            f"dataset_repo={train_config.data.repo_id!r}, dataset_root={train_config.data.root!r}, "
            f"vision={train_config.model.vision.model_name!r}, "
            f"language={train_config.model.language.model_name!r}",
        )
        seed_everything(train_config.seed, train_config.deterministic)

        dataset = None

        def load_and_validate_dataset() -> None:
            nonlocal dataset
            dataset = load_lerobot_dataset(train_config.data)
            validate_action_normalization(dataset, train_config.data)

        log("loading and validating LeRobot dataset...")
        collectively_validate(ctx, "Evaluation dataset validation", load_and_validate_dataset)
        log(f"dataset load complete: rows={len(dataset)}")
        eval_dataset = None
        loader = None
        model = None

        # Every operation below can fail locally (missing split manifest,
        # malformed rows, unavailable HF cache, or a model-construction
        # mismatch).  Keep it inside one collective guard so one rank cannot
        # proceed into metric all-reduces while another rank has already
        # exited with an exception.
        def prepare_eval_components() -> None:
            nonlocal eval_dataset, loader, model
            log("prepare: resolving episode split and temporal windows...")
            checkpoint_dir = Path(args.checkpoint).resolve().parent
            candidates = []
            if train_config.data.split_manifest:
                configured_manifest = Path(train_config.data.split_manifest).expanduser()
                candidates.append(configured_manifest)
                if not configured_manifest.is_absolute():
                    candidates.extend(
                        [
                            checkpoint_dir / configured_manifest,
                            checkpoint_dir.parent / configured_manifest,
                        ]
                    )
            candidates.extend([checkpoint_dir.parent / "episode_split.json", checkpoint_dir / "episode_split.json"])
            manifest = next((path for path in candidates if path.exists()), None)
            if args.split == "all":
                episode_ids = None
            else:
                if manifest is None:
                    raise FileNotFoundError("Could not find episode_split.json next to the checkpoint.")
                split = load_episode_split(manifest)
                episode_ids = split.val if args.split == "val" else split.train
            log(
                f"prepare: manifest={str(manifest) if manifest is not None else None}, "
                f"episode_count={None if episode_ids is None else len(episode_ids)}"
            )
            eval_dataset = LeRobotWorldCriticDataset(dataset, train_config.data, episode_ids)
            if len(eval_dataset) == 0:
                raise ValueError(f"The selected {args.split!r} split contains no valid temporal windows.")
            log(f"prepare: temporal windows={len(eval_dataset)}")
            log("prepare: loading image processor and tokenizer...")
            processor = build_processor(train_config.model)
            log("prepare: processor/tokenizer ready")
            collator = WorldCriticCollator(
                processor,
                train_config.model.vision.image_size,
                train_config.model.language.max_length,
            )
            sampler = DistributedEvalSampler(eval_dataset, ctx.rank, ctx.world_size) if ctx.distributed else None
            loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                pin_memory=ctx.device.type == "cuda",
                collate_fn=collator,
            )
            log(
                f"prepare: DataLoader ready, batches={len(loader)}, "
                f"batch_size={args.batch_size}, workers={args.num_workers}"
            )
            log("prepare: constructing WorldCriticModel (HF Loading weights may appear next)...")
            model = WorldCriticModel(train_config.model)
            log("prepare: WorldCriticModel construction complete")

        log("preparing evaluation dataset, loader, and model...")
        collectively_validate(ctx, "Evaluation dataset/loader/model preparation", prepare_eval_components)
        log("evaluation components prepared")

        def load_model_weights() -> None:
            log("loading checkpoint model state_dict into WorldCriticModel...")
            model.load_state_dict(payload["model"], strict=True)
            log("checkpoint model state_dict loaded")

        collectively_validate(
            ctx,
            "Strict evaluation checkpoint load",
            load_model_weights,
        )

        def prepare_evaluation_model() -> None:
            log("moving model to evaluation device and disabling gradients...")
            model.to(ctx.device).eval().requires_grad_(False)
            first_parameter = next(model.parameters(), None)
            log(
                "model device preparation complete: "
                f"parameter_device={None if first_parameter is None else first_parameter.device}"
            )

        log("preparing model device...")
        collectively_validate(ctx, "Evaluation model device preparation", prepare_evaluation_model)
        log(
            f"starting evaluate_loader: max_batches={args.max_batches}, "
            f"log_every_batches={args.log_every_batches}"
        )
        metrics = evaluate_loader(
            model,
            loader,
            train_config,
            ctx,
            args.max_batches,
            collect_episode_curves=args.episode_curves,
            log_every_batches=args.log_every_batches,
        )
        log(f"evaluate_loader complete: metric_keys={sorted(metrics.keys())}")
        curve_records = metrics.pop("episode_curves", None)
        output_dir = Path(args.output_dir)
        curve_summary = None
        curve_dir = Path(args.curve_output_dir) if args.curve_output_dir else output_dir / "episode_curves"

        def write_episode_curves() -> None:
            nonlocal curve_summary
            if not ctx.is_main or not args.episode_curves:
                return
            if curve_records is None:
                raise RuntimeError("Rank 0 did not receive episode curve records.")
            records_to_write = curve_records
            if args.max_curve_episodes is not None:
                records_to_write = curve_records[: args.max_curve_episodes]
            curve_summary = write_episode_curve_artifacts(
                records_to_write,
                curve_dir,
                render_plots=True,
            )
            curve_summary["total_num_episodes"] = len(curve_records)
            curve_summary["max_episodes"] = args.max_curve_episodes
            Path(curve_summary["summary"]).write_text(
                json.dumps(_json_safe(curve_summary), indent=2), encoding="utf-8"
            )

        log("writing episode curve artifacts..." if args.episode_curves else "episode curve writing disabled")
        collectively_validate(ctx, "Evaluation episode curve write", write_episode_curves)
        log("episode curve stage complete")
        result = _json_safe({
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "split": args.split,
            "world_size": ctx.world_size,
            "num_windows": len(eval_dataset),
            "metrics": metrics,
            "episode_curves": curve_summary,
        })

        # Rank 0 is the only writer, but its filesystem operation still needs
        # to be collectively checked.  Otherwise a read-only/full output
        # volume would make rank 0 fail while the other ranks wait forever at
        # the following barrier.
        def write_summary() -> None:
            if ctx.is_main:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "summary.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )

        log("writing evaluation summary...")
        collectively_validate(ctx, "Evaluation summary write", write_summary)
        log("evaluation summary write complete")
        if ctx.is_main:
            print(json.dumps(result, indent=2), flush=True)
        barrier(ctx)
    finally:
        log("cleaning up distributed context")
        cleanup_distributed(ctx)


if __name__ == "__main__":
    run()
