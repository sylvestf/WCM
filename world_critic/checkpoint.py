from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .config import TrainConfig
from .distributed import DistributedContext, broadcast_object, gather_objects


CHECKPOINT_SCHEMA_VERSION = 1


def collectively_validate(
    ctx: DistributedContext,
    operation: str,
    function,
) -> None:
    """Make rank-local validation failures visible before the next collective."""
    error = None
    try:
        function()
    except Exception as exc:
        error = repr(exc)
    statuses = gather_objects({"rank": ctx.rank, "error": error}, ctx, dst=0)
    result = None
    if ctx.is_main:
        if statuses is None or len(statuses) != ctx.world_size:
            failures = [{"rank": ctx.rank, "error": "collective status gather incomplete"}]
        else:
            failures = [status for status in statuses if status["error"] is not None]
        result = {"ok": not failures, "failures": failures}
    result = broadcast_object(result, ctx, src=0)
    if not result["ok"]:
        raise RuntimeError(f"{operation} failed: {result['failures']}")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while True:
        if hasattr(model, "module"):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        return model


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainConfig,
    epoch: int,
    global_step: int,
    best_metric: float,
    ctx: DistributedContext,
) -> None:
    # Capture can theoretically fail (for example a CUDA context error).  Do
    # not let one rank raise before the gather, because every other rank would
    # then block forever waiting in the checkpoint collective.
    rng_error = None
    try:
        rng_state = capture_rng_state()
    except Exception as exc:
        rng_state = None
        rng_error = repr(exc)
    rng_by_rank = gather_objects({"state": rng_state, "error": rng_error}, ctx, dst=0)
    result: dict[str, Any] | None = None
    if ctx.is_main:
        try:
            if rng_by_rank is None or len(rng_by_rank) != ctx.world_size:
                raise RuntimeError(
                    "Checkpoint RNG gather returned an incomplete rank list: "
                    f"expected {ctx.world_size}, got {None if rng_by_rank is None else len(rng_by_rank)}."
                )
            rng_failures = []
            for rank, item in enumerate(rng_by_rank):
                if not isinstance(item, dict):
                    rng_failures.append({"rank": rank, "error": "invalid RNG gather payload"})
                elif item.get("error") is not None:
                    rng_failures.append({"rank": rank, "error": item.get("error")})
            if rng_failures:
                raise RuntimeError(f"RNG capture failed on one or more ranks: {rng_failures}")
            payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "artifact_type": "full_resume",
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": best_metric,
                "config": asdict(config),
                "rng_by_rank": [item["state"] for item in rng_by_rank],
            }
            atomic_torch_save(payload, path)
            result = {"ok": True}
        except Exception as exc:  # all ranks must observe the failure instead of hanging
            result = {"ok": False, "error": repr(exc)}
    result = broadcast_object(result, ctx, src=0)
    if not result["ok"]:
        raise RuntimeError(f"Checkpoint save failed on rank 0: {result['error']}")


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    ctx: DistributedContext,
    expected_config: TrainConfig | None = None,
) -> dict[str, Any]:
    checkpoint = load_checkpoint_payload(path, ctx)
    validation_error = None
    try:
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema in {path}")
        if checkpoint.get("artifact_type") != "full_resume":
            raise ValueError(f"Checkpoint {path} is not a full-resume artifact.")
        if expected_config is not None:
            saved = checkpoint["config"]
            current = asdict(expected_config)
            immutable_paths = [
                ("data", "repo_id"),
                ("data", "root"),
                ("data", "revision"),
                ("data", "image_keys"),
                ("data", "action_key"),
                ("data", "state_key"),
                ("data", "return_key"),
                ("data", "history_size"),
                ("data", "prediction_horizon"),
                ("data", "val_fraction"),
                ("data", "split_seed"),
                ("data", "split_manifest"),
                ("data", "allow_missing_return"),
                ("data", "normalize_action"),
                ("data", "normalization_epsilon"),
                ("model",),
                ("loss",),
                ("optim", "lr"),
                ("optim", "weight_decay"),
                ("optim", "betas"),
                ("optim", "warmup_steps"),
                ("optim", "min_lr_ratio"),
                ("per_device_batch_size",),
                ("gradient_accumulation_steps",),
                ("precision",),
                ("epochs",),
            ]

            def select(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
                value: Any = mapping
                for key in keys:
                    value = value[key]
                return value

            mismatches = {
                ".".join(keys): (select(saved, keys), select(current, keys))
                for keys in immutable_paths
                if select(saved, keys) != select(current, keys)
            }
            if mismatches:
                raise ValueError(f"Resume config is incompatible with checkpoint: {mismatches}")
        rng_by_rank = checkpoint.get("rng_by_rank")
        if not isinstance(rng_by_rank, (list, tuple)) or len(rng_by_rank) != ctx.world_size:
            raise ValueError(
                "Checkpoint RNG state is missing or has the wrong world size: "
                f"expected {ctx.world_size}, got {None if rng_by_rank is None else len(rng_by_rank)}."
            )
    except Exception as exc:
        validation_error = repr(exc)

    statuses = gather_objects({"rank": ctx.rank, "error": validation_error}, ctx, dst=0)
    validation_result = None
    if ctx.is_main:
        if statuses is None or len(statuses) != ctx.world_size:
            failures = [{"rank": ctx.rank, "error": "checkpoint validation gather incomplete"}]
        else:
            failures = [status for status in statuses if status["error"] is not None]
        validation_result = {"ok": not failures, "failures": failures}
    validation_result = broadcast_object(validation_result, ctx, src=0)
    if not validation_result["ok"]:
        raise RuntimeError(f"Checkpoint validation failed: {validation_result['failures']}")
    if ctx.distributed:
        reference = torch.tensor([checkpoint["global_step"]], dtype=torch.int64, device=ctx.device)
        minimum = reference.clone()
        maximum = reference.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if minimum.item() != maximum.item():
            raise RuntimeError("Ranks loaded different checkpoint global_step values.")
    def restore_state() -> None:
        unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

    collectively_validate(ctx, "Checkpoint state restoration", restore_state)
    rng_by_rank = checkpoint.get("rng_by_rank")
    if rng_by_rank is not None:
        restore_rng_state(rng_by_rank[ctx.rank])
    return checkpoint


def save_deploy_bundle(
    path: str | Path,
    model: torch.nn.Module,
    config: TrainConfig,
    ctx: DistributedContext,
) -> None:
    result: dict[str, Any] | None = None
    if ctx.is_main:
        try:
            atomic_torch_save(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "artifact_type": "deploy",
                    "model": unwrap_model(model).state_dict(),
                    "config": asdict(config),
                },
                path,
            )
            result = {"ok": True}
        except Exception as exc:
            result = {"ok": False, "error": repr(exc)}
    result = broadcast_object(result, ctx, src=0)
    if not result["ok"]:
        raise RuntimeError(f"Deploy bundle save failed on rank 0: {result['error']}")


def load_checkpoint_payload(path: str | Path, ctx: DistributedContext) -> dict[str, Any]:
    """Collectively load an artifact and make every rank fail consistently.

    Each rank loads locally so CUDA/NCCL training does not broadcast a multi-GB Python object over the
    process group. Use deploy artifacts for evaluation to avoid loading optimizer state.
    """
    payload = None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        local_status = {"rank": ctx.rank, "ok": True}
    except Exception as exc:
        local_status = {"rank": ctx.rank, "ok": False, "error": repr(exc)}
    statuses = gather_objects(local_status, ctx, dst=0)
    result = None
    if ctx.is_main:
        if statuses is None or len(statuses) != ctx.world_size:
            failures = [{"rank": ctx.rank, "ok": False, "error": "checkpoint status gather incomplete"}]
        else:
            failures = [status for status in statuses if not status["ok"]]
        result = {"ok": not failures, "failures": failures}
    result = broadcast_object(result, ctx, src=0)
    if not result["ok"]:
        raise RuntimeError(f"Checkpoint load failed: {result['failures']}")
    return payload


def inspect_checkpoint_config(
    path: str | Path,
    ctx: DistributedContext | None = None,
) -> dict[str, Any]:
    result = None
    if ctx is None or ctx.is_main:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(f"Unsupported checkpoint schema in {path}")
            if "config" not in payload:
                raise KeyError(f"Checkpoint {path} has no resolved config.")
            result = {"ok": True, "config": payload["config"]}
        except Exception as exc:
            result = {"ok": False, "error": repr(exc)}
    if ctx is not None:
        result = broadcast_object(result, ctx, src=0)
    if not result["ok"]:
        raise RuntimeError(f"Could not inspect checkpoint config: {result['error']}")
    return result["config"]
