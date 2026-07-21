from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    expected_world_size: int | None = None,
    timeout_minutes: int = 30,
) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size < 1:
        raise RuntimeError(f"WORLD_SIZE must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"RANK={rank} is outside WORLD_SIZE={world_size}.")
    if local_rank < 0:
        raise RuntimeError(f"LOCAL_RANK must be non-negative, got {local_rank}.")
    if timeout_minutes <= 0:
        raise ValueError(f"timeout_minutes must be positive, got {timeout_minutes}.")

    if expected_world_size is not None and world_size != expected_world_size:
        raise RuntimeError(
            f"Expected WORLD_SIZE={expected_world_size}, but torchrun provided {world_size}."
        )

    # ``WCM_FORCE_CPU=1`` is useful for an intentional Gloo smoke test on a
    # workstation that has one or more GPUs but not the full eight requested
    # by the production launcher.  Without this override, merely setting
    # ``WCM_ALLOW_CPU_DDP`` would still select NCCL whenever *any* CUDA device
    # is visible, and ranks whose LOCAL_RANK exceeds that device count would
    # fail before the process group could be formed.
    explicit_force_cpu = os.environ.get("WCM_FORCE_CPU", "0") == "1"
    allow_cpu_ddp = os.environ.get("WCM_ALLOW_CPU_DDP", "0") == "1" or explicit_force_cpu
    # The public smoke-test switch means "use Gloo/CPU", even on a machine
    # where CUDA happens to be installed.  Keeping this interpretation here
    # makes direct ``torchrun`` invocations behave like the checked-in
    # launchers; ``WCM_FORCE_CPU`` remains a more explicit synonym for scripts.
    force_cpu = explicit_force_cpu or allow_cpu_ddp
    visible_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if world_size > 1 and visible_cuda == 0 and not allow_cpu_ddp:
        raise RuntimeError(
            "A multi-process CPU/Gloo run is disabled by default. Set "
            "WCM_ALLOW_CPU_DDP=1 (or use the launcher smoke-test mode) explicitly."
        )
    if not force_cpu and visible_cuda > 0 and world_size > visible_cuda:
        # This check is intentionally rank-independent.  If only the ranks
        # whose LOCAL_RANK is out of range raised before process-group init,
        # rank 0 could enter init_process_group and wait forever for them.
        raise RuntimeError(
            f"WORLD_SIZE={world_size} exceeds the {visible_cuda} visible CUDA devices. "
            "Reduce --nproc-per-node or expose enough GPUs."
        )
    if torch.cuda.is_available() and not force_cpu:
        if local_rank >= visible_cuda:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {visible_cuda} CUDA devices are visible."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
    return DistributedContext(rank, local_rank, world_size, device)


def barrier(ctx: DistributedContext) -> None:
    if ctx.distributed:
        dist.barrier()


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(tensor: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    if ctx.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_object(value: Any, ctx: DistributedContext, src: int = 0) -> Any:
    if not ctx.distributed:
        return value
    if src < 0 or src >= ctx.world_size:
        raise ValueError(f"broadcast src={src} outside world_size={ctx.world_size}.")
    values = [value if ctx.rank == src else None]
    dist.broadcast_object_list(values, src=src)
    return values[0]


def gather_objects(value: Any, ctx: DistributedContext, dst: int = 0) -> list[Any] | None:
    if not ctx.distributed:
        return [value]
    if dst < 0 or dst >= ctx.world_size:
        raise ValueError(f"gather dst={dst} outside world_size={ctx.world_size}.")
    output = [None for _ in range(ctx.world_size)] if ctx.rank == dst else None
    dist.gather_object(value, object_gather_list=output, dst=dst)
    return output


class DistributedEvalSampler(torch.utils.data.Sampler[int]):
    """Shard evaluation indices without padding or duplication."""

    def __init__(self, dataset: torch.utils.data.Dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size
