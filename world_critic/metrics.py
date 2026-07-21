from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .distributed import DistributedContext, all_reduce_sum


@dataclass
class RegressionMetrics:
    count: float = 0.0
    squared_error: float = 0.0
    absolute_error: float = 0.0
    sum_target: float = 0.0
    sum_prediction: float = 0.0
    sum_target2: float = 0.0
    sum_prediction2: float = 0.0
    sum_product: float = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        prediction = prediction.detach().double()
        target = target.detach().double()
        mask = mask.detach().bool()
        prediction = prediction[mask]
        target = target[mask]
        if target.numel() == 0:
            return
        error = prediction - target
        self.count += float(target.numel())
        self.squared_error += error.square().sum().item()
        self.absolute_error += error.abs().sum().item()
        self.sum_target += target.sum().item()
        self.sum_prediction += prediction.sum().item()
        self.sum_target2 += target.square().sum().item()
        self.sum_prediction2 += prediction.square().sum().item()
        self.sum_product += (target * prediction).sum().item()

    def reduce(self, ctx: DistributedContext) -> "RegressionMetrics":
        values = torch.tensor(
            [
                self.count,
                self.squared_error,
                self.absolute_error,
                self.sum_target,
                self.sum_prediction,
                self.sum_target2,
                self.sum_prediction2,
                self.sum_product,
            ],
            dtype=torch.float64,
            device=ctx.device,
        )
        all_reduce_sum(values, ctx)
        return RegressionMetrics(*values.cpu().tolist())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {"count": 0.0, "mse": math.nan, "rmse": math.nan, "mae": math.nan, "pearson": math.nan}
        mse = self.squared_error / self.count
        numerator = self.count * self.sum_product - self.sum_target * self.sum_prediction
        denominator = math.sqrt(
            max(self.count * self.sum_target2 - self.sum_target**2, 0.0)
            * max(self.count * self.sum_prediction2 - self.sum_prediction**2, 0.0)
        )
        pearson = numerator / denominator if denominator > 0 else math.nan
        return {
            "count": self.count,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self.absolute_error / self.count,
            "pearson": pearson,
        }


def masked_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction={prediction.shape}, target={target.shape}")
    if mask.shape != target.shape:
        mask = torch.broadcast_to(mask, target.shape)
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    mask = mask.bool() & finite
    squared_error = (prediction - target).square()
    return squared_error.masked_fill(~mask, 0.0).sum(), mask.sum()


def ddp_global_mean_loss(
    local_sum: torch.Tensor,
    local_count: torch.Tensor,
    ctx: DistributedContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a backward loss whose DDP-averaged gradient is a true global mean."""
    global_count = local_count.detach().to(dtype=torch.float64, device=local_sum.device)
    all_reduce_sum(global_count, ctx)
    if global_count.item() <= 0:
        return local_sum * 0.0, global_count
    backward_loss = local_sum * ctx.world_size / global_count.to(local_sum.dtype)
    return backward_loss, global_count
