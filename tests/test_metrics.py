import torch

from world_critic.metrics import RegressionMetrics, masked_squared_error
from world_critic.data import canonicalize_return_target


def test_masked_squared_error_has_no_broadcast_surprise():
    prediction = torch.tensor([[[1.0], [3.0]]])
    target = torch.tensor([[[2.0], [1.0]]])
    error_sum, count = masked_squared_error(
        prediction,
        target,
        torch.tensor([[[True], [False]]]),
    )
    assert error_sum.item() == 1.0
    assert count.item() == 1


def test_regression_metrics():
    metric = RegressionMetrics()
    prediction = torch.tensor([1.0, 2.0, 4.0])
    target = torch.tensor([1.0, 3.0, 5.0])
    metric.update(prediction, target, torch.ones(3, dtype=torch.bool))
    result = metric.compute()
    assert result["mse"] == 2 / 3
    assert result["mae"] == 2 / 3


def test_canonicalize_scalar_lerobot_return():
    value = canonicalize_return_target(torch.zeros(2, 3))
    assert value.shape == (2, 3, 1)


def test_global_sigreg_ddp_scaling_algebra():
    """Document the all-gather/DDP compensation used by the training path.

    The differentiable all-gather backward sums the W replicated scalar-loss
    gradients, while DDP averages parameter gradients by W.  Their product is
    one, so the global SIGReg loss must not be divided by world size.
    """
    world_size = 8
    all_gather_backward_scale = world_size
    ddp_gradient_average_scale = 1.0 / world_size
    assert all_gather_backward_scale * ddp_gradient_average_scale == 1.0
