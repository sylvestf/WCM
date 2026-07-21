import pytest

torch = pytest.importorskip("torch")

from world_critic.config import LanguageConfig, ModelConfig, VisionConfig
from world_critic.model import WorldCriticModel


class FakeVisionEncoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.proj = torch.nn.Linear(3, latent_dim)

    def forward(self, images):
        return self.proj(images.mean(dim=(-1, -2)))


class FakeLanguageEncoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, latent_dim)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids), attention_mask.bool()


def make_model():
    config = ModelConfig(
        action_dim=2,
        latent_dim=16,
        trunk_depth=1,
        trunk_heads=4,
        trunk_mlp_ratio=2.0,
        dropout=0.0,
        max_history=4,
        value_hidden_dim=16,
        dynamics_depth=1,
        action_hidden_dim=16,
        vision=VisionConfig(pretrained=False),
        language=LanguageConfig(pretrained=False, fusion_layers=1, fusion_heads=4),
    )
    model = WorldCriticModel.__new__(WorldCriticModel)
    torch.nn.Module.__init__(model)
    model.config = config
    model.vision_encoder = FakeVisionEncoder(16)
    model.language_encoder = FakeLanguageEncoder(16)
    from world_critic.model import ActionConditionedDynamics, ActionFreeContextTrunk, MLP, StateLanguageFusion

    model.view_pool_query = torch.nn.Parameter(torch.zeros(1, 1, 16))
    model.view_attention = torch.nn.MultiheadAttention(16, 4, batch_first=True)
    model.language_fusion = StateLanguageFusion(config)
    model.context_trunk = ActionFreeContextTrunk(config)
    model.value_head = MLP(16, 16, 1)
    model.dynamics = ActionConditionedDynamics(config)
    model.state_vector_head = None
    return model.eval()


def test_value_is_action_invariant_but_dynamics_is_conditioned():
    torch.manual_seed(0)
    model = make_model()
    images = torch.randn(2, 4, 1, 3, 8, 8)
    actions_a = torch.randn(2, 3, 2, requires_grad=True)
    actions_b = actions_a.detach() + 3.0
    ids = torch.randint(0, 32, (2, 5))
    text_mask = torch.ones_like(ids)
    valid = torch.ones(2, 3, dtype=torch.bool)

    output_a = model(images, actions_a, ids, text_mask, valid)
    output_b = model(images, actions_b, ids, text_mask, valid)
    torch.testing.assert_close(output_a.value, output_b.value, rtol=0, atol=0)
    value_gradient = torch.autograd.grad(
        output_a.value.sum(), actions_a, allow_unused=True, retain_graph=True
    )[0]
    assert value_gradient is None
    assert not torch.equal(output_a.next_state_pred, output_b.next_state_pred)
    gradient = torch.autograd.grad(output_a.next_state_pred.sum(), actions_a)[0]
    assert gradient is not None and gradient.abs().sum() > 0


def test_forward_rejects_inconsistent_batch_shapes():
    model = make_model()
    images = torch.randn(2, 3, 1, 3, 8, 8)
    actions = torch.randn(1, 2, 2)
    ids = torch.randint(0, 32, (2, 5))
    mask = torch.ones_like(ids)
    with pytest.raises(ValueError, match="batch sizes differ"):
        model(images, actions, ids, mask)


def test_rollout_uses_trained_dynamics_branch():
    torch.manual_seed(1)
    model = make_model()
    images = torch.randn(2, 3, 1, 3, 8, 8)
    actions = torch.randn(2, 2, 2)
    ids = torch.randint(0, 32, (2, 5))
    text_mask = torch.ones_like(ids)
    rollout = model.rollout_latent(images, actions, ids, text_mask)
    assert rollout.latents.shape == (2, 5, 16)
    assert rollout.values.shape == (2, 2, 1)


def test_dynamics_is_anchored_to_current_visual_latent():
    torch.manual_seed(2)
    model = make_model()
    context = torch.randn(2, 3, 16)
    actions = torch.randn(2, 3, 2)
    current = torch.randn(2, 3, 16)
    offset = torch.randn(1, 1, 16)

    prediction = model.dynamics(current, context, actions)
    shifted_prediction = model.dynamics(current + offset, context, actions)

    torch.testing.assert_close(shifted_prediction - prediction, offset.expand_as(prediction))


def test_dynamics_rejects_non_sequence_inputs():
    model = make_model()
    with pytest.raises(ValueError, match="expects current_state_latent/context/actions"):
        model.dynamics(
            torch.randn(2, 16),
            torch.randn(2, 16),
            torch.randn(2, 2),
        )


def test_forward_and_one_step_rollout_share_identical_dynamics():
    torch.manual_seed(3)
    model = make_model()
    images = torch.randn(2, 4, 1, 3, 8, 8)
    actions = torch.randn(2, 3, 2)
    ids = torch.randint(0, 32, (2, 5))
    text_mask = torch.ones_like(ids)
    valid = torch.ones(2, 3, dtype=torch.bool)

    forward = model(images, actions, ids, text_mask, valid)
    rollout = model.rollout_latent(images[:, :-1], actions[:, -1:], ids, text_mask)

    torch.testing.assert_close(rollout.latents[:, -1:], forward.next_state_pred[:, -1:])
    torch.testing.assert_close(rollout.values, forward.value[:, -1:])
