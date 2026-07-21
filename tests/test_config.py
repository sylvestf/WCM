import pytest

from world_critic.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
    _construct,
    apply_runtime_overrides,
    validate_train_config,
)


def test_history_must_fit_model_context():
    config = TrainConfig(data=DataConfig(repo_id="test/repo", history_size=5), model=ModelConfig(max_history=4))
    with pytest.raises(ValueError, match="max_history"):
        validate_train_config(config)


def test_fp16_is_rejected_without_scaler():
    config = TrainConfig(data=DataConfig(repo_id="test/repo"), precision="fp16")
    with pytest.raises(ValueError, match="fp16"):
        validate_train_config(config)


def test_legacy_return_weight_config_migrates_to_value_weight():
    config = _construct(LossConfig, {"return_weight": 2.5})
    assert config.value_weight == 2.5


def test_action_conditioned_dynamics_cannot_be_disabled():
    config = TrainConfig(data=DataConfig(repo_id="test/repo"), model=ModelConfig(dynamics_depth=0))
    with pytest.raises(ValueError, match="action-conditioned"):
        validate_train_config(config)


def test_value_head_cannot_be_disabled():
    config = TrainConfig(data=DataConfig(repo_id="test/repo"), loss=LossConfig(value_weight=0.0))
    with pytest.raises(ValueError, match="value_weight"):
        validate_train_config(config)


def test_state_vector_head_requires_a_loss_term():
    config = TrainConfig(
        data=DataConfig(repo_id="test/repo", state_key="observation.state"),
        model=ModelConfig(predict_state_vector=True, state_dim=3),
        loss=LossConfig(next_state_vector_weight=0.0),
    )
    with pytest.raises(ValueError, match="next_state_vector_weight"):
        validate_train_config(config)


def test_max_views_must_be_positive():
    config = TrainConfig(data=DataConfig(repo_id="test/repo"), model=ModelConfig(max_views=0))
    with pytest.raises(ValueError, match="max_views"):
        validate_train_config(config)


def test_at_least_one_camera_is_required():
    config = TrainConfig(data=DataConfig(repo_id="test/repo", image_keys=[]))
    with pytest.raises(ValueError, match="image_keys"):
        validate_train_config(config)


def test_launcher_runtime_overrides_cover_dataset_output_and_world_size(monkeypatch):
    config = TrainConfig(data=DataConfig(repo_id="yaml/repo"))
    monkeypatch.setenv("WCM_DATASET_REPO_ID", "launcher/repo")
    monkeypatch.setenv("WCM_DATASET_ROOT", "/datasets/lerobot")
    monkeypatch.setenv("WCM_OUTPUT_DIR", "/runs/wcm")
    monkeypatch.setenv("WCM_EXPECTED_WORLD_SIZE", "8")
    apply_runtime_overrides(config)
    assert config.data.repo_id == "launcher/repo"
    assert config.data.root == "/datasets/lerobot"
    assert config.output_dir == "/runs/wcm"
    assert config.expected_world_size == 8
