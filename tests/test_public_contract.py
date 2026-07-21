from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import tomllib

from world_critic.config import DataConfig, LossConfig
from world_critic.model import LatentRolloutOutput, WorldCriticOutput


ROOT = Path(__file__).resolve().parents[1]


def test_value_return_naming_contract_is_unambiguous() -> None:
    """The public API must distinguish predictions (value) from targets (return)."""
    assert DataConfig(repo_id="user/data").return_key == "return"
    assert {field.name for field in fields(LossConfig)} >= {"value_weight"}
    assert {field.name for field in fields(WorldCriticOutput)} >= {"value", "next_state_pred"}
    assert [field.name for field in fields(LatentRolloutOutput)] == ["latents", "values"]
    assert not any(field.name.startswith("return_") for field in fields(WorldCriticOutput))
    assert not any(field.name.startswith("return_") for field in fields(LatentRolloutOutput))


def test_eight_gpu_launchers_and_config_are_explicit() -> None:
    train = (ROOT / "run_train_8gpu.sh").read_text(encoding="utf-8")
    evaluate = (ROOT / "run_eval_8gpu.sh").read_text(encoding="utf-8")
    config = (ROOT / "configs" / "train_8gpu.yaml").read_text(encoding="utf-8")
    for launcher in (train, evaluate):
        assert "--nproc-per-node=8" in launcher
        assert "WCM_ALLOW_CPU_DDP" in launcher
    assert "expected_world_size: 8" in config
    assert "return_key: return" in config


def test_single_file_launcher_exposes_editable_train_eval_knobs() -> None:
    launcher = (ROOT / "run_wcm.sh").read_text(encoding="utf-8")
    for variable in (
        'MODE="train"',
        "GPUS=1",
        "DATASET_REPO_ID=",
        "DATASET_ROOT=",
        "OUTPUT_DIR=",
        "CHECKPOINT=",
        "EVAL_SPLIT=",
        "PLOT_EPISODE_CURVES=",
    ):
        assert variable in launcher
    assert "torchrun --standalone --nproc-per-node=8" in launcher
    assert "python -m world_critic.train" in launcher
    assert "python -m world_critic.evaluate" in launcher

    train = (ROOT / "run_train.sh").read_text(encoding="utf-8")
    evaluate = (ROOT / "run_eval.sh").read_text(encoding="utf-8")
    for script in (train, evaluate):
        assert "GPUS=8" in script
        assert "DATASET_REPO_ID=" in script
        assert "DATASET_ROOT=" in script
        assert "ALLOW_CPU_SMOKE" in script
        assert "torchrun --standalone --nproc-per-node=8" in script
    assert "PLOT_EPISODE_CURVES=" in evaluate
    assert "--episode-curves" in evaluate


def test_install_extra_and_huggingface_dependency_generation_are_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = project["dependencies"]
    extras = project["optional-dependencies"]
    assert "transformers>=5,<6" in dependencies
    assert "huggingface-hub>=1,<2" in dependencies
    assert "all" in extras


def test_training_prepare_phase_uses_collective_error_propagation() -> None:
    source = (ROOT / "world_critic" / "train.py").read_text(encoding="utf-8")
    assert "collectively_validate(ctx, \"Output directory/config preparation\"" in source
    assert "collectively_validate(ctx, \"Episode split preparation\"" in source
    assert "collectively_validate(ctx, \"Training action-statistics preparation\"" in source
    assert "collectively_validate(ctx, \"Dataset/schema preparation\"" in source


def test_evaluation_summary_write_is_collectively_guarded() -> None:
    source = (ROOT / "world_critic" / "evaluate.py").read_text(encoding="utf-8")
    assert "collectively_validate(ctx, \"Evaluation summary write\"" in source


def test_episode_curve_uses_last_valid_token_and_frame_alignment() -> None:
    source = (ROOT / "world_critic" / "training.py").read_text(encoding="utf-8")
    curves = (ROOT / "world_critic" / "curves.py").read_text(encoding="utf-8")
    assert "last_index" in source
    assert "endpoint_frames" in source
    assert "endpoint_values = output.value[row_index, last_index]" in source
    assert "endpoint_metrics.reduce(ctx)" in source
    assert "token_metrics.reduce(ctx)" in source
    assert "build_episode_curves(records)" in source
    assert "token_value_" in source
    assert "next_state_mse" in source
    assert "token_next_state_mse" in source
    assert "Duplicate episode/frame" in curves


def test_training_validation_log_write_is_collectively_guarded() -> None:
    source = (ROOT / "world_critic" / "train.py").read_text(encoding="utf-8")
    assert "collectively_validate(ctx, \"Validation metrics write\"" in source


def test_worker_processor_is_picklable_for_spawn_workers() -> None:
    source = (ROOT / "world_critic" / "data.py").read_text(encoding="utf-8")
    assert "class CombinedProcessor" in source
    assert "return CombinedProcessor(" in source
    assert "class CombinedProcessor" in source.split("def build_processor", 1)[0]
