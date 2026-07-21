from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass
class DataConfig:
    repo_id: str
    root: str | None = None
    revision: str | None = None
    image_keys: list[str] = field(default_factory=lambda: ["observation.images.front"])
    action_key: str = "action"
    state_key: str | None = "observation.state"
    return_key: str = "return"
    history_size: int = 3
    prediction_horizon: int = 1
    val_fraction: float = 0.1
    split_seed: int = 3072
    split_manifest: str | None = None
    # Kept as a compatibility field for old configs.  Value supervision is
    # mandatory in this critic implementation, so enabling it is rejected.
    allow_missing_return: bool = False
    normalize_action: bool = True
    action_mean: list[float] | None = None
    action_std: list[float] | None = None
    normalization_epsilon: float = 1e-6


@dataclass
class VisionConfig:
    model_name: str = "google/vit-base-patch16-224-in21k"
    image_size: int = 224
    trainable: bool = True
    pretrained: bool = True


@dataclass
class LanguageConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    max_length: int = 77
    trainable: bool = False
    pretrained: bool = True
    fusion_layers: int = 2
    fusion_heads: int = 8
    fusion_dropout: float = 0.0


@dataclass
class ModelConfig:
    action_dim: int | None = None
    state_dim: int | None = None
    latent_dim: int = 384
    max_views: int = 16
    trunk_depth: int = 6
    trunk_heads: int = 8
    trunk_mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_history: int = 16
    value_hidden_dim: int = 384
    dynamics_depth: int = 3
    action_hidden_dim: int = 384
    predict_state_vector: bool = False
    vision: VisionConfig = field(default_factory=VisionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)


@dataclass
class LossConfig:
    value_weight: float = 1.0
    next_state_weight: float = 0.1
    next_state_vector_weight: float = 0.0
    sigreg_weight: float = 0.01
    sigreg_knots: int = 17
    sigreg_num_projections: int = 1024


@dataclass
class OptimConfig:
    lr: float = 5e-5
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1


@dataclass
class TrainConfig:
    output_dir: str = "outputs/wcm_v2"
    seed: int = 3072
    epochs: int = 100
    per_device_batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 8
    # Variable valid-token counts and global SIGReg are normalized per microbatch.
    # Keep this at 1 until accumulation-window global-count normalization is implemented.
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    log_every: int = 20
    eval_every_epochs: int = 1
    save_every_epochs: int = 1
    resume: str | None = None
    compile: bool = False
    expected_world_size: int | None = None
    ddp_timeout_minutes: int = 30
    deterministic: bool = False
    data: DataConfig | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class EvalConfig:
    checkpoint: str
    output_dir: str = "outputs/wcm_v2_eval"
    batch_size: int = 64
    num_workers: int = 8
    precision: str = "bf16"
    expected_world_size: int | None = None
    max_batches: int | None = None


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    values = dict(values)
    if cls is TrainConfig:
        if "data" not in values:
            raise ValueError("Training config requires a 'data' section.")
        values["data"] = _construct(DataConfig, values["data"])
        values["model"] = _construct(ModelConfig, values.get("model", {}))
        values["loss"] = _construct(LossConfig, values.get("loss", {}))
        values["optim"] = _construct(OptimConfig, values.get("optim", {}))
    elif cls is ModelConfig:
        values["vision"] = _construct(VisionConfig, values.get("vision", {}))
        values["language"] = _construct(LanguageConfig, values.get("language", {}))
    elif cls is LossConfig and "return_weight" in values:
        if "value_weight" in values:
            raise ValueError("Specify only loss.value_weight; return_weight is a deprecated alias.")
        values["value_weight"] = values.pop("return_weight")
    elif cls is OptimConfig and "betas" in values:
        values["betas"] = tuple(values["betas"])
    return cls(**values)


def load_config(path: str | Path, cls: type[T] = TrainConfig) -> T:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML configs require PyYAML. Use JSON or install pyyaml.") from exc
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format: {path}")
    if not isinstance(values, dict):
        raise TypeError(f"Expected an object at the root of {path}")
    return _construct(cls, values)


def apply_runtime_overrides(config: TrainConfig) -> TrainConfig:
    """Apply small launcher-friendly overrides supplied through ``WCM_*`` env vars.

    The checked-in ``run_wcm.sh`` intentionally keeps all experiment knobs at
    the top of one shell file.  Keeping the override layer here avoids
    generating a temporary YAML file and preserves the exact same config
    validation/checkpoint schema as a direct ``--config`` invocation.  Empty
    variables are ignored, so users can leave a field at its YAML value.
    """

    if config.data is None:
        raise ValueError("Runtime overrides require a training config with data settings.")

    def value(name: str) -> str | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        raw = raw.strip()
        return raw if raw else None

    def integer(name: str) -> int | None:
        raw = value(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc

    repo_id = value("WCM_DATASET_REPO_ID")
    root = value("WCM_DATASET_ROOT")
    revision = value("WCM_DATASET_REVISION")
    output_dir = value("WCM_OUTPUT_DIR")
    expected_world_size = integer("WCM_EXPECTED_WORLD_SIZE")
    num_workers = integer("WCM_NUM_WORKERS")
    per_device_batch_size = integer("WCM_PER_DEVICE_BATCH_SIZE")
    eval_batch_size = integer("WCM_EVAL_BATCH_SIZE")
    epochs = integer("WCM_EPOCHS")
    resume = value("WCM_RESUME")
    precision = value("WCM_PRECISION")

    if repo_id is not None:
        config.data.repo_id = repo_id
    if root is not None:
        config.data.root = root
    if revision is not None:
        config.data.revision = revision
    if output_dir is not None:
        config.output_dir = output_dir
    if expected_world_size is not None:
        config.expected_world_size = expected_world_size
    if num_workers is not None:
        config.num_workers = num_workers
    if per_device_batch_size is not None:
        config.per_device_batch_size = per_device_batch_size
    if eval_batch_size is not None:
        config.eval_batch_size = eval_batch_size
    if epochs is not None:
        config.epochs = epochs
    if resume is not None:
        config.resume = resume
    if precision is not None:
        config.precision = precision
    return config


def save_resolved_config(config: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def config_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a JSON or YAML configuration file.")
    return parser


def validate_train_config(config: TrainConfig) -> None:
    if config.data is None:
        raise ValueError("Training config requires data settings.")
    if config.data.history_size < 1:
        raise ValueError("data.history_size must be positive.")
    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if config.eval_every_epochs < 1:
        raise ValueError("eval_every_epochs must be positive.")
    if config.save_every_epochs < 1:
        raise ValueError("save_every_epochs must be positive.")
    if config.log_every < 1:
        raise ValueError("log_every must be positive.")
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    if not config.data.image_keys:
        raise ValueError("data.image_keys must contain at least one camera feature.")
    if any(not str(key).strip() for key in config.data.image_keys):
        raise ValueError("data.image_keys cannot contain empty feature names.")
    if config.model.max_history < 1:
        raise ValueError("model.max_history must be positive.")
    if config.data.history_size > config.model.max_history:
        raise ValueError("data.history_size cannot exceed model.max_history.")
    if config.model.latent_dim < 1:
        raise ValueError("model.latent_dim must be positive.")
    if config.model.max_views < 1:
        raise ValueError("model.max_views must be positive.")
    if config.model.trunk_depth < 1:
        raise ValueError("model.trunk_depth must be positive.")
    if config.model.trunk_heads < 1:
        raise ValueError("model.trunk_heads must be positive.")
    if config.model.language.fusion_layers < 1:
        raise ValueError("model.language.fusion_layers must be positive.")
    if config.model.language.fusion_heads < 1:
        raise ValueError("model.language.fusion_heads must be positive.")
    if config.model.dynamics_depth < 1:
        raise ValueError("model.dynamics_depth must be positive so dynamics remains action-conditioned.")
    if not (0.0 <= config.data.val_fraction < 1.0):
        raise ValueError("data.val_fraction must be in [0, 1).")
    if config.data.normalization_epsilon <= 0:
        raise ValueError("data.normalization_epsilon must be positive.")
    if not config.data.return_key.strip():
        raise ValueError("data.return_key must name the supervised return field.")
    if config.data.allow_missing_return:
        raise ValueError("data.allow_missing_return is unsupported: value training requires a return field.")
    if (config.data.action_mean is None) != (config.data.action_std is None):
        raise ValueError("data.action_mean and data.action_std must either both be set or both be null.")
    if not config.data.normalize_action and config.data.action_mean is not None:
        raise ValueError("Action statistics must be null when normalize_action=false.")
    if config.model.latent_dim % config.model.trunk_heads != 0:
        raise ValueError("model.latent_dim must be divisible by model.trunk_heads.")
    if config.model.latent_dim % config.model.language.fusion_heads != 0:
        raise ValueError("model.latent_dim must be divisible by language.fusion_heads.")
    if config.model.predict_state_vector and config.data.state_key is None:
        raise ValueError("predict_state_vector requires data.state_key.")
    if config.loss.next_state_vector_weight > 0 and not config.model.predict_state_vector:
        raise ValueError("next_state_vector_weight>0 requires model.predict_state_vector=true.")
    if config.loss.sigreg_knots < 2:
        raise ValueError("loss.sigreg_knots must be at least 2.")
    if config.loss.sigreg_num_projections < 1:
        raise ValueError("loss.sigreg_num_projections must be positive.")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if config.per_device_batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if config.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16; fp16 is disabled without GradScaler support.")
    for name in (
        "value_weight",
        "next_state_weight",
        "next_state_vector_weight",
        "sigreg_weight",
    ):
        if getattr(config.loss, name) < 0:
            raise ValueError(f"loss.{name} cannot be negative.")
    if config.loss.value_weight <= 0:
        raise ValueError(
            "loss.value_weight must be positive: the requested World Critic always trains a value head."
        )
    if config.loss.next_state_weight <= 0:
        raise ValueError("next_state_weight must be positive to retain the requested world-model auxiliary task.")
    if config.model.predict_state_vector and config.loss.next_state_vector_weight <= 0:
        raise ValueError(
            "predict_state_vector=true requires a positive next_state_vector_weight; "
            "otherwise its parameters would be unused under DDP."
        )
    if config.expected_world_size is not None and config.expected_world_size < 1:
        raise ValueError("expected_world_size must be positive when set.")
    if config.ddp_timeout_minutes < 1:
        raise ValueError("ddp_timeout_minutes must be positive.")
