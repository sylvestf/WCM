from __future__ import annotations

import argparse
import json

import torch

from .config import load_config, validate_train_config
from .data import (
    LeRobotWorldCriticDataset,
    WorldCriticCollator,
    build_episode_split,
    build_processor,
    episode_ids_from_dataset,
    fit_action_normalization,
    load_lerobot_dataset,
)


def run() -> None:
    parser = argparse.ArgumentParser(description="Inspect one canonical LeRobot World Critic window.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_train_config(config)
    dataset = load_lerobot_dataset(config.data)
    if config.data.normalize_action and config.data.action_mean is None:
        split = build_episode_split(
            episode_ids_from_dataset(dataset),
            config.data.val_fraction,
            config.data.split_seed,
        )
        fit_action_normalization(dataset, config.data, split.train)
    windows = LeRobotWorldCriticDataset(dataset, config.data)
    if len(windows) == 0:
        raise ValueError("Dataset contains no valid windows.")
    collator = WorldCriticCollator(
        build_processor(config.model),
        config.model.vision.image_size,
        config.model.language.max_length,
    )
    batch = collator([windows[0]])
    summary = {
        key: list(value.shape) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key != "images"
    }
    summary["images"] = list(batch["images"].shape)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
