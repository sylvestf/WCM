import json
from pathlib import Path

import numpy as np
import pytest

from scripts.add_returns_to_lerobot import (
    compute_pi06_returns,
    dataset_root,
    detect_lerobot_version,
    legacy_migration_error,
    load_success_labels,
    load_task_indices,
    normalize_version,
    validate_paths,
)


def test_task_indices_can_be_derived_from_scalar_task_feature():
    class Dataset:
        features = {"task": {"dtype": "string"}}
        hf_dataset = None
        rows = [
            {"task": "place block"},
            {"task": "place block"},
            {"task": "stack block"},
        ]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    np.testing.assert_array_equal(load_task_indices(Dataset()), [0, 0, 1])


def test_pi06_returns_are_task_normalized_and_failure_separated():
    episodes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    tasks = np.array([7, 7, 7, 7, 7, 8, 8, 8, 8])
    raw, normalized, maxima = compute_pi06_returns(
        episodes,
        tasks,
        success_by_episode={0: True, 1: False, 2: True},
        failure_penalty=None,
        normalize=True,
        step_index=np.array([0, 1, 2, 0, 1, 0, 1, 2, 3]),
    )
    np.testing.assert_array_equal(raw[:3, 0], [-2, -1, 0])
    np.testing.assert_array_equal(raw[3:5, 0], [-4, -3])
    np.testing.assert_array_equal(raw[5:, 0], [-3, -2, -1, 0])
    assert maxima == {7: 3, 8: 4}
    np.testing.assert_allclose(normalized[:3, 0], [-2 / 3, -1 / 3, 0])
    np.testing.assert_array_equal(normalized[3:5, 0], [-1, -1])


def test_explicit_failure_penalty_and_step_scale_are_used():
    raw, normalized, _ = compute_pi06_returns(
        np.array([0, 0]),
        np.array([1, 1]),
        {0: False},
        failure_penalty=10,
        normalize=False,
        step_index=np.array([0, 1]),
        step_scale=2,
    )
    np.testing.assert_array_equal(raw[:, 0], [-12, -10])
    np.testing.assert_array_equal(normalized, raw)


def test_global_minmax_matches_legacy_h5_scaling_and_keeps_raw_targets():
    raw, scaled, _ = compute_pi06_returns(
        np.array([0, 0, 1, 1]),
        np.array([3, 3, 3, 3]),
        {0: True, 1: False},
        failure_penalty=3,
        normalization="global_minmax",
        step_index=np.array([0, 1, 0, 1]),
    )
    np.testing.assert_array_equal(raw[:, 0], [-1, 0, -4, -3])
    np.testing.assert_allclose(scaled[:, 0], [0.5, 1.0, -1.0, -0.5], atol=1e-6)


def test_normalize_compatibility_flag_rejects_conflicting_explicit_mode():
    with pytest.raises(ValueError, match="conflicting"):
        compute_pi06_returns(
            np.array([0]),
            np.array([1]),
            {0: True},
            failure_penalty=None,
            normalize=False,
            normalization="task_max",
        )


@pytest.mark.parametrize("failure_penalty", [0, -1, np.inf, np.nan])
def test_invalid_failure_penalty_is_rejected(failure_penalty):
    with pytest.raises(ValueError, match="failure_penalty"):
        compute_pi06_returns(
            np.array([0]),
            np.array([1]),
            {0: False},
            failure_penalty=failure_penalty,
            normalize=False,
        )


@pytest.mark.parametrize("step_scale", [0, -1, np.inf, np.nan])
def test_invalid_step_scale_is_rejected(step_scale):
    with pytest.raises(ValueError, match="step_scale"):
        compute_pi06_returns(
            np.array([0]),
            np.array([1]),
            {0: True},
            failure_penalty=None,
            normalize=False,
            step_scale=step_scale,
        )


def test_nonconsecutive_steps_fail_fast():
    with pytest.raises(ValueError, match="non-consecutive"):
        compute_pi06_returns(
            np.array([0, 0]),
            np.array([1, 1]),
            {0: True},
            failure_penalty=None,
            normalize=True,
            step_index=np.array([0, 2]),
        )


def test_success_json_accepts_only_bool_or_zero_one(tmp_path: Path):
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"0": True, "1": 0, "2": 1}), encoding="utf-8")
    assert load_success_labels(valid) == {0: True, 1: False, 2: True}

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"0": "false"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON bool"):
        load_success_labels(invalid)


def test_float_success_flags_accept_only_exact_binary_values():
    from scripts.add_returns_to_lerobot import strict_bool

    assert strict_bool(0.0, source="flag") is False
    assert strict_bool(1.0, source="flag") is True
    with pytest.raises(ValueError):
        strict_bool(0.5, source="flag")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1.6", "v1.6"), ("v2", "v2.0"), ("2.1", "v2.1"), ("v3", "v3.0")],
)
def test_version_normalization(value, expected):
    assert normalize_version(value) == expected


def test_detects_v3_layout_without_declared_version(tmp_path: Path):
    (tmp_path / "meta" / "episodes").mkdir(parents=True)
    (tmp_path / "meta" / "tasks.parquet").touch()
    assert detect_lerobot_version(tmp_path) == "v3.0"


@pytest.mark.parametrize(
    ("with_stats", "expected"),
    [(False, "v2.0"), (True, "v2.1")],
)
def test_detects_v2_layouts(tmp_path: Path, with_stats, expected):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "episodes.jsonl").touch()
    (tmp_path / "meta" / "tasks.jsonl").touch()
    if with_stats:
        (tmp_path / "meta" / "episodes_stats.jsonl").touch()
    assert detect_lerobot_version(tmp_path) == expected


def test_detects_v16_layout(tmp_path: Path):
    (tmp_path / "meta_data").mkdir()
    assert detect_lerobot_version(tmp_path) == "v1.6"


@pytest.mark.parametrize(
    ("version", "module"),
    [
        ("v1.6", "convert_dataset_v1_to_v2"),
        ("v2.0", "convert_dataset_v20_to_v21"),
        ("v2.1", "convert_dataset_v21_to_v30"),
    ],
)
def test_legacy_failure_explains_official_safe_migration(version, module):
    message = str(legacy_migration_error(version, "user/data", "/datasets/user/data"))
    assert module in message
    assert "copy" in message
    assert "test branch" in message
    assert "v3.0" in message


def test_dataset_root_accepts_parent_or_direct_root(tmp_path: Path):
    nested = tmp_path / "user" / "data"
    (nested / "meta").mkdir(parents=True)
    (nested / "meta" / "info.json").write_text('{"codebase_version":"v3.0"}', encoding="utf-8")
    assert dataset_root(tmp_path, "user/data") == nested.resolve()
    assert dataset_root(nested, "user/data") == nested.resolve()


def test_output_must_not_exist_or_be_inside_source(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        validate_paths(source, existing)
    with pytest.raises(ValueError, match="inside"):
        validate_paths(source, source / "augmented")
