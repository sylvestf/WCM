import json

import pytest

from scripts.add_returns_to_lerobot import (
    conversion_plan,
    detect_lerobot_version,
    default_converter_commands,
    normalize_version,
    parse_converter_commands,
    parser,
    resolve_success_key,
)
from scripts.convert_lerobot_v20_to_v21_local import _call_supported
from scripts.convert_lerobot_v21_to_v30_local import _find_converted_v3_root, _invoke_converter
from scripts.add_returns_to_lerobot import _format_command_template, _template_contains_any_flag
from scripts.convert_lerobot_v16_to_v20_local import _find_v1_video, _infer_features
from scripts.convert_lerobot_v21_to_v30_local import _root_mode_from_converter_source


def test_all_supported_generations_plan_to_v3():
    assert conversion_plan("v1.6") == [("v1.6", "v2.0"), ("v2.0", "v2.1"), ("v2.1", "v3.0")]
    assert conversion_plan("v2.0") == [("v2.0", "v2.1"), ("v2.1", "v3.0")]
    assert conversion_plan("v2.1") == [("v2.1", "v3.0")]
    assert conversion_plan("v3.0") == []
    assert normalize_version("v1") == "v1.6"


def test_detects_metadata_versions_and_v3_layout(tmp_path):
    v20 = tmp_path / "v20"
    (v20 / "meta").mkdir(parents=True)
    (v20 / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.0"}))
    assert detect_lerobot_version(v20) == "v2.0"

    v30 = tmp_path / "v30"
    (v30 / "meta" / "episodes").mkdir(parents=True)
    (v30 / "meta" / "tasks.parquet").touch()
    assert detect_lerobot_version(v30) == "v3.0"


def test_layout_markers_override_stale_declared_version(tmp_path):
    """A converter may retain the input info.json while writing a new layout."""

    root = tmp_path / "converted"
    (root / "meta" / "episodes").mkdir(parents=True)
    (root / "meta" / "tasks.parquet").touch()
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}))
    assert detect_lerobot_version(root) == "v3.0"


def test_detects_v16_metadata_only_snapshot(tmp_path):
    root = tmp_path / "v16"
    (root / "meta_data").mkdir(parents=True)
    (root / "meta_data" / "info.json").write_text(json.dumps({"fps": 30}))
    assert detect_lerobot_version(root) == "v1.6"


def test_rejects_unknown_declared_schema_version(tmp_path):
    root = tmp_path / "unknown"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v4.0"}))
    with pytest.raises(ValueError, match="Unsupported LeRobot codebase_version"):
        detect_lerobot_version(root)


def test_converter_command_map_requires_stage_prefix():
    commands = parse_converter_commands(["v1.6=python converter.py --input {source} --output {output}"])
    assert commands["v1.6"].startswith("python converter.py")
    with pytest.raises(ValueError, match="FROM=COMMAND"):
        parse_converter_commands(["python converter.py"])


def test_default_converter_chain_covers_every_legacy_stage():
    commands = default_converter_commands()
    assert set(commands) == {"v1.6", "v2.0", "v2.1"}
    assert all(isinstance(command, list) for command in commands.values())


@pytest.mark.parametrize(
    "key",
    ["next.success", "next.is_success", "is_episode_successful", "episode.success", "episode_success", "is_success", "success"],
)
def test_success_key_auto_detection_covers_common_exporters(key):
    class Dataset:
        features = {key: {"dtype": "bool"}}
        hf_dataset = None

    assert resolve_success_key(Dataset(), None) == key


def test_source_version_parser_accepts_common_aliases():
    args = parser().parse_args(["--repo-id", "user/data", "--output-dir", "out", "--source-version", "2.1"])
    assert args.source_version == "v2.1"


def test_v16_video_lookup_supports_nested_mirror_layout(tmp_path):
    """v1.6 mirrors may retain the nested videos*/.../... layout."""
    nested = tmp_path / "videos-000" / "camera" / "chunk-000"
    nested.mkdir(parents=True)
    expected = nested / "observation.images.front_episode_000007.mp4"
    expected.write_bytes(b"placeholder")
    assert _find_v1_video(tmp_path, expected.name) == expected


def test_v21_converter_invocation_and_nested_output_discovery(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    nested = staging / "repo" / "repo_v30"
    calls = []

    class FakeModule:
        __name__ = "fake_converter"

        @staticmethod
        def convert_dataset(repo_id, root, push_to_hub=False, force_conversion=False):
            calls.append((repo_id, root, push_to_hub, force_conversion))
            (nested / "meta" / "episodes").mkdir(parents=True)
            (nested / "meta" / "tasks.parquet").touch()

    _invoke_converter(FakeModule, repo_id="repo", root=staging)
    assert calls == [("repo", staging, False, True)]
    assert _find_converted_v3_root(staging, staging / "repo") == nested


def test_historical_helper_signature_filtering_is_strict():
    assert _call_supported(lambda dataset, num_workers=0: (dataset, num_workers), dataset="d", num_workers=3) == (
        "d",
        3,
    )
    with pytest.raises(TypeError, match="required arguments"):
        _call_supported(lambda dataset, required: None, dataset="d")


def test_custom_v16_flag_detection_is_token_exact():
    assert _template_contains_any_flag("python convert.py --single-task 'pick'", ("--single-task",))
    assert not _template_contains_any_flag("python convert.py --single-task-name foo", ("--single-task",))


def test_converter_template_preserves_braces_in_instruction_text():
    assert _format_command_template(
        "python convert.py --single-task 'pick {cube}' --source {source}",
        {"source": "/tmp/source", "repo_id": "repo", "output": "/tmp/out", "from_version": "v1.6", "to_version": "v2.0"},
    ) == "python convert.py --single-task 'pick {cube}' --source /tmp/source"


def test_v21_to_v30_root_mode_handles_04_and_050_api_semantics():
    # 0.4.x uses ``Path(root) / repo_id`` (root is a parent directory), while
    # 0.5.0/0.5.1 use the exact dataset directory passed as root.
    assert _root_mode_from_converter_source("root = Path(root) / repo_id") == "parent"
    assert _root_mode_from_converter_source("root = Path(root)") == "exact"
    assert _root_mode_from_converter_source("root = HF_HOME / repo_id") == "exact"


def test_v16_feature_inference_handles_variable_length_sequences():
    """Old datasets exports may omit Sequence.length; infer it from row zero."""

    class Value:
        def __init__(self, dtype):
            self.dtype = dtype

    class Sequence:
        def __init__(self, feature, length=None):
            self.feature = feature
            self.length = length

    class FakeDataset:
        features = {"observation.state": Sequence(Value("float32"), length=None)}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {"observation.state": [0.0, 1.0, 2.0, 3.0]}

    class Official:
        datasets = type("Datasets", (), {"Value": Value, "Sequence": Sequence})

    features = _infer_features(Official, FakeDataset())
    assert features["observation.state"]["dtype"] == "float32"
    assert features["observation.state"]["shape"] == (4,)
    assert features["observation.state"]["names"] == {
        "motors": ["motor_0", "motor_1", "motor_2", "motor_3"]
    }


def test_v16_feature_inference_treats_minus_one_sequence_length_as_unknown():
    class Value:
        dtype = "float32"

    class Sequence:
        feature = Value()
        length = -1  # datasets.Sequence's historical variable-length sentinel

    class FakeDataset:
        features = {"state": Sequence()}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"state": [0.0, 1.0, 2.0]}

    class Official:
        datasets = type("Datasets", (), {"Value": Value, "Sequence": Sequence})

    assert _infer_features(Official, FakeDataset())["state"]["shape"] == (3,)


def test_v16_feature_inference_accepts_feature_class_names_without_public_aliases():
    """Deserialised mirrors can expose feature classes only by class name."""

    class Array2D:
        dtype = "float32"
        shape = (2, 3)

    class VideoFrame:
        _type = "VideoFrame"

    class FakeDataset:
        features = {"depth": Array2D(), "camera": VideoFrame()}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"depth": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "camera": None}

    class Official:
        # No Value/Sequence/Image aliases on purpose.
        datasets = type("Datasets", (), {})

    features = _infer_features(Official, FakeDataset())
    assert features["depth"] == {"dtype": "float32", "shape": (2, 3), "names": None}
    assert features["camera"] == {
        "dtype": "video",
        "shape": None,
        "names": ["height", "width", "channels"],
    }


def test_v16_feature_inference_rejects_ragged_variable_sequences():
    class Value:
        dtype = "float32"

    class Sequence:
        feature = Value()
        length = None

    class FakeDataset:
        features = {"state": Sequence()}
        rows = [{"state": [0.0, 1.0]}, {"state": [0.0, 1.0, 2.0]}]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    class Official:
        datasets = type("Datasets", (), {"Value": Value, "Sequence": Sequence})

    with pytest.raises(ValueError, match="ragged"):
        _infer_features(Official, FakeDataset())
