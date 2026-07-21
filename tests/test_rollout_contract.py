"""The runtime rollout behavior itself is covered in test_model_action_isolation.

This small static test protects the public result names without loading pretrained towers.
"""

from world_critic.model import LatentRolloutOutput


def test_rollout_output_fields_are_explicit():
    assert list(LatentRolloutOutput.__dataclass_fields__) == ["latents", "values"]
