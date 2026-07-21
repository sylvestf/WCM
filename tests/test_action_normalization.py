import numpy as np

from world_critic.config import DataConfig
from world_critic.data import fit_action_normalization


class FakeDataset:
    features = {"action": {"shape": (2,)}}

    def __init__(self):
        self.hf_dataset = {
            "episode_index": [0, 0, 1, 1],
            "action": [[1.0, 10.0], [3.0, 14.0], [100.0, 100.0], [200.0, 200.0]],
        }
        self.hf_dataset = FakeTable(self.hf_dataset)


class FakeTable(dict):
    @property
    def column_names(self):
        return list(self)


def test_action_stats_use_train_episodes_only():
    config = DataConfig(repo_id="test/repo")
    fit_action_normalization(FakeDataset(), config, train_episode_ids=[0])
    np.testing.assert_allclose(config.action_mean, [2.0, 12.0])
    np.testing.assert_allclose(config.action_std, [1.0, 2.0])
