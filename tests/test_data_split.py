from world_critic.data import build_episode_split


def test_episode_split_has_no_leakage_and_is_deterministic():
    a = build_episode_split(range(20), val_fraction=0.2, seed=42)
    b = build_episode_split(range(20), val_fraction=0.2, seed=42)
    assert a == b
    assert not (set(a.train) & set(a.val))
    assert set(a.train) | set(a.val) == set(range(20))
