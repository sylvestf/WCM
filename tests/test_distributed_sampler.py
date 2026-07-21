import torch

from world_critic.distributed import DistributedEvalSampler


def test_eval_sampler_has_exact_union_without_padding():
    dataset = torch.utils.data.TensorDataset(torch.arange(11))
    shards = [list(DistributedEvalSampler(dataset, rank, 3)) for rank in range(3)]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))
