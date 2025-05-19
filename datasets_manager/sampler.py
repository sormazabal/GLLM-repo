import torch
from torch.utils.data.sampler import Sampler


class SubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class SubsetWeightedRandomSampler(Sampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices, weights, replacement=True):
        self.indices = indices
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        for i in torch.multinomial(self.weights, self.num_samples, self.replacement):
            yield self.indices[i]

    def __len__(self):
        return self.num_samples

    @property
    def num_samples(self):
        return len(self.indices)

class BootstrapSubsetSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices, replacement=False):
        self.indices = indices
        self.replacement = replacement

    def __iter__(self):
        if self.replacement:
            for _ in range(self.num_samples // 16):
                for i in torch.randint(high=self.num_samples, size=(16,)).tolist():
                    yield self.indices[i]
            for i in torch.randint(high=self.num_samples, size=(self.num_samples % 16,)).tolist():
                yield self.indices[i]
        else:
            for i in torch.randperm(self.num_samples):
                yield self.indices[i]

    def __len__(self):
        return self.num_samples

    @property
    def num_samples(self):
        return len(self.indices)
