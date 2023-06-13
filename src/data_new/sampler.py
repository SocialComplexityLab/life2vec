from typing import Iterator, Sequence

from torch.utils.data.sampler import Sampler


class FixedSampler(Sampler[int]):
    r"""Samples elements from a given list of indices.
    Args:
        indices (sequence): a sequence of indices
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        for i in self.indices:
            yield i

    def __len__(self) -> int:
        return len(self.indices)
