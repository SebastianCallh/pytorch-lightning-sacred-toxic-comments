import torch
from typing import Sequence, Tuple, Iterator
from torch.utils.data import Dataset
from tabulate import tabulate


class Corpus(Dataset):
    """Represents a set of sentences with string labels."""

    def __init__(self, text: Sequence[str], labels: Sequence[str]):
        assert len(text) == len(labels)

        self.text = list(text)
        self.labels = list(labels)
        self.label_dict = {l: i for i, l in enumerate(sorted(set(labels)))}
        self.index_dict = {i: l for l, i in self.label_dict.items()}

    def __repr__(self) -> str:
        def truncate(x, n=75):
            return x[:n] + ('...' if len(x) > n else '')

        lim = 10
        truncated = map(truncate, self.text[:lim])
        return tabulate(zip(truncated, self.labels[:lim]),
                        headers=['Text', 'Labels'])

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx) -> Tuple[str, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.text[idx], self.labels[idx]

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return iter(zip(self.text, self.labels))

    def num_classes(self) -> int:
        indices = torch.tensor([self.index_of(l) for l in self.labels])
        return len(indices.unique())

    def class_weights(self) -> torch.Tensor:
        """Computes a tensor of class weights.
        The indices of the weights correspond
        to the indices returned by "self.index_of".
        """
        indices = torch.tensor([self.index_of(l) for l in self.labels])
        freq = torch.bincount(indices).float()
        return freq / sum(freq)

    def label_of(self, index: int) -> str:
        """Maps an integer index to a string label."""
        return self.index_dict[index]

    def index_of(self, label: str) -> int:
        """Maps a text label to an integer index."""
        return self.label_dict[label]
