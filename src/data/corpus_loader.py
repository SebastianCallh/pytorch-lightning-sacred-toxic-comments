from functools import partial
from typing import Callable, List, Tuple, Sequence

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


def collate_tokens(tokens: List[torch.Tensor],
                   max_len: int,
                   pad_token: int) -> torch.Tensor:
    """ Convert a list of 1d tensors into a padded 2d tensor.
    Args:
        max_len: Each token tensor is truncated to this dimension.
        pad_token: Token used to pad token sequences up to max_len.
        this token is replicated to pad to it.
    Returns:
        A mini-batch tensor of size (size(tokens, 0), max_len).
    """

    size = max(min(x.size(0), max_len) for x in tokens)

    def pad(x):
        return F.pad(x, (0, size - len(x)), value=pad_token)

    return torch.stack([pad(x) for x in tokens])


def collate_fn(obs: List[Tuple[str, str]],
               labelize: Callable[[str], int],
               tokenize: Callable[[str], torch.Tensor],
               pad_token: int,
               max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function used to collate observations into a batch.
    Args:
        labelize: Function to map a string class label onto integer.
        tokenize: Function to map a sentence to a token tensor.
        max_seq_len: Dimension to truncate the batch to.
        pad_token: Token used to pad short token sequences up to max_seq_len.
    Returns:
        The truncated and padded batch and corresponding integer labels.
    """
    sentences, labels = zip(*obs)
    X = collate_tokens(list(map(tokenize, sentences)),
                       max_len=max_seq_len,
                       pad_token=pad_token)
    y = torch.tensor(list(map(labelize, labels)))
    return X, y


def CorpusLoader(observations: Sequence[Tuple[str, str]],
                 batch_size: int,
                 max_seq_len: int,
                 pad_token: int,
                 tokenize: Callable[[str], torch.Tensor],
                 labelize: Callable[[str], int],
                 drop_last: bool = True,
                 shuffle: bool = True) -> DataLoader:
    """Convenience function for creating a DataLoader.
    Args:
        observations: The sequence of data to load data from.
        batch_size: Size of a single mini-batch.
        max_seq_len: Maximum number of tokens per sentence.
        shuffle: Whether to shuffle the observations before loading or not.
        pad_token: Token used to pad short token sequences up to max_seq_len.
        labelize: Function to map a string class label onto integer.
        tokenize: Function to map a sentence to a token tensor.
    Returns:
        A DataLoader object.
    """
    collate = partial(collate_fn,
                      max_seq_len=max_seq_len,
                      pad_token=pad_token,
                      tokenize=tokenize,
                      labelize=labelize)

    return DataLoader(observations,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      collate_fn=collate,
                      num_workers=4)
