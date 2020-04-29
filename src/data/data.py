import os
import pandas as pd

from .corpus import Corpus
from .corpus_loader import CorpusLoader
from torch.utils.data import DataLoader

DATA_PATH = 'data'
toxic_labels = ["severe_toxic", "obscene", "threat",
                "insult", "identity_hate"]


def pre_process(df: pd.DataFrame) -> pd.DataFrame:

    def is_toxic(c: pd.Series) -> int:
        cols = ["severe_toxic", "obscene", "threat",
                "insult", "identity_hate"]
        return int(any(c[col] == 1 for col in cols))

    return pd.DataFrame({
        'comment': df.comment_text,
        'toxic': df.apply(is_toxic, axis=1)
    })


def load_corpus(path: str) -> Corpus:
    df = pd.read_csv(path)
    return Corpus(text=df.comment, labels=df.toxic)


def train_corpus() -> Corpus:
    path = os.path.join(DATA_PATH, 'processed_train.csv')
    return load_corpus(path)


def test_corpus() -> Corpus:
    path = os.path.join(DATA_PATH, 'processed_test.csv')
    return load_corpus(path)


def train_loader(**kwargs) -> DataLoader:
    return CorpusLoader(train_corpus(), **kwargs)


def test_loader(**kwargs) -> DataLoader:
    return CorpusLoader(test_corpus(), **kwargs)
