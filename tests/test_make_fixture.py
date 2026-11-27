import pandas as pd

from src import config
from src.make_fixture import make_fixture


def make_labeled(counts):
    """Build a DataFrame with the given number of rows per label."""
    labels = [label for label, k in counts.items() for _ in range(k)]
    texts = [f"tweet {i}" for i in range(len(labels))]
    return pd.DataFrame({config.TEXT_COL: texts, config.LABEL_COL: labels})


def test_make_fixture_size_and_proportions():
    train = make_labeled({"Positive": 600, "Negative": 300, "Neutral": 100})
    sample = make_fixture(train, n=100, seed=0)
    assert len(sample) == 100
    assert sample[config.LABEL_COL].value_counts().to_dict() == {
        "Positive": 60,
        "Negative": 30,
        "Neutral": 10,
    }


def test_make_fixture_same_seed_same_sample():
    train = make_labeled({"Positive": 60, "Negative": 40})
    a = make_fixture(train, n=20, seed=1)
    b = make_fixture(train, n=20, seed=1)
    pd.testing.assert_frame_equal(a, b)