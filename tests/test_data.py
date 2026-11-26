import pandas as pd
import pytest

from src import config
from src.data import (
    dedupe,
    drop_empty_text,
    load_csv,
    prepare,
    remove_overlap,
    select_columns,
)
from src.preprocessing import normalize_for_dedup

TEXT = config.TEXT_COL
LABEL = config.LABEL_COL


def make_df(texts, labels=None):
    """Build a tiny DataFrame with the project's text and label columns."""
    if labels is None:
        labels = ["Neutral"] * len(texts)
    return pd.DataFrame({TEXT: texts, LABEL: labels})


def test_select_columns_keeps_only_text_and_label():
    raw = make_df(["a tweet"]).assign(UserName=[1], Location=["London"])
    assert list(select_columns(raw).columns) == [TEXT, LABEL]


def test_select_columns_does_not_change_input():
    raw = make_df(["a tweet"]).assign(UserName=[1])
    select_columns(raw)
    assert "UserName" in raw.columns


def test_drop_empty_text_removes_mention_only_and_blank():
    df = make_df(["@bob @ann", "hello @ann", "   ", "real tweet"])
    result = drop_empty_text(df)
    assert list(result[TEXT]) == ["hello @ann", "real tweet"]
    assert list(result.index) == [0, 1]


def test_remove_overlap_uses_normalized_text():
    train = make_df(["Stay home @bob", "Buy masks", "unique"])
    test = make_df(["stay home @ann https://t.co/x"])
    result = remove_overlap(train, test)
    assert list(result[TEXT]) == ["Buy masks", "unique"]


def test_remove_overlap_leaves_test_unchanged():
    train = make_df(["Stay home"])
    test = make_df(["stay home"])
    remove_overlap(train, test)
    assert list(test[TEXT]) == ["stay home"]


def test_dedupe_keeps_first_copy():
    df = make_df(
        ["Panic buying @bob", "panic buying", "other"],
        ["Negative", "Extremely Negative", "Neutral"],
    )
    result = dedupe(df)
    assert list(result[TEXT]) == ["Panic buying @bob", "other"]
    assert list(result[LABEL]) == ["Negative", "Neutral"]


def test_prepare_runs_all_steps():
    train = make_df(["@bob", "Stay home", "Buy masks", "buy masks @ann"])
    train = train.assign(UserName=[1, 2, 3, 4])
    test = make_df(["stay home"]).assign(UserName=[9])
    new_train, new_test = prepare(train, test)
    assert list(new_train[TEXT]) == ["Buy masks"]
    assert list(new_train.columns) == [TEXT, LABEL]
    assert list(new_test[TEXT]) == ["stay home"]


TRAIN_CSV = config.DATA_DIR / "Corona_NLP_train.csv"
TEST_CSV = config.DATA_DIR / "Corona_NLP_test.csv"


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="full dataset not available (e.g. in CI)")
def test_prepare_on_full_data_matches_eda():
    train, test = prepare(load_csv(TRAIN_CSV), load_csv(TEST_CSV))

    assert len(train) == 41_075
    assert len(test) == 3_798

    train_norm = train[TEXT].map(normalize_for_dedup)
    test_norm = test[TEXT].map(normalize_for_dedup)
    assert not set(train_norm) & set(test_norm), "train/test overlap remains"
    assert not train_norm.duplicated().any(), "duplicates remain in train"