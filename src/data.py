"""Data loading and row-level preparation. Training, tests and the notebook all use this."""

from pathlib import Path

import pandas as pd

from src import config
from src.preprocessing import normalize_for_dedup

"""Data loading. The notebook, training, and tests all read CSVs through here."""
def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a tweets CSV with the project's encoding settings."""
    return pd.read_csv(
        path,
        encoding=config.CSV_ENCODING,
        encoding_errors=config.CSV_ENCODING_ERRORS,
    )


def _normalized(df: pd.DataFrame) -> pd.Series:
    """Normalized tweet text, used to decide which rows count as the same tweet."""
    return df[config.TEXT_COL].map(normalize_for_dedup)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the text and label columns."""
    return df[[config.TEXT_COL, config.LABEL_COL]]


def drop_empty_text(df: pd.DataFrame) -> pd.DataFrame:
    """Drop tweets that are empty after normalization (e.g. mention-only)."""
    keep = _normalized(df) != ""
    return df[keep].reset_index(drop=True)


def remove_overlap(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Drop train rows whose normalized text also appears in test. Test is not changed."""
    test_texts = set(_normalized(test))
    keep = ~_normalized(train).isin(test_texts)
    return train[keep].reset_index(drop=True)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first row of each normalized text."""
    keep = ~_normalized(df).duplicated(keep="first")
    return df[keep].reset_index(drop=True)


def prepare(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all preparation steps. Returns (train, test); test only loses columns."""
    train = select_columns(train)
    test = select_columns(test)
    train = drop_empty_text(train)
    train = remove_overlap(train, test)
    train = dedupe(train)
    return train, test