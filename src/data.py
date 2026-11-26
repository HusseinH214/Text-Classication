"""Data loading. The notebook, training, and tests all read CSVs through here."""

from pathlib import Path

import pandas as pd

from src import config


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a tweets CSV with the project's encoding settings."""
    return pd.read_csv(
        path,
        encoding=config.CSV_ENCODING,
        encoding_errors=config.CSV_ENCODING_ERRORS,
    )

