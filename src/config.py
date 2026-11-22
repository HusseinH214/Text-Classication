"""Project-wide settings. Keep parameters here, not scattered across modules."""

from pathlib import Path

# Repo root, found from this file's location, so paths work from any working directory.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# The train CSV has 4 truncated UTF-8 sequences; "replace" turns them into U+FFFD.
CSV_ENCODING = "utf-8"
CSV_ENCODING_ERRORS = "replace"
