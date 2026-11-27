"""Project-wide settings. Keep parameters here, not scattered across modules."""

from pathlib import Path

# Repo root, found from this file's location, so paths work from any working directory.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# The train CSV has 4 truncated UTF-8 sequences; "replace" turns them into U+FFFD.
CSV_ENCODING = "utf-8"
CSV_ENCODING_ERRORS = "replace"
SEED = 6

# Training cleaner replaces URLs with this token 
# Letters only, so lowercasing and TF-IDF's default tokenizer leave it intact.
URL_TOKEN = "urltoken"

# Columns used by the model; everything else is dropped in src/data.py.
TEXT_COL = "OriginalTweet"
LABEL_COL = "Sentiment"

# Allowed labels, ordered from most negative to most positive.
LABELS = (
    "Extremely Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Extremely Positive",
)

# Full dataset (local only; gitignored, so CI never has it).
TRAIN_CSV = DATA_DIR / "Corona_NLP_train.csv"
TEST_CSV = DATA_DIR / "Corona_NLP_test.csv"

# Small stratified sample of the prepared train set; the only data CI sees.
FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "sample.csv"
FIXTURE_SIZE = 1000