"""Build the stratified sample CI uses. Run from the repo root: python -m src.make_fixture"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data import load_csv, prepare


def make_fixture(
    train: pd.DataFrame, n: int = config.FIXTURE_SIZE, seed: int = config.SEED
) -> pd.DataFrame:
    """Stratified sample of n rows, with the same class proportions as train."""
    sample, _ = train_test_split(
        train, train_size=n, stratify=train[config.LABEL_COL], random_state=seed
    )
    return sample.reset_index(drop=True)


def main() -> None:
    train, _ = prepare(load_csv(config.TRAIN_CSV), load_csv(config.TEST_CSV))
    fixture = make_fixture(train)
    fixture.to_csv(config.FIXTURE_PATH, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Wrote {len(fixture)} rows to {config.FIXTURE_PATH}")
    print(fixture[config.LABEL_COL].value_counts().to_string())


if __name__ == "__main__":
    main()