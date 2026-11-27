import pandas as pd
import pytest

from src import config
from src.data import load_csv, prepare
from src.make_fixture import make_fixture
from src.preprocessing import normalize_for_dedup

TEXT = config.TEXT_COL
LABEL = config.LABEL_COL

needs_full_data = pytest.mark.skipif(
    not config.TRAIN_CSV.exists(), reason="full dataset not available (e.g. in CI)"
)


def load_fixture():
    return load_csv(config.FIXTURE_PATH)


def load_prepared_train():
    train, _ = prepare(load_csv(config.TRAIN_CSV), load_csv(config.TEST_CSV))
    return train


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(load_fixture, id="fixture"),
        pytest.param(load_prepared_train, id="full-train", marks=needs_full_data),
    ],
)
def dataset(request):
    return request.param()


def test_required_columns(dataset):
    assert {TEXT, LABEL} <= set(dataset.columns)


def test_no_missing_values(dataset):
    assert not dataset[[TEXT, LABEL]].isna().any().any()


def test_labels_from_allowed_set(dataset):
    assert set(dataset[LABEL]) <= set(config.LABELS)


def test_all_classes_present(dataset):
    assert set(dataset[LABEL]) == set(config.LABELS)


def test_no_empty_text(dataset):
    assert (dataset[TEXT].map(normalize_for_dedup) != "").all()


def test_no_duplicate_text(dataset):
    assert not dataset[TEXT].map(normalize_for_dedup).duplicated().any()


def test_fixture_size():
    assert len(load_fixture()) == config.FIXTURE_SIZE


@needs_full_data
def test_fixture_is_up_to_date():
    expected = make_fixture(load_prepared_train())
    pd.testing.assert_frame_equal(load_fixture(), expected)