import pytest

from src import config
from src.preprocessing import (
    clean_text,
    normalize_for_dedup,
    normalize_whitespace,
    remove_mentions,
    remove_urls,
    replace_urls,
    unescape_html,
)

TOKEN = config.URL_TOKEN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fish &amp; chips", "Fish & chips"),
        ("I &lt;3 this", "I <3 this"),
        ("no entities here", "no entities here"),
        ("", ""),
    ],
)
def test_unescape_html(raw, expected):
    assert unescape_html(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("too   many  spaces", "too many spaces"),
        ("line\r\nbreak", "line break"),
        ("  padded  ", "padded"),
        ("   ", ""),
        ("", ""),
    ],
)
def test_normalize_whitespace(raw, expected):
    assert normalize_whitespace(raw) == expected

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("see https://t.co/abc now", "see now"),
        ("http://example.com", ""),
        ("no link here", "no link here"),
        ("", ""),
    ],
)
def test_remove_urls(raw, expected):
    assert remove_urls(raw).split() == expected.split()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("see https://t.co/abc now", f"see {TOKEN} now"),
        ("a http://x.co b https://y.co", f"a {TOKEN} b {TOKEN}"),
        ("glued:https://t.co/x", f"glued: {TOKEN}"),
        ("", ""),
    ],
)
def test_replace_urls(raw, expected):
    assert replace_urls(raw).split() == expected.split()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("@bob hi there", "hi there"),
        ("thanks @Ann_99!", "thanks !"),
        ("#covid stays", "#covid stays"),
        ("@bob @ann", ""),
        ("", ""),
    ],
)
def test_remove_mentions(raw, expected):
    assert remove_mentions(raw).split() == expected.split()

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RT @bob: Masks &amp; gloves https://t.co/x", "rt : masks & gloves"),
        ("Read https://medium.com/@user/post", "read"),
        ("Line\r\nBREAK", "line break"),
        ("@bob @ann", ""),
        ("", ""),
    ],
)
def test_normalize_for_dedup(raw, expected):
    assert normalize_for_dedup(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Masks &amp; gloves https://t.co/x", f"masks & gloves {TOKEN}"),
        ("Read https://medium.com/@user/post", f"read {TOKEN}"),
        ("@bob @ann", ""),
        ("", ""),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected