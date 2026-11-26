"""Text cleaning: small pure functions, string in, string out."""

import html
import re

from src import config

URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")

def unescape_html(text: str) -> str:
    """Turn HTML entities into characters: '&amp;' -> '&'."""
    return html.unescape(text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (incl. line breaks) to one space; trim the ends."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Cleaner for training and prediction. URLs become a token instead."""
    text = unescape_html(text)
    text = text.lower()
    text = replace_urls(text)  # before mentions: URLs can contain '@'
    text = remove_mentions(text)
    return normalize_whitespace(text)


def remove_urls(text: str) -> str:
    """Delete URLs. Used for duplicate matching."""
    return URL_PATTERN.sub(" ", text)


def replace_urls(text: str) -> str:
    """Swap each URL for config.URL_TOKEN, keeping the 'has a link' signal."""
    return URL_PATTERN.sub(f" {config.URL_TOKEN} ", text)


def remove_mentions(text: str) -> str:
    """Delete @mentions. Hashtags are kept, since they carry meaning."""
    return MENTION_PATTERN.sub(" ", text)


def normalize_for_dedup(text: str) -> str:
    """Defines 'same tweet' for dedup and leakage checks. URLs are removed."""
    text = unescape_html(text)
    text = text.lower()
    text = remove_urls(text)  # before mentions: URLs can contain '@'
    text = remove_mentions(text)
    return normalize_whitespace(text)