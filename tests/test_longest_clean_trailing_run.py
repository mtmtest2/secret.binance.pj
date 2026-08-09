"""Unit tests for the shared trimming rule used by both Module A's ingestion-
time quarantine and Module B's read-time storage-integrity gate.
"""

from __future__ import annotations

from core.utils import longest_clean_trailing_run

TF_MS = 5 * 60 * 1_000
BASE = 1_700_000_000_000


def ts(*offsets: int) -> list[int]:
    return [BASE + offset * TF_MS for offset in offsets]


def test_no_gap_keeps_everything() -> None:
    timestamps = ts(0, 1, 2, 3, 4)
    result = longest_clean_trailing_run(timestamps, bad=set(), timeframe_ms=TF_MS)
    assert result == timestamps


def test_gap_in_the_middle_keeps_only_the_trailing_run() -> None:
    timestamps = ts(0, 1, 2, 10, 11, 12)  # a gap between index 2 and 3
    result = longest_clean_trailing_run(timestamps, bad=set(), timeframe_ms=TF_MS)
    assert result == ts(10, 11, 12)


def test_bad_timestamps_are_dropped_and_open_a_gap() -> None:
    timestamps = ts(0, 1, 2, 3, 4, 5)
    bad = set(ts(2, 3))  # a corrupt pair in the middle
    result = longest_clean_trailing_run(timestamps, bad=bad, timeframe_ms=TF_MS)
    assert result == ts(4, 5)


def test_multiple_gaps_keeps_only_the_last_run() -> None:
    timestamps = ts(0, 1, 5, 6, 20, 21, 22)
    result = longest_clean_trailing_run(timestamps, bad=set(), timeframe_ms=TF_MS)
    assert result == ts(20, 21, 22)


def test_everything_bad_returns_empty() -> None:
    timestamps = ts(0, 1, 2)
    result = longest_clean_trailing_run(timestamps, bad=set(timestamps), timeframe_ms=TF_MS)
    assert result == []


def test_empty_input_returns_empty() -> None:
    assert longest_clean_trailing_run([], bad=set(), timeframe_ms=TF_MS) == []


def test_unsorted_input_is_sorted_before_scanning() -> None:
    timestamps = ts(2, 0, 1, 10, 11)  # deliberately out of order
    result = longest_clean_trailing_run(timestamps, bad=set(), timeframe_ms=TF_MS)
    assert result == ts(10, 11)
