"""Mainnet-only operation, and recovery from the failures the first live run hit.

Four separate defects shared one shape: a transient condition was recorded as a
permanent fact, and nothing downstream could tell the difference afterwards.

* a sandbox flag silently swapped the market data every model trains on;
* a download timeout wrote the same "this day does not exist" marker a genuine
  404 wrote, so one bad run pinned a dataset at 0% coverage forever;
* a heal budget expiring with one bar still flagged discarded a symbol whose
  other ~120,000 bars had healed cleanly;
* a scheduler skip during a two-year backfill logged the same warning it logs
  when live candles are actually being missed.

Each test below fixes the *distinction*, not the symptom.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from config.settings import Settings
from module_a_data.archive_loader import (
    _ABSENT_MARKER_SUFFIX,
    _LEGACY_ABSENT_SUFFIX,
    ArchiveCoverage,
    BinanceArchiveLoader,
    DownloadOutcome,
    purge_legacy_absent_markers,
)
from module_a_data.pipeline import summarise_archive_coverage_by_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DAY = date(2026, 1, 5)
_DATASET = "bookTicker"
_SYMBOL = "BTCUSDT"


# ---------------------------------------------------------------------------
# Mainnet only
# ---------------------------------------------------------------------------
def test_no_production_code_enables_sandbox_mode() -> None:
    """No module may put the exchange client into testnet mode.

    Asserted against the source rather than against one client, because the
    regression this prevents is a *new* call site somewhere else - the original
    one was a single line in the fetcher.
    """
    result = subprocess.run(
        ["grep", "-rn", "set_sandbox_mode", "--include=*.py", "."],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    offenders = [
        line
        for line in result.stdout.splitlines()
        if line.strip()
        and not line.startswith("./tests/")
        and not line.lstrip("./").startswith("scripts/")
        # A comment explaining why the call is absent is not a call.
        and "#" not in line.split("set_sandbox_mode")[0].split(":", 2)[-1]
    ]
    assert offenders == [], "sandbox mode must never be enabled:\n" + "\n".join(offenders)


def test_testnet_setting_defaults_to_false() -> None:
    assert Settings().exchange.testnet is False


def test_stale_testnet_flag_is_reported_not_silently_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A leftover ``EXCHANGE__TESTNET=true`` must be loud.

    Ignoring it quietly would be the same class of bug as honouring it: the
    operator keeps believing the system runs where they configured it to.
    """
    with caplog.at_level(logging.ERROR):
        settings = Settings(exchange={"testnet": True})
    assert settings.exchange.testnet is True  # preserved, so it can be reported
    assert any("mainnet-only" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# Archive: a failed download is not an absent day
# ---------------------------------------------------------------------------
@pytest.fixture()
def loader(tmp_path: Path) -> BinanceArchiveLoader:
    return BinanceArchiveLoader(Settings(), cache_dir=tmp_path)


def _marker_for(loader: BinanceArchiveLoader) -> Path:
    return loader._cache_path(_SYMBOL, _DATASET, _DAY).with_suffix(_ABSENT_MARKER_SUFFIX)


def _stub_download(
    loader: BinanceArchiveLoader,
    outcome: DownloadOutcome,
    calls: list[Any],
) -> None:
    async def _fake(archive_symbol: str, dataset: str, day: date) -> tuple[DownloadOutcome, None]:
        calls.append((archive_symbol, dataset, day))
        return outcome, None

    loader._download = _fake  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_failed_download_is_retried_not_cached_as_absent(
    loader: BinanceArchiveLoader,
) -> None:
    """The poisoning bug: a timeout must leave no permanent marker behind."""
    calls: list[Any] = []
    _stub_download(loader, DownloadOutcome.FAILED, calls)
    coverage = ArchiveCoverage(symbol=_SYMBOL, dataset=_DATASET)

    assert await loader._one_day(_SYMBOL, _DATASET, _DAY, coverage) is None
    assert coverage.days_failed == 1
    assert coverage.days_absent == 0
    assert not _marker_for(loader).exists()

    # ...and the next run tries again rather than trusting a cached verdict.
    assert await loader._one_day(_SYMBOL, _DATASET, _DAY, coverage) is None
    assert len(calls) == 2
    assert coverage.days_failed == 2


@pytest.mark.asyncio
async def test_absent_day_is_cached_and_not_re_requested(
    loader: BinanceArchiveLoader,
) -> None:
    """A real 404 still short-circuits: that saves an HTTP round trip per day."""
    calls: list[Any] = []
    _stub_download(loader, DownloadOutcome.ABSENT, calls)
    coverage = ArchiveCoverage(symbol=_SYMBOL, dataset=_DATASET)

    assert await loader._one_day(_SYMBOL, _DATASET, _DAY, coverage) is None
    assert coverage.days_absent == 1
    assert _marker_for(loader).exists()

    assert await loader._one_day(_SYMBOL, _DATASET, _DAY, coverage) is None
    assert len(calls) == 1, "a cached absence must not be re-downloaded"
    assert coverage.days_absent == 2


@pytest.mark.asyncio
async def test_legacy_absent_marker_is_retried_once_and_rewritten(
    loader: BinanceArchiveLoader,
) -> None:
    """Markers from before outcomes existed cannot be believed, so they are retried."""
    legacy = loader._cache_path(_SYMBOL, _DATASET, _DAY).with_suffix(_LEGACY_ABSENT_SUFFIX)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.touch()

    calls: list[Any] = []
    _stub_download(loader, DownloadOutcome.ABSENT, calls)
    coverage = ArchiveCoverage(symbol=_SYMBOL, dataset=_DATASET)

    assert await loader._one_day(_SYMBOL, _DATASET, _DAY, coverage) is None
    assert len(calls) == 1, "an untrustworthy marker must not short-circuit the download"
    assert not legacy.exists()
    assert _marker_for(loader).exists(), "the retry rewrites the marker in the trusted format"


def test_purge_legacy_absent_markers_removes_only_the_untrusted_generation(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / _DATASET / _SYMBOL / f"{_DAY.isoformat()}{_LEGACY_ABSENT_SUFFIX}"
    trusted = tmp_path / _DATASET / _SYMBOL / f"2026-01-06{_ABSENT_MARKER_SUFFIX}"
    payload = tmp_path / _DATASET / _SYMBOL / "2026-01-07.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    for path in (legacy, trusted, payload):
        path.touch()

    assert purge_legacy_absent_markers(tmp_path) == 1
    assert not legacy.exists()
    assert trusted.exists()
    assert payload.exists()


def test_purge_is_safe_on_a_cache_that_does_not_exist(tmp_path: Path) -> None:
    assert purge_legacy_absent_markers(tmp_path / "never-created") == 0


# ---------------------------------------------------------------------------
# Archive: zero coverage with zero failures is an error, not health
# ---------------------------------------------------------------------------
def test_zero_coverage_with_no_failures_is_escalated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The exact shape of the live run: every day "absent", nothing flagged."""
    reports = [
        ArchiveCoverage(
            symbol=f"SYM{index}",
            dataset=_DATASET,
            days_requested=400,
            days_downloaded=0,
            days_absent=400,
            days_failed=0,
        )
        for index in range(3)
    ]
    with caplog.at_level(logging.ERROR):
        summary = summarise_archive_coverage_by_dataset(reports)

    assert summary[_DATASET]["coverage_pct"] == 0.0
    assert summary[_DATASET]["failure_rate"] == 0.0
    messages = [record.getMessage() for record in caplog.records]
    assert any("near-zero coverage" in message for message in messages)


def test_healthy_coverage_does_not_escalate(caplog: pytest.LogCaptureFixture) -> None:
    reports = [
        ArchiveCoverage(
            symbol="BTCUSDT",
            dataset=_DATASET,
            days_requested=400,
            days_downloaded=380,
            days_absent=20,
            days_failed=0,
        )
    ]
    with caplog.at_level(logging.ERROR):
        summary = summarise_archive_coverage_by_dataset(reports)
    assert summary[_DATASET]["coverage_pct"] == 0.95
    assert caplog.records == []


def test_short_history_is_not_judged_on_coverage(caplog: pytest.LogCaptureFixture) -> None:
    """A symbol listed last week legitimately has almost no archive."""
    reports = [
        ArchiveCoverage(
            symbol="NEWUSDT",
            dataset=_DATASET,
            days_requested=5,
            days_downloaded=0,
            days_absent=5,
            days_failed=0,
        )
    ]
    with caplog.at_level(logging.ERROR):
        summarise_archive_coverage_by_dataset(reports)
    assert caplog.records == []


def test_datasets_are_summarised_independently() -> None:
    """One number cannot say "book empty, liquidations fine" - two can."""
    reports = [
        ArchiveCoverage(
            symbol="BTCUSDT",
            dataset="bookTicker",
            days_requested=100,
            days_downloaded=0,
            days_absent=100,
        ),
        ArchiveCoverage(
            symbol="BTCUSDT",
            dataset="liquidationSnapshot",
            days_requested=100,
            days_downloaded=53,
            days_absent=47,
        ),
    ]
    summary = summarise_archive_coverage_by_dataset(reports)
    assert summary["bookTicker"]["coverage_pct"] == 0.0
    assert summary["liquidationSnapshot"]["coverage_pct"] == 0.53
