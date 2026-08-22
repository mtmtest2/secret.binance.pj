"""Small, dependency-free helpers used across the whole system.

The functions here are deliberately pure and side-effect free (with the obvious
exception of :func:`async_retry`, which sleeps) so they can be reasoned about
and unit-tested in isolation.
"""

from __future__ import annotations

import asyncio
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

MS_PER_MINUTE: int = 60_000


# ---------------------------------------------------------------------------
# Time helpers -- the entire system works in UTC milliseconds internally.
# ---------------------------------------------------------------------------
def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(tz=timezone.utc)


def utc_now_ms() -> int:
    """Return the current UTC time as integer milliseconds since the epoch."""
    return int(utc_now().timestamp() * 1_000)


def ms_to_datetime(timestamp_ms: int) -> datetime:
    """Convert epoch milliseconds into a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(timestamp_ms / 1_000.0, tz=timezone.utc)


def datetime_to_ms(moment: datetime) -> int:
    """Convert a datetime (naive values are assumed UTC) into epoch milliseconds."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1_000)


def floor_to_timeframe(timestamp_ms: int, timeframe_ms: int) -> int:
    """Floor an epoch-ms timestamp onto the timeframe grid.

    5-minute candles always open on a timestamp that is an exact multiple of
    300_000 ms, which makes grid alignment a pure modulo operation.
    """
    if timeframe_ms <= 0:
        raise ValueError("timeframe_ms must be positive")
    return timestamp_ms - (timestamp_ms % timeframe_ms)


def last_closed_candle_open_ms(timeframe_ms: int, now_ms: int | None = None) -> int:
    """Return the open time of the most recent *fully closed* candle."""
    reference: int = utc_now_ms() if now_ms is None else now_ms
    current_open: int = floor_to_timeframe(reference, timeframe_ms)
    return current_open - timeframe_ms


def expected_timestamps(start_ms: int, end_ms: int, timeframe_ms: int) -> list[int]:
    """Return every timestamp on the grid within ``[start_ms, end_ms]`` inclusive."""
    if timeframe_ms <= 0:
        raise ValueError("timeframe_ms must be positive")
    start: int = floor_to_timeframe(start_ms, timeframe_ms)
    end: int = floor_to_timeframe(end_ms, timeframe_ms)
    if end < start:
        return []
    return list(range(start, end + timeframe_ms, timeframe_ms))


def longest_clean_trailing_run(
    timestamps: Iterable[int],
    bad: set[int],
    timeframe_ms: int,
) -> list[int]:
    """Return the longest contiguous run of timestamps ending at the newest one.

    Every timestamp in ``bad`` is dropped outright (e.g. a candle flagged
    corrupt by QC); a timestamp that was never present in ``timestamps`` at
    all opens the same kind of gap naturally.  Either way, the last such gap -
    scanning backward from the newest entry - marks where the most recent
    *unhealable* damage sits, and everything from there forward is what's
    returned: downstream consumers of a candle series expect a single
    contiguous grid, never history with a hole punched in the middle of it.

    Shared by :class:`module_a_data.qc_validator.QCValidator` (quarantining an
    unhealable window at ingestion) and
    :class:`module_b_features.processor.DatasetProcessor` (trimming a stored
    window that turns out to be gappy before it reaches feature engineering),
    so both apply the exact same "keep the newest clean run" rule.

    Returns:
        The kept timestamps, sorted ascending.  Empty if nothing survives.
    """
    cleaned: list[int] = sorted(ts for ts in timestamps if ts not in bad)
    if not cleaned:
        return []
    cut_index: int = 0
    for index in range(1, len(cleaned)):
        if cleaned[index] - cleaned[index - 1] > timeframe_ms:
            cut_index = index
    return cleaned[cut_index:]


def git_commit_hash() -> str:
    """Best-effort short git commit hash of the running checkout.

    Used to stamp model artifacts and diagnostic reports so a training run is
    reproducible - never raises: returns ``"unknown"`` outside a git checkout
    or when the ``git`` binary is unavailable (e.g. a stripped-down
    deployment image).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment-dependent
        return "unknown"
    commit: str = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to ``float``, falling back to ``default`` on failure."""
    if value is None:
        return default
    try:
        result: float = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):  # NaN / inf guard
        return default
    return result


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` into the inclusive ``[minimum, maximum]`` interval."""
    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum")
    return max(minimum, min(maximum, value))


def percentile_rank(series: Sequence[float], value: float) -> float:
    """Return the fraction of ``series`` entries that are <= ``value``."""
    if not series:
        return 0.5
    below: int = sum(1 for item in series if item <= value)
    return below / len(series)


def chunked(items: Sequence[T], size: int) -> Iterable[list[T]]:
    """Yield consecutive ``size``-length chunks from ``items``."""
    if size <= 0:
        raise ValueError("size must be positive")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------
def backoff_delay(
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    jitter: float = 0.25,
) -> float:
    """Compute an exponentially-growing, jittered retry delay.

    Args:
        attempt: Zero-based retry attempt index.
        base_seconds: Delay for ``attempt == 0``.
        max_seconds: Hard ceiling on the returned delay.
        jitter: Relative jitter applied symmetrically (``0.25`` == +/-25 %).

    Returns:
        A non-negative delay in seconds.
    """
    raw: float = base_seconds * (2.0**max(0, attempt))
    capped: float = min(raw, max_seconds)
    if jitter <= 0.0:
        return capped
    spread: float = capped * jitter
    return max(0.0, capped + random.uniform(-spread, spread))


async def async_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_seconds: float,
    max_seconds: float,
    jitter: float = 0.25,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    give_up_on: tuple[type[BaseException], ...] = (),
    on_error: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Await ``operation`` with exponential backoff.

    Args:
        operation: Zero-argument coroutine factory to invoke on each attempt.
        attempts: Total number of attempts (>= 1).
        base_seconds: Initial backoff delay.
        max_seconds: Maximum backoff delay.
        jitter: Relative jitter applied to each delay.
        retry_on: Exception types that trigger another attempt.
        give_up_on: Exception types that abort immediately, even if they are also
            covered by ``retry_on`` (checked first).
        on_error: Optional callback ``(attempt_index, error, next_delay)``.

    Returns:
        Whatever ``operation`` returns on its first successful attempt.

    Raises:
        The final exception once the attempt budget is exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except give_up_on:
            raise
        except retry_on as error:  # type: ignore[misc]
            last_error = error
            if attempt == attempts - 1:
                break
            delay: float = backoff_delay(attempt, base_seconds, max_seconds, jitter)
            if on_error is not None:
                on_error(attempt, error, delay)
            await asyncio.sleep(delay)

    assert last_error is not None  # defensive: loop always assigns before breaking
    raise last_error
