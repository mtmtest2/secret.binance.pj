"""Purged, embargoed walk-forward cross-validation for overlapping-label data.

Why this exists
----------------
Every label the system trains on looks ``horizon`` bars into the future (see
:class:`~module_b_features.labeler.TradeLabeler`), so labels at adjacent
timestamps share most of their look-ahead window.  Two consequences that a
plain ``sklearn.model_selection.KFold`` ignores completely:

1. **Forward leakage.**  A training row within ``horizon`` bars of a
   validation block's start has a label that looks *into* that validation
   block.  Training on it and then scoring the validation block overstates
   accuracy - the model effectively got to see (part of) the answer.
2. **Non-stationarity / regime dependence.**  A single 80/20 tail split (the
   system's previous validation scheme) reports exactly one point estimate,
   drawn from whichever regime the last 20% of the series happened to be in.
   A model can look good or bad purely because of which slice it landed in.

This module implements **expanding-window walk-forward CV with a purge
margin**: each fold's validation block is a contiguous, chronologically later
slice than its training data (never trained on the future relative to its own
validation block - the property that makes reported metrics a genuine
simulation of "deploy after this point in time, then trade forward"), and a
purge gap of ``horizon + embargo_bars`` bars is removed immediately before
each validation block so no training label reaches into it.

A note on "embargo": the classic Combinatorial Purged K-Fold (de Prado,
*Advances in Financial Machine Learning*, ch. 7) also purges a strip *after*
each validation block, because in K-fold CV a validation block can sit in the
middle of the timeline with training data on both sides. In this expanding-
window walk-forward scheme, a fold's training set is always a strict prefix
of its own validation block - there is no "training data after this fold's
own validation block" for that same fold to leak from. We fold the embargo
margin into the same pre-validation purge width instead (``horizon +
embargo_bars``), which is the standard, simpler adaptation of the technique
for expanding-window (rather than scattered K-fold) splits, and still gives
an extra safety margin beyond the strict label horizon for the serial
correlation label construction can introduce (rolling GARCH/HMM refits,
intra-candle refinement, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

_MIN_FOLD_ROWS: Final[int] = 50


@dataclass(slots=True, frozen=True)
class CVFold:
    """One walk-forward fold's index arrays, already purged and embargoed.

    Both index arrays are positional (``0..n_rows-1``), ready to use with
    ``DataFrame.iloc`` / ``ndarray`` fancy indexing.
    """

    fold_index: int
    train_index: np.ndarray
    validation_index: np.ndarray


def purged_walk_forward_splits(
    n_rows: int,
    n_splits: int,
    horizon: int,
    embargo_bars: int = 0,
) -> list[CVFold]:
    """Generate expanding-window, purged & embargoed walk-forward folds.

    ``range(n_rows)`` is cut into ``n_splits`` contiguous, equal-sized blocks
    in chronological order.  Block 0 is never a validation block (there is no
    prior data to train on without leaking the future), so this yields at
    most ``n_splits - 1`` folds; blocks too small to be useful on either side
    are silently skipped rather than raising, since callers run this across
    datasets of very different sizes (a full year of pooled 5m data vs. a
    quick smoke test).

    Args:
        n_rows: Total rows in the (already time-ordered) dataset.
        n_splits: Number of equal-sized blocks to cut the series into.
        horizon: Label look-ahead in bars (``labels.max_holding_bars``) - the
            core purge width.
        embargo_bars: Extra bars folded into the same pre-validation purge
            margin (see module docstring for why this differs from classic
            K-fold embargo).

    Returns:
        One :class:`CVFold` per usable split, in chronological order.  Can be
        empty if the dataset is too small for even one purged fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if horizon < 0 or embargo_bars < 0:
        raise ValueError("horizon and embargo_bars must be >= 0")
    if n_rows <= n_splits:
        return []

    purge_width: int = horizon + embargo_bars
    block_size: int = n_rows // n_splits
    boundaries: list[int] = [i * block_size for i in range(n_splits)]
    boundaries.append(n_rows)

    folds: list[CVFold] = []
    for split in range(1, n_splits):
        val_start: int = boundaries[split]
        val_end: int = boundaries[split + 1] if split + 1 < len(boundaries) else n_rows

        train_end: int = val_start - purge_width
        if train_end < _MIN_FOLD_ROWS:
            continue  # not enough history left after purging to train on
        if val_end - val_start < _MIN_FOLD_ROWS:
            continue  # trailing sliver, not worth scoring

        folds.append(
            CVFold(
                fold_index=len(folds),
                train_index=np.arange(0, train_end, dtype=np.int64),
                validation_index=np.arange(val_start, val_end, dtype=np.int64),
            )
        )

    return folds


def out_of_fold_coverage(folds: list[CVFold], n_rows: int) -> np.ndarray:
    """Boolean mask of rows that were validated by at least one fold.

    The leading block (before the first fold's validation start) never gets
    an out-of-fold prediction - there is no earlier data to have trained a
    model on without peeking at the future - so callers must handle a
    partially-covered output (e.g. treat uncovered rows' stacked features as
    neutral/missing, same as any other warm-up gap in this pipeline).
    """
    covered: np.ndarray = np.zeros(n_rows, dtype=bool)
    for fold in folds:
        covered[fold.validation_index] = True
    return covered
