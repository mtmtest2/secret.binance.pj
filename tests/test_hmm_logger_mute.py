"""hmmlearn's ConvergenceMonitor logs through `logging`, bypassing `warnings`
filters - module_b_features/features.py::_muted_logger closes that gap."""

from __future__ import annotations

import logging

from module_b_features.features import _muted_logger


def test_muted_logger_suppresses_only_within_the_context() -> None:
    logger = logging.getLogger("hmmlearn.base")
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        logger.warning("before")
        with _muted_logger("hmmlearn.base"):
            logger.warning("during")
        logger.warning("after")
    finally:
        logger.removeHandler(handler)

    assert records == ["before", "after"]


def test_muted_logger_restores_level_even_on_exception() -> None:
    logger = logging.getLogger("hmmlearn.base")
    original_level = logger.level
    try:
        with _muted_logger("hmmlearn.base"):
            assert logger.level == logging.ERROR
            raise ValueError("boom")
    except ValueError:
        pass
    assert logger.level == original_level
