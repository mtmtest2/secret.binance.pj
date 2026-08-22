"""Tasks 3 & 4 - downloadable backtest trades CSV and log file(s) from the
web panel.

Uses ``starlette.testclient.TestClient`` (httpx-backed) to exercise
``module_f_panel.web_app.build_app`` end-to-end against a minimal fake
controller, rather than calling the route closures directly (they are
defined inline inside ``build_app`` and are not otherwise reachable).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from module_f_panel.web_app import build_app


class _FakeController:
    """Minimal stand-in for main.TradingSystem - only the pieces the routes
    under test actually call. Not a full SystemController implementation;
    fine at runtime since build_app never isinstance-checks its argument.
    """

    def __init__(self, settings: Settings, report: dict[str, Any] | None = None) -> None:
        self.settings = settings
        self._report = report if report is not None else {"status": "NOT_AVAILABLE"}

    async def ml_diagnostics(self) -> dict[str, Any]:
        return self._report

    async def ml_diagnostics_markdown(self) -> str:
        return "# fake report"


def _client(tmp_path: Path, report: dict[str, Any] | None = None) -> TestClient:
    settings = Settings(log_dir=tmp_path)
    controller = _FakeController(settings, report)
    app = build_app(controller)
    return TestClient(app)


# ------------------------------------------------------------------
# /api/backtest/trades.csv
# ------------------------------------------------------------------

def test_backtest_trades_csv_strict_variant_downloads_csv(tmp_path: Path) -> None:
    report = {
        "run": {"run_id": "abc123"},
        "backtest": {
            "trades": [
                {"symbol": "BTC/USDT:USDT", "realized_pnl": 1.5, "close_reason": "TAKE_PROFIT"},
                {"symbol": "ETH/USDT:USDT", "realized_pnl": -0.5, "close_reason": "STOP_LOSS"},
            ]
        },
    }
    client = _client(tmp_path, report)
    response = client.get("/api/backtest/trades.csv", params={"variant": "strict"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="backtest_trades_strict_abc123.csv"' in response.headers["content-disposition"]
    body = response.text
    assert "BTC/USDT:USDT" in body
    assert "realized_pnl" in body


def test_backtest_trades_csv_relaxed_variant_reads_the_relaxed_section(tmp_path: Path) -> None:
    report = {
        "run": {"run_id": "xyz789"},
        "backtest": {"trades": []},
        "backtest_diagnostic_relaxed": {
            "trades": [{"symbol": "SOL/USDT:USDT", "realized_pnl": 2.0}],
        },
    }
    client = _client(tmp_path, report)
    response = client.get("/api/backtest/trades.csv", params={"variant": "relaxed"})

    assert response.status_code == 200
    assert "SOL/USDT:USDT" in response.text
    assert "backtest_trades_relaxed_xyz789.csv" in response.headers["content-disposition"]


def test_backtest_trades_csv_404_when_no_trades(tmp_path: Path) -> None:
    client = _client(tmp_path, {"run": {"run_id": "none"}, "backtest": {"trades": []}})
    response = client.get("/api/backtest/trades.csv", params={"variant": "strict"})
    assert response.status_code == 404


def test_backtest_trades_csv_rejects_unknown_variant(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/backtest/trades.csv", params={"variant": "bogus"})
    assert response.status_code == 422


# ------------------------------------------------------------------
# /api/logs/download
# ------------------------------------------------------------------

def test_logs_download_single_file_as_plain_text(tmp_path: Path) -> None:
    (tmp_path / "quant_system.log").write_text("hello world\nsecond line\n", encoding="utf-8")
    client = _client(tmp_path)

    response = client.get("/api/logs/download")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert 'filename="quant_system.log"' in response.headers["content-disposition"]
    assert "hello world" in response.text


def test_logs_download_zips_rotated_backups(tmp_path: Path) -> None:
    (tmp_path / "quant_system.log").write_text("current\n", encoding="utf-8")
    (tmp_path / "quant_system.log.1").write_text("rotated-1\n", encoding="utf-8")
    (tmp_path / "quant_system.log.2").write_text("rotated-2\n", encoding="utf-8")
    client = _client(tmp_path)

    response = client.get("/api/logs/download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert 'filename="quant_system_logs.zip"' in response.headers["content-disposition"]

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(archive.namelist())
    assert names == {"quant_system.log", "quant_system.log.1", "quant_system.log.2"}
    assert archive.read("quant_system.log.1").decode("utf-8") == "rotated-1\n"


def test_logs_download_404_when_no_log_file_exists(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/logs/download")
    assert response.status_code == 404
