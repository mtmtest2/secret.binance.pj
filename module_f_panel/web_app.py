"""FastAPI monitoring & control panel (Module F).

Runs on ``0.0.0.0:8000`` and is reached directly by IP - no domain, no reverse
proxy, no TLS termination required, which is the point on a small VPS.

Security posture: the panel is **not** hardened for the public internet.  When
``web.api_token`` is set, every mutating ``/api/*`` endpoint requires it via the
``X-API-Token`` header (or a ``token`` field in the JSON body); the read-only
endpoints stay open so the dashboard can poll them.  Bind the port to a private
interface, an SSH tunnel or a firewall allow-list before exposing it.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from jinja2 import DictLoader, Environment, select_autoescape

from config.settings import Settings
from core.logger import LOG_BUFFER, get_logger
from module_f_panel.templates import TEMPLATES

_LOGGER = get_logger(__name__)


@runtime_checkable
class SystemController(Protocol):
    """The slice of the orchestrator the panel is allowed to touch.

    Declaring it as a Protocol keeps Module F free of any import from
    ``main.py``, which is what avoids a circular dependency between the panel and
    the orchestrator that mounts it.
    """

    settings: Settings

    async def status_snapshot(self) -> dict[str, Any]:
        """Full system status for the dashboard."""

    async def recent_audit(
        self, limit: int, symbol: str | None, verdict: str | None
    ) -> list[dict[str, Any]]:
        """Recent audit rows."""

    async def recent_trades(self, limit: int, status_filter: str | None) -> list[dict[str, Any]]:
        """Recent trade rows."""

    async def list_universe_candidates(self, refresh: bool) -> dict[str, Any]:
        """Screened list of every USDT-M perpetual on Binance."""

    async def suggest_universe(self, limit: int | None) -> dict[str, Any]:
        """Top-scoring eligible symbols."""

    async def save_universe(
        self, symbols: Sequence[str], start_setup: bool, operator: str
    ) -> dict[str, Any]:
        """Persist the operator's pair selection."""

    async def setup_status(self) -> dict[str, Any]:
        """Progress of the data-collection and training pipeline."""

    async def start_setup(self, force_retrain: bool) -> dict[str, Any]:
        """Re-run collection and training."""

    async def start_trading(self, mode: str) -> dict[str, Any]:
        """Arm paper or live trading."""

    async def stop_trading(self, flatten: bool) -> dict[str, Any]:
        """Disarm trading."""

    async def set_trading_enabled(self, enabled: bool) -> dict[str, Any]:
        """Pause or resume signal generation."""

    async def set_trading_mode(self, mode: str) -> dict[str, Any]:
        """Switch between paper and live execution."""

    async def engage_kill_switch(self, reason: str) -> dict[str, Any]:
        """Trip the Risk Guard RED and flatten everything."""

    async def reset_risk_guard(self, operator: str) -> dict[str, Any]:
        """Clear a RED latch after manual review."""

    async def ml_diagnostics(self) -> dict[str, Any]:
        """The full ML diagnostic report for the most recent training run."""

    async def ml_diagnostics_markdown(self) -> str:
        """The same report, rendered as human-readable Markdown."""


def build_app(controller: SystemController) -> FastAPI:
    """Construct the FastAPI application bound to ``controller``."""
    settings: Settings = controller.settings
    environment: Environment = Environment(
        loader=DictLoader(TEMPLATES),
        autoescape=select_autoescape(["html"]),
        enable_async=False,
    )

    app = FastAPI(
        title=settings.web.title,
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def render(content_template: str, scripts_template: str, **context: Any) -> HTMLResponse:
        """Render a page by composing the base shell with a content block.

        ``content`` and ``scripts`` are rendered first and injected as markup, so
        the base template's autoescaping does not mangle them.
        """
        from markupsafe import Markup

        content: str = environment.get_template(content_template).render(**context)
        scripts: str = environment.get_template(scripts_template).render(**context)
        html: str = environment.get_template("base.html").render(
            title=settings.web.title,
            content=Markup(content),
            scripts=Markup(scripts),
            **context,
        )
        return HTMLResponse(content=html)

    def authorise(request: Request, payload: dict[str, Any] | None = None) -> None:
        """Enforce the shared secret on mutating endpoints.

        A blank ``web.api_token`` disables the check entirely - convenient for a
        firewalled box, and an explicit decision rather than an oversight.
        """
        expected: str = settings.web.api_token
        if not expected:
            return
        supplied: str = request.headers.get("X-API-Token", "")
        if not supplied and payload:
            supplied = str(payload.get("token", ""))
        if supplied != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API token"
            )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, summary="Operations dashboard")
    async def dashboard() -> HTMLResponse:
        """System status, active positions, model health and the live log."""
        return render("dashboard_content.html", "dashboard_scripts.html")

    @app.get("/universe", response_class=HTMLResponse, summary="Select trading pairs")
    async def universe_page() -> HTMLResponse:
        """Live Binance perpetual list with the screens applied, ticked by the operator."""
        return render(
            "universe_content.html",
            "universe_scripts.html",
            target_count=settings.universe.target_count,
        )

    @app.get("/audit", response_class=HTMLResponse, summary="Decision audit trail")
    async def audit_page() -> HTMLResponse:
        """Every decision the engine made, with the exact rule that fired."""
        return render(
            "audit_content.html", "audit_scripts.html", page_size=settings.web.audit_page_size
        )

    @app.get("/trades", response_class=HTMLResponse, summary="Trade history")
    async def trades_page() -> HTMLResponse:
        """Closed and open trades with fees, funding and realised PnL."""
        return render("trades_content.html", "trades_scripts.html")

    @app.get("/ml-report", response_class=HTMLResponse, summary="ML diagnostic report")
    async def ml_report_page() -> HTMLResponse:
        """Full training diagnostics: dataset health, per-head metrics, data
        quality, features, labels, backtest and the AI-ready summary."""
        return render("ml_report_content.html", "ml_report_scripts.html")

    # ------------------------------------------------------------------
    # Read-only API
    # ------------------------------------------------------------------
    @app.get("/health", summary="Liveness probe")
    async def health() -> JSONResponse:
        """Cheap liveness check for process supervisors."""
        return JSONResponse({"status": "ok"})

    @app.get("/api/status", summary="Full system status")
    async def api_status() -> JSONResponse:
        """Risk guard state, equity, positions, cycle timing and model health."""
        try:
            return JSONResponse(await controller.status_snapshot())
        except Exception as error:  # pragma: no cover - the panel must not 500
            _LOGGER.error("Status snapshot failed: %s", error, exc_info=True)
            raise HTTPException(status_code=500, detail=f"status unavailable: {error}") from error

    @app.get("/api/audit", summary="Recent audit records")
    async def api_audit(
        limit: int = Query(default=100, ge=1, le=1_000),
        symbol: str | None = Query(default=None),
        verdict: str | None = Query(default=None),
    ) -> JSONResponse:
        """Newest audit rows, optionally filtered by symbol and verdict."""
        rows: list[dict[str, Any]] = await controller.recent_audit(limit, symbol, verdict)
        return JSONResponse({"rows": rows, "count": len(rows)})

    @app.get("/api/trades", summary="Recent trades")
    async def api_trades(
        limit: int = Query(default=200, ge=1, le=1_000),
        status_filter: str | None = Query(default=None, alias="status"),
    ) -> JSONResponse:
        """Newest trades, optionally filtered by status."""
        rows: list[dict[str, Any]] = await controller.recent_trades(limit, status_filter)
        return JSONResponse({"rows": rows, "count": len(rows)})

    @app.get("/api/positions", summary="Active positions")
    async def api_positions() -> JSONResponse:
        """Just the open positions slice of the status snapshot."""
        snapshot: dict[str, Any] = await controller.status_snapshot()
        return JSONResponse({"rows": snapshot.get("positions", [])})

    @app.get("/api/logs", summary="Recent log lines")
    async def api_logs(limit: int = Query(default=200, ge=1, le=800)) -> JSONResponse:
        """Tail of the in-memory ring buffer - no filesystem access required."""
        return JSONResponse({"rows": LOG_BUFFER.snapshot(limit)})

    @app.get("/api/logs/download", summary="Download the on-disk log file(s)")
    async def api_logs_download() -> Any:
        """Serve ``core.logger.configure_logging``'s rotating log file(s).

        Unlike ``/api/logs`` (the in-memory ring buffer, capped at the most
        recent 800 formatted lines), this serves the actual on-disk
        ``quant_system.log`` written by the ``RotatingFileHandler`` - the
        full history up to its rotation caps. When rotation has produced
        backups (``quant_system.log.1``, ``.2``, ...) they are all zipped
        together in memory (``io.BytesIO`` / ``zipfile`` - no temp files);
        a single current file is returned as a plain-text attachment.
        """
        log_dir: Path = settings.log_dir
        log_files: list[Path] = sorted(
            (path for path in log_dir.glob("quant_system.log*") if path.is_file()),
            key=lambda path: path.name,
        )
        if not log_files:
            raise HTTPException(
                status_code=404,
                detail=f"no log file found in {log_dir} - is settings.log_dir configured?",
            )

        if len(log_files) == 1:
            return PlainTextResponse(
                log_files[0].read_text(encoding="utf-8", errors="replace"),
                media_type="text/plain",
                headers={"Content-Disposition": f'attachment; filename="{log_files[0].name}"'},
            )

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in log_files:
                archive.write(path, arcname=path.name)
        buffer.seek(0)
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="quant_system_logs.zip"'},
        )

    # ------------------------------------------------------------------
    # ML diagnostic report API
    # ------------------------------------------------------------------
    @app.get("/api/ml/diagnostics", summary="Full ML diagnostic report (JSON)")
    async def api_ml_diagnostics() -> JSONResponse:
        """The structured report for the most recent training run.

        Every section is either real, measured data or the literal string
        ``"NOT_AVAILABLE"`` - never a fabricated value. See ``/ml-report`` for
        the human-readable dashboard built on top of the same data.
        """
        try:
            return JSONResponse(await controller.ml_diagnostics())
        except Exception as error:  # pragma: no cover - the panel must not 500
            _LOGGER.error("ML diagnostics unavailable: %s", error, exc_info=True)
            raise HTTPException(status_code=500, detail=f"diagnostics unavailable: {error}") from error

    @app.get("/api/ml/diagnostics/export.json", summary="Download the full report as JSON")
    async def api_ml_diagnostics_export_json() -> JSONResponse:
        """Same content as ``/api/ml/diagnostics``, offered as a download."""
        report: dict[str, Any] = await controller.ml_diagnostics()
        run_id: str = str(report.get("run", {}).get("run_id", "latest"))
        return JSONResponse(
            report,
            headers={"Content-Disposition": f'attachment; filename="ml_diagnostic_{run_id}.json"'},
        )

    @app.get("/api/ml/diagnostics/export.md", summary="Download the human-readable report")
    async def api_ml_diagnostics_export_markdown() -> Any:
        """Markdown rendering of the same report, offered as a download."""
        markdown: str = await controller.ml_diagnostics_markdown()
        report: dict[str, Any] = await controller.ml_diagnostics()
        run_id: str = str(report.get("run", {}).get("run_id", "latest"))
        return PlainTextResponse(
            markdown,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="ml_diagnostic_{run_id}.md"'},
        )

    @app.get("/api/backtest/trades.csv", summary="Download the diagnostic backtest's trades as CSV")
    async def api_backtest_trades_csv(
        variant: str = Query(
            default="strict",
            pattern="^(strict|relaxed)$",
            description="'strict' (live-configured thresholds) or 'relaxed' (diagnostic-only, "
            "loosened thresholds - see backtest_diagnostic_relaxed in /api/ml/diagnostics).",
        ),
    ) -> Any:
        """Every trade from the most recent diagnostic backtest run, as a CSV.

        Pulls from the same ``report["backtest"]`` / ``report["backtest_diagnostic_relaxed"]``
        trades list already embedded in the full JSON export (``/api/ml/diagnostics/export.json``)
        - this just offers it as a directly-downloadable, spreadsheet-friendly file instead of
        requiring the operator to extract it from the larger report by hand.
        """
        report: dict[str, Any] = await controller.ml_diagnostics()
        section_key: str = "backtest" if variant == "strict" else "backtest_diagnostic_relaxed"
        section: dict[str, Any] = report.get(section_key, {})
        trades: list[dict[str, Any]] = section.get("trades", []) if isinstance(section, dict) else []

        if not trades:
            raise HTTPException(
                status_code=404,
                detail=f"no {variant} backtest trades available yet - run training first",
            )

        run_id: str = str(report.get("run", {}).get("run_id", "latest"))
        csv_text: str = pd.DataFrame(trades).to_csv(index=False)
        return PlainTextResponse(
            csv_text,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="backtest_trades_{variant}_{run_id}.csv"'
                )
            },
        )

    # ------------------------------------------------------------------
    # Universe API
    # ------------------------------------------------------------------
    @app.get("/api/universe/available", summary="Screened perpetual futures pairs")
    async def api_universe_available(
        refresh: bool = Query(default=False, description="Bypass the discovery cache."),
    ) -> JSONResponse:
        """Every active USDT-M perpetual, annotated with the screening metrics."""
        try:
            return JSONResponse(await controller.list_universe_candidates(refresh))
        except Exception as error:
            _LOGGER.error("Universe discovery failed: %s", error)
            raise HTTPException(
                status_code=503, detail=f"could not reach the exchange: {error}"
            ) from error

    @app.get("/api/universe/suggest", summary="Auto-select the best-scoring pairs")
    async def api_universe_suggest(
        limit: int | None = Query(default=None, ge=1, le=200),
    ) -> JSONResponse:
        """Top eligible symbols by screening score; never returns an ineligible one."""
        try:
            return JSONResponse(await controller.suggest_universe(limit))
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/universe/selected", summary="Current saved universe")
    async def api_universe_selected() -> JSONResponse:
        """The pairs the system is configured to trade."""
        snapshot: dict[str, Any] = await controller.status_snapshot()
        return JSONResponse(snapshot.get("universe", {}))

    @app.post("/api/universe/select", summary="Save the pair selection")
    async def api_universe_select(
        request: Request,
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        """Persist the ticked pairs and (by default) start data collection + training.

        Body: ``{"symbols": ["BTC/USDT:USDT", ...], "start_setup": true}``.
        """
        authorise(request, payload)
        raw: Any = payload.get("symbols")
        if not isinstance(raw, list) or not raw:
            raise HTTPException(status_code=400, detail="'symbols' must be a non-empty list")

        start_setup: bool = bool(payload.get("start_setup", True))
        try:
            result: dict[str, Any] = await controller.save_universe(
                [str(item) for item in raw], start_setup, "web-panel"
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return JSONResponse(result)

    # ------------------------------------------------------------------
    # Setup API
    # ------------------------------------------------------------------
    @app.get("/api/setup/status", summary="Data collection & training progress")
    async def api_setup_status() -> JSONResponse:
        """Phase, current step, percentage and any setup error."""
        return JSONResponse(await controller.setup_status())

    @app.post("/api/setup/start", summary="Re-run collection and training")
    async def api_setup_start(
        request: Request,
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        """Backfill any missing candles and retrain.  ``force_retrain`` skips the
        "models are already current" shortcut."""
        authorise(request, payload)
        force: bool = bool(payload.get("force_retrain", False))
        return JSONResponse(await controller.start_setup(force))

    # ------------------------------------------------------------------
    # Control API
    # ------------------------------------------------------------------
    @app.post("/api/trading/start", summary="Arm paper or live trading")
    async def api_trading_start(
        request: Request,
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        """Arm trading.  Body: ``{"mode": "paper"|"live"}`` (default: paper).

        Rejected unless setup finished, the Risk Guard is clear, and - for live -
        all four models are genuinely trained artifacts.
        """
        authorise(request, payload)
        mode: str = str(payload.get("mode", "paper")).lower()
        try:
            return JSONResponse(await controller.start_trading(mode))
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/trading/stop", summary="Disarm trading")
    async def api_trading_stop(
        request: Request,
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        """Stop opening new positions.

        Open positions keep being managed to their own TP/SL unless
        ``{"flatten": true}`` is passed.
        """
        authorise(request, payload)
        return JSONResponse(await controller.stop_trading(bool(payload.get("flatten", False))))


    @app.post("/api/toggle_trading", summary="Pause/resume or switch execution mode")
    async def api_toggle_trading(
        request: Request,
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        """Pause or resume signal generation, and/or switch paper <-> live.

        Body accepts ``{"action": "pause"|"resume"|"toggle"}`` and/or
        ``{"mode": "paper"|"live"}``.  Switching mode always flattens the
        outgoing engine first - carrying positions across engines would leave
        them unmanaged.
        """
        authorise(request, payload)
        result: dict[str, Any] = {}

        action: str = str(payload.get("action", "")).lower()
        if action in {"pause", "resume", "toggle"}:
            snapshot: dict[str, Any] = await controller.status_snapshot()
            enabled: bool = (
                not bool(snapshot.get("trading_enabled", True))
                if action == "toggle"
                else action == "resume"
            )
            result["trading"] = await controller.set_trading_enabled(enabled)

        mode: str = str(payload.get("mode", "")).lower()
        if mode:
            if mode not in {"paper", "live"}:
                raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")
            result["mode"] = await controller.set_trading_mode(mode)

        if not result:
            raise HTTPException(status_code=400, detail="supply an 'action' and/or a 'mode'")
        return JSONResponse(result)

    @app.post("/api/kill_switch", summary="Panic button")
    async def api_kill_switch(
        request: Request,
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        """Trip the Risk Guard RED: cancel every order, close every position, halt."""
        authorise(request, payload)
        reason: str = str(payload.get("reason", "manual kill switch via web panel"))
        _LOGGER.critical("KILL SWITCH requested from the web panel: %s", reason)
        return JSONResponse(await controller.engage_kill_switch(reason))

    @app.post("/api/reset_risk_guard", summary="Clear a RED latch")
    async def api_reset_risk_guard(
        request: Request,
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        """Clear the halt after a human has reviewed why it fired."""
        authorise(request, payload)
        operator: str = str(payload.get("operator", "web-panel"))
        return JSONResponse(await controller.reset_risk_guard(operator))

    return app
