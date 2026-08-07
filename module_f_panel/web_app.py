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

from typing import Any, Protocol, runtime_checkable

from fastapi import Body, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
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

    async def set_trading_enabled(self, enabled: bool) -> dict[str, Any]:
        """Pause or resume signal generation."""

    async def set_trading_mode(self, mode: str) -> dict[str, Any]:
        """Switch between paper and live execution."""

    async def engage_kill_switch(self, reason: str) -> dict[str, Any]:
        """Trip the Risk Guard RED and flatten everything."""

    async def reset_risk_guard(self, operator: str) -> dict[str, Any]:
        """Clear a RED latch after manual review."""


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

    # ------------------------------------------------------------------
    # Control API
    # ------------------------------------------------------------------
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
