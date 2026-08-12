"""The panel's inline JavaScript must survive the trip through Python and Jinja.

The templates are Python string literals that emit JavaScript.  A backslash
escape written for the *browser* (``\\n``, ``\\'``) is consumed by **Python**
first unless it is doubled, so the browser receives a real newline or a bare
quote - which terminates the JS string literal early and kills the entire
``<script>`` block with ``Uncaught SyntaxError``.  Nothing on the page works
after that, and the failure is invisible from the server side.

Both bugs this file guards against shipped and broke the dashboard and the
universe page in exactly that way, so the check is a scanner rather than a
spot assertion: any string literal that spans a newline fails, wherever it is.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import pytest
from jinja2 import DictLoader, Environment, select_autoescape
from markupsafe import Markup

from config.settings import get_settings
from module_f_panel.templates import TEMPLATES

PAGES: dict[str, tuple[str, str]] = {
    "dashboard": ("dashboard_content.html", "dashboard_scripts.html"),
    "universe": ("universe_content.html", "universe_scripts.html"),
    "trades": ("trades_content.html", "trades_scripts.html"),
    "audit": ("audit_content.html", "audit_scripts.html"),
}


def render_page(name: str) -> str:
    """Render one page exactly as ``web_app.render`` does, Markup included."""
    content, scripts = PAGES[name]
    context = {"target_count": get_settings().universe.target_count}
    environment = Environment(
        loader=DictLoader(TEMPLATES), autoescape=select_autoescape(["html"])
    )
    return environment.get_template("base.html").render(
        content=Markup(environment.get_template(content).render(**context)),
        scripts=Markup(environment.get_template(scripts).render(**context)),
        **context,
    )


def script_blocks(html: str) -> list[str]:
    """Every inline ``<script>`` body (external ``src=`` tags are skipped)."""
    import re

    pattern = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.DOTALL)
    return [match.group(1) for match in pattern.finditer(html)]


def unterminated_string_lines(source: str) -> list[tuple[int, str]]:
    """Return every line on which a JS string literal is left open.

    A hand-rolled scan rather than a regex: it tracks quote state and honours
    backslash escapes, which is precisely the distinction the bugs turned on.
    Template literals may legally span lines and are therefore not flagged.
    """
    offenders: list[tuple[int, str]] = []
    quote: str | None = None
    escaped: bool = False
    line_number: int = 1
    line_start: int = 1

    for character in source:
        if character == "\n":
            if quote in {"'", '"'}:
                offenders.append((line_start, source.splitlines()[line_start - 1]))
                quote = None
            line_number += 1
            escaped = False
            continue
        if quote is None:
            if character in {"'", '"', "`"}:
                quote = character
                line_start = line_number
            continue
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            quote = None
    return offenders


# ---------------------------------------------------------------------------
# The scanner must actually catch the two bugs that shipped.
# ---------------------------------------------------------------------------
def test_scanner_detects_a_newline_inside_a_string_literal() -> None:
    """A Python-consumed ``\\n`` leaves a literal newline mid-string."""
    broken = "if (!confirm('Save?\nData collection starts now.')) return;"
    assert unterminated_string_lines(broken)


def test_node_parser_detects_a_collapsed_quote_escape() -> None:
    """A Python-consumed ``\\'`` closes the literal one quote too early.

    Deliberately *not* asserted against the scanner: this failure leaves the
    quotes balanced on a single line, so it is a semantic break rather than an
    unterminated literal, and only a real parser sees it.  The scanner covers
    newline-split literals; this and
    :func:`test_symbol_click_handler_is_correctly_quoted` cover collapsed
    escapes, the latter without needing node.
    """
    node = shutil.which("node")
    if node is None:  # pragma: no cover - CI without node
        pytest.skip("node is not installed")

    broken = "x = '<input onchange=\"toggle('' + r.symbol + '', this.checked)\"/>';"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(broken)
        path = Path(handle.name)
    try:
        result = subprocess.run(
            [node, "--check", str(path)], capture_output=True, text=True, timeout=30
        )
    finally:
        path.unlink(missing_ok=True)
    assert result.returncode != 0
    assert "Unexpected string" in result.stderr


def test_scanner_accepts_correctly_escaped_javascript() -> None:
    """Properly doubled escapes must not trip it."""
    fine = (
        "if (!confirm('Save?\\n\\nStarts now.')) return;\n"
        "x = '<input onchange=\"toggle(\\'' + s + '\\', true)\"/>';\n"
        "y = `a template\nliteral may span lines`;\n"
    )
    assert unterminated_string_lines(fine) == []


# ---------------------------------------------------------------------------
# The real templates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("page", sorted(PAGES))
def test_inline_scripts_have_no_unterminated_string_literals(page: str) -> None:
    """No emitted string literal may be split by a raw newline."""
    for index, block in enumerate(script_blocks(render_page(page))):
        offenders = unterminated_string_lines(block)
        assert not offenders, f"{page} script block {index}: {offenders}"


@pytest.mark.parametrize("page", sorted(PAGES))
def test_inline_scripts_parse_as_javascript(page: str) -> None:
    """Parse every block with a real JS engine when one is available."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - CI without node
        pytest.skip("node is not installed")

    for index, block in enumerate(script_blocks(render_page(page))):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(block)
            path = Path(handle.name)
        try:
            # `--check` parses without executing, so no fetch/DOM stubs are needed.
            result = subprocess.run(
                [node, "--check", str(path)], capture_output=True, text=True, timeout=30
            )
        finally:
            path.unlink(missing_ok=True)
        assert result.returncode == 0, f"{page} block {index}: {result.stderr}"


def test_symbol_click_handler_is_correctly_quoted() -> None:
    """The universe checkbox passes the symbol as a quoted JS string.

    Symbols contain ``/`` and ``:`` (``BTC/USDT:USDT``), so the handler only
    works if the inner quotes survive into the browser as escapes.
    """
    html = render_page("universe")
    assert "onchange=\\\"toggle(\\'" in html or "toggle(\\'" in html
    assert "toggle('' + r.symbol" not in html


def test_confirm_dialogs_keep_their_line_breaks() -> None:
    """The two-line confirm prompts must emit ``\\n``, not a real newline."""
    universe = render_page("universe")
    dashboard = render_page("dashboard")
    assert "pair(s)?\\n\\nData collection" in universe
    assert "REAL FUNDS?\\n\\nAny open paper positions" in dashboard


def test_api_token_input_lives_inside_a_form() -> None:
    """A bare password field outside a form trips the browser's credential warning."""
    for page in ("dashboard", "universe"):
        html = render_page(page)
        assert 'id="token"' in html
        token_index = html.index('id="token"')
        preceding = html[:token_index]
        assert preceding.rindex("<form") > preceding.rindex("</form>") if "</form>" in preceding else True
        assert "<form" in preceding
