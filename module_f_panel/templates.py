"""Self-contained Jinja2 templates for the monitoring panel.

The panel is deliberately a single dependency-free bundle: Tailwind arrives from
a CDN, but a compact inline stylesheet keeps the page fully legible when the VPS
has no outbound internet access.  No build step, no static-file directory, no
node_modules - just ``uvicorn`` on ``IP:8000``.
"""

from __future__ import annotations

from typing import Final

_BASE: Final[
    str
] = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{{ title }}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    :root { color-scheme: dark; }
    body { background:#0b1020; color:#e6ebf5;
           font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin:0; }
    a { color:#7dd3fc; text-decoration:none; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { padding:6px 8px; border-bottom:1px solid #1e293b; text-align:left; white-space:nowrap; }
    th { color:#94a3b8; font-weight:600; text-transform:uppercase; font-size:10px; letter-spacing:.06em; }
    .card { background:#111827; border:1px solid #1f2937; border-radius:10px; padding:14px; }
    .muted { color:#94a3b8; }
    .pos { color:#34d399; } .neg { color:#f87171; } .warn { color:#fbbf24; }
    .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700; }
    .grid { display:grid; gap:14px; }
    .scroll { overflow-x:auto; }
  </style>
</head>
<body class="min-h-screen">
  <header class="border-b border-slate-800 bg-slate-900/60 sticky top-0 z-10 backdrop-blur">
    <div class="max-w-[1600px] mx-auto px-5 py-3 flex items-center justify-between flex-wrap gap-3">
      <div class="flex items-center gap-3">
        <span class="text-lg font-bold tracking-tight">AI QUANT &middot; BINANCE USDT-M</span>
        <span class="muted text-xs">5m perpetual futures engine</span>
      </div>
      <nav class="flex items-center gap-4 text-sm">
        <a href="/">Dashboard</a>
        <a href="/audit">Audit</a>
        <a href="/trades">Trades</a>
        <a href="/api/status">API</a>
        <span id="clock" class="muted text-xs"></span>
      </nav>
    </div>
  </header>
  <main class="max-w-[1600px] mx-auto px-5 py-5 grid gap-5">
    {{ content }}
  </main>
  <script>
    function fmt(x, d) { return (x === null || x === undefined || isNaN(x)) ? '-' : Number(x).toFixed(d === undefined ? 2 : d); }
    function pct(x, d) { return (x === null || x === undefined || isNaN(x)) ? '-' : (Number(x) * 100).toFixed(d === undefined ? 2 : d) + '%'; }
    function tick() { document.getElementById('clock').textContent = new Date().toISOString().replace('T',' ').slice(0,19) + ' UTC'; }
    setInterval(tick, 1000); tick();
  </script>
  {{ scripts }}
</body>
</html>
"""

_DASHBOARD_CONTENT: Final[
    str
] = """
<section class="grid" style="grid-template-columns:repeat(auto-fit,minmax(200px,1fr));">
  <div class="card"><div class="muted text-xs">RISK GUARD</div>
    <div id="guard-state" class="text-2xl font-bold mt-1">-</div>
    <div id="guard-reason" class="muted text-xs mt-1"></div></div>
  <div class="card"><div class="muted text-xs">MODE</div>
    <div id="mode" class="text-2xl font-bold mt-1">-</div>
    <div id="trading-enabled" class="muted text-xs mt-1"></div></div>
  <div class="card"><div class="muted text-xs">EQUITY (USDT)</div>
    <div id="equity" class="text-2xl font-bold mt-1">-</div>
    <div id="upnl" class="muted text-xs mt-1"></div></div>
  <div class="card"><div class="muted text-xs">DAILY DRAWDOWN</div>
    <div id="dd" class="text-2xl font-bold mt-1">-</div>
    <div id="dd-limit" class="muted text-xs mt-1"></div></div>
  <div class="card"><div class="muted text-xs">OPEN POSITIONS</div>
    <div id="npos" class="text-2xl font-bold mt-1">-</div>
    <div id="streak" class="muted text-xs mt-1"></div></div>
  <div class="card"><div class="muted text-xs">LAST CYCLE</div>
    <div id="cycle" class="text-lg font-bold mt-1">-</div>
    <div id="cycle-detail" class="muted text-xs mt-1"></div></div>
</section>

<section class="card">
  <div class="flex items-center justify-between flex-wrap gap-3">
    <div class="font-bold">CONTROLS</div>
    <div class="flex items-center gap-2 flex-wrap">
      <input id="token" type="password" placeholder="API token (if set)"
             class="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs"/>
      <button onclick="post('/api/toggle_trading', {action:'pause'})"
              class="bg-amber-600 hover:bg-amber-500 rounded px-3 py-1 text-xs font-bold">PAUSE</button>
      <button onclick="post('/api/toggle_trading', {action:'resume'})"
              class="bg-emerald-700 hover:bg-emerald-600 rounded px-3 py-1 text-xs font-bold">RESUME</button>
      <button onclick="post('/api/toggle_trading', {mode:'paper'})"
              class="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-xs font-bold">PAPER</button>
      <button onclick="confirmLive()"
              class="bg-indigo-700 hover:bg-indigo-600 rounded px-3 py-1 text-xs font-bold">LIVE</button>
      <button onclick="post('/api/reset_risk_guard', {})"
              class="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs font-bold">RESET GUARD</button>
      <button onclick="killSwitch()"
              class="bg-red-700 hover:bg-red-600 rounded px-3 py-1 text-xs font-bold">KILL SWITCH</button>
    </div>
  </div>
  <div id="control-result" class="muted text-xs mt-2"></div>
</section>

<section class="grid" style="grid-template-columns:repeat(auto-fit,minmax(460px,1fr));">
  <div class="card">
    <div class="font-bold mb-2">ACTIVE POSITIONS</div>
    <div class="scroll"><table>
      <thead><tr><th>Symbol</th><th>Side</th><th>Lev</th><th>Entry</th><th>Mark</th>
        <th>TP</th><th>SL</th><th>Liq</th><th>uPnL</th><th>Tier</th></tr></thead>
      <tbody id="positions"><tr><td colspan="10" class="muted">no open positions</td></tr></tbody>
    </table></div>
  </div>
  <div class="card">
    <div class="font-bold mb-2">MODEL &amp; SYSTEM HEALTH</div>
    <div class="scroll"><table>
      <thead><tr><th>Component</th><th>State</th></tr></thead>
      <tbody id="health"></tbody>
    </table></div>
  </div>
</section>

<section class="card">
  <div class="font-bold mb-2">LATEST DECISIONS</div>
  <div class="scroll"><table>
    <thead><tr><th>Time</th><th>Symbol</th><th>Verdict</th><th>Rule</th>
      <th>P(long)</th><th>P(short)</th><th>Entry</th><th>Lev</th><th>Regime</th><th>Reason</th></tr></thead>
    <tbody id="decisions"></tbody>
  </table></div>
</section>

<section class="card">
  <div class="font-bold mb-2">LIVE LOG</div>
  <pre id="logs" class="text-[11px] leading-relaxed muted whitespace-pre-wrap max-h-80 overflow-y-auto"></pre>
</section>
"""

_DASHBOARD_SCRIPTS: Final[
    str
] = """
<script>
const GUARD_COLOR = {GREEN:'#34d399', YELLOW:'#fbbf24', RED:'#f87171'};

function token() { return document.getElementById('token').value.trim(); }

async function post(url, body) {
  const el = document.getElementById('control-result');
  try {
    const res = await fetch(url, {
      method:'POST',
      headers:{'Content-Type':'application/json','X-API-Token':token()},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    el.textContent = (res.ok ? 'OK: ' : 'ERROR: ') + JSON.stringify(data);
    el.className = res.ok ? 'pos text-xs mt-2' : 'neg text-xs mt-2';
    refresh();
  } catch (err) { el.textContent = 'Request failed: ' + err; el.className = 'neg text-xs mt-2'; }
}

function killSwitch() {
  if (confirm('KILL SWITCH: cancel all orders, close all positions and halt trading?')) {
    post('/api/kill_switch', {reason:'panel kill switch'});
  }
}
function confirmLive() {
  if (confirm('Switch to LIVE trading with real funds?')) { post('/api/toggle_trading', {mode:'live'}); }
}

function row(cells) { return '<tr>' + cells.map(c => '<td>' + c + '</td>').join('') + '</tr>'; }

async function refresh() {
  let s;
  try { s = await (await fetch('/api/status')).json(); }
  catch (err) { document.getElementById('guard-state').textContent = 'UNREACHABLE'; return; }

  const guard = s.risk_guard || {};
  const acct = s.account || {};
  const g = document.getElementById('guard-state');
  g.textContent = guard.state || '-';
  g.style.color = GUARD_COLOR[guard.state] || '#e6ebf5';
  document.getElementById('guard-reason').textContent = guard.halt_reason || 'nominal';

  document.getElementById('mode').textContent = (s.trading_mode || '-').toUpperCase();
  document.getElementById('trading-enabled').textContent =
    s.trading_enabled ? 'trading enabled' : 'trading PAUSED';
  document.getElementById('equity').textContent = fmt(acct.equity);
  const up = document.getElementById('upnl');
  up.textContent = 'unrealised ' + fmt(acct.unrealized_pnl) + ' | balance ' + fmt(acct.balance);
  up.className = (acct.unrealized_pnl || 0) >= 0 ? 'pos text-xs mt-1' : 'neg text-xs mt-1';

  const dd = document.getElementById('dd');
  dd.textContent = pct(guard.daily_drawdown_pct);
  dd.style.color = (guard.daily_drawdown_pct || 0) >= (guard.daily_drawdown_limit || 1)
    ? '#f87171' : '#e6ebf5';
  document.getElementById('dd-limit').textContent =
    'limit ' + pct(guard.daily_drawdown_limit) + ' | peak DD ' + pct(guard.total_drawdown_pct);
  document.getElementById('npos').textContent = (s.positions || []).length;
  document.getElementById('streak').textContent =
    'loss streak ' + (guard.consecutive_losses || 0) + '/' + (guard.consecutive_losses_limit || '-') +
    ' | trades today ' + (guard.trades_today || 0);
  document.getElementById('cycle').textContent = s.last_cycle_at || 'never';
  document.getElementById('cycle-detail').textContent =
    'symbols ok ' + (s.last_cycle_symbols_ok || 0) + ' | decisions ' + (s.last_cycle_decisions || 0) +
    ' | ' + fmt(s.last_cycle_duration_s) + 's';

  const positions = s.positions || [];
  document.getElementById('positions').innerHTML = positions.length ? positions.map(p => row([
    p.symbol, p.side, p.leverage + 'x', fmt(p.entry_price, 6), fmt(p.mark_price, 6),
    fmt(p.take_profit, 6), fmt(p.stop_loss, 6), fmt(p.liquidation_price, 6),
    '<span class="' + (p.unrealized_pnl >= 0 ? 'pos' : 'neg') + '">' + fmt(p.unrealized_pnl, 4) +
      ' (' + pct(p.unrealized_pnl_pct) + ')</span>',
    p.risk_tier
  ])).join('') : '<tr><td colspan="10" class="muted">no open positions</td></tr>';

  const health = s.health || {};
  document.getElementById('health').innerHTML = Object.keys(health).map(k =>
    row([k, typeof health[k] === 'boolean'
      ? '<span class="' + (health[k] ? 'pos' : 'warn') + '">' + (health[k] ? 'READY' : 'FALLBACK') + '</span>'
      : health[k]])).join('');

  try {
    const audit = await (await fetch('/api/audit?limit=25')).json();
    document.getElementById('decisions').innerHTML = (audit.rows || []).map(r => row([
      (r.created_at || '').slice(11, 19),
      r.symbol,
      '<span class="pill" style="background:' +
        (r.verdict === 'EXECUTE' ? '#065f46' : r.verdict === 'BLOCKED' ? '#7f1d1d' : '#1f2937') +
        '">' + r.verdict + '</span>',
      r.rule_triggered, pct(r.prob_long, 1), pct(r.prob_short, 1),
      pct(r.entry_probability, 1), r.recommended_leverage + 'x', r.hmm_regime,
      '<span class="muted">' + (r.reason || '').slice(0, 90) + '</span>'
    ])).join('');
  } catch (err) { /* audit table is non-critical */ }

  try {
    const logs = await (await fetch('/api/logs?limit=120')).json();
    document.getElementById('logs').textContent =
      (logs.rows || []).map(l => l.timestamp + '  ' + l.level.padEnd(8) + ' ' + l.message).join('\\n');
  } catch (err) { /* logs are non-critical */ }
}

refresh();
setInterval(refresh, 5000);
</script>
"""

_AUDIT_CONTENT: Final[
    str
] = """
<section class="card">
  <div class="flex items-center justify-between flex-wrap gap-3 mb-3">
    <div class="font-bold">AUDIT TRAIL &mdash; every decision, including NO_TRADE</div>
    <div class="flex gap-2">
      <input id="f-symbol" placeholder="symbol filter"
             class="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs"/>
      <select id="f-verdict" class="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs">
        <option value="">all verdicts</option>
        <option value="EXECUTE">EXECUTE</option>
        <option value="NO_TRADE">NO_TRADE</option>
        <option value="BLOCKED">BLOCKED</option>
      </select>
      <button onclick="loadAudit()"
              class="bg-sky-700 hover:bg-sky-600 rounded px-3 py-1 text-xs font-bold">APPLY</button>
    </div>
  </div>
  <div class="scroll"><table>
    <thead><tr>
      <th>Time</th><th>Symbol</th><th>Verdict</th><th>Rule</th><th>Reason</th>
      <th>P(L)</th><th>P(S)</th><th>P(NT)</th><th>Conf</th><th>Entry p</th>
      <th>TP%</th><th>SL%</th><th>Trail%</th><th>Lev</th><th>Alloc</th>
      <th>Tier</th><th>Regime</th><th>GARCH</th><th>Vol pct</th><th>KAMA</th><th>FDI</th><th>ms</th>
    </tr></thead>
    <tbody id="audit-rows"></tbody>
  </table></div>
  <div id="audit-count" class="muted text-xs mt-2"></div>
</section>
"""

_AUDIT_SCRIPTS: Final[
    str
] = """
<script>
function row(cells) { return '<tr>' + cells.map(c => '<td>' + c + '</td>').join('') + '</tr>'; }

async function loadAudit() {
  const symbol = document.getElementById('f-symbol').value.trim();
  const verdict = document.getElementById('f-verdict').value;
  const params = new URLSearchParams({limit: '{{ page_size }}'});
  if (symbol) params.set('symbol', symbol);
  if (verdict) params.set('verdict', verdict);

  const data = await (await fetch('/api/audit?' + params.toString())).json();
  const rows = data.rows || [];
  document.getElementById('audit-rows').innerHTML = rows.map(r => row([
    (r.created_at || '').replace('T', ' ').slice(0, 19),
    r.symbol,
    '<span class="pill" style="background:' +
      (r.verdict === 'EXECUTE' ? '#065f46' : r.verdict === 'BLOCKED' ? '#7f1d1d' : '#1f2937') +
      '">' + r.verdict + '</span>',
    r.rule_triggered,
    '<span class="muted">' + (r.reason || '') + '</span>',
    pct(r.prob_long, 1), pct(r.prob_short, 1), pct(r.prob_no_trade, 1),
    pct(r.direction_confidence, 1), pct(r.entry_probability, 1),
    pct(r.take_profit_pct, 2), pct(r.stop_loss_pct, 2), pct(r.trailing_trigger_pct, 2),
    r.recommended_leverage + 'x', pct(r.capital_allocation_pct, 1),
    r.risk_tier, r.hmm_regime, fmt(r.garch_volatility, 6), pct(r.garch_vol_percentile, 0),
    fmt(r.kama_slope, 5), fmt(r.fdi, 3), fmt(r.latency_ms, 1)
  ])).join('');
  document.getElementById('audit-count').textContent = rows.length + ' record(s)';
}
loadAudit();
setInterval(loadAudit, 15000);
</script>
"""

_TRADES_CONTENT: Final[
    str
] = """
<section class="card">
  <div class="font-bold mb-3">TRADE HISTORY</div>
  <div class="scroll"><table>
    <thead><tr><th>Opened</th><th>Closed</th><th>Symbol</th><th>Mode</th><th>Side</th><th>Status</th>
      <th>Lev</th><th>Qty</th><th>Entry</th><th>Exit</th><th>Fees</th><th>Funding</th>
      <th>PnL</th><th>Reason</th></tr></thead>
    <tbody id="trade-rows"></tbody>
  </table></div>
  <div id="trade-summary" class="muted text-xs mt-2"></div>
</section>
"""

_TRADES_SCRIPTS: Final[
    str
] = """
<script>
function row(cells) { return '<tr>' + cells.map(c => '<td>' + c + '</td>').join('') + '</tr>'; }

async function loadTrades() {
  const data = await (await fetch('/api/trades?limit=200')).json();
  const rows = data.rows || [];
  document.getElementById('trade-rows').innerHTML = rows.map(t => row([
    (t.opened_at || '').replace('T', ' ').slice(0, 19),
    (t.closed_at || '').replace('T', ' ').slice(0, 19),
    t.symbol, t.mode, t.side, t.status, t.leverage + 'x',
    fmt(t.quantity, 6), fmt(t.entry_price, 6), fmt(t.exit_price, 6),
    fmt(t.fees_paid, 4), fmt(t.funding_paid, 4),
    '<span class="' + (t.realized_pnl >= 0 ? 'pos' : 'neg') + '">' + fmt(t.realized_pnl, 4) + '</span>',
    t.close_reason || '-'
  ])).join('');

  const closed = rows.filter(t => t.status !== 'OPEN');
  const wins = closed.filter(t => t.realized_pnl > 0).length;
  const net = closed.reduce((a, t) => a + (t.realized_pnl || 0), 0);
  document.getElementById('trade-summary').textContent =
    closed.length + ' closed | ' + wins + ' wins (' +
    (closed.length ? (100 * wins / closed.length).toFixed(1) : '0.0') +
    '%) | net ' + net.toFixed(4) + ' USDT';
}
loadTrades();
setInterval(loadTrades, 15000);
</script>
"""

#: Template registry consumed by the Jinja2 ``DictLoader`` in ``web_app.py``.
TEMPLATES: Final[dict[str, str]] = {
    "base.html": _BASE,
    "dashboard_content.html": _DASHBOARD_CONTENT,
    "dashboard_scripts.html": _DASHBOARD_SCRIPTS,
    "audit_content.html": _AUDIT_CONTENT,
    "audit_scripts.html": _AUDIT_SCRIPTS,
    "trades_content.html": _TRADES_CONTENT,
    "trades_scripts.html": _TRADES_SCRIPTS,
}
