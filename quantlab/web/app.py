"""Web dashboard for Quant Agent factor factory."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project-root relative paths
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parents[2]
_LIBRARY_PATH = _PROJECT_DIR / "assistant_data" / "memory" / "factor_discovery" / "factor_library.json"
_RUNS_PATH = _PROJECT_DIR / "data" / "scheduler" / "daily_runs.json"
_ALERT_PATHS = [
    _PROJECT_DIR / "assistant_data" / "alerts.json",
    _PROJECT_DIR / "assistant_data" / "alerts" / "alert_log.json",
]

# ---------------------------------------------------------------------------
# Severity sort order (lower = higher priority)
# ---------------------------------------------------------------------------
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# Active statuses (not DRAFT/REJECTED/ARCHIVED/RETIRED)
_ACTIVE_STATUSES = {"candidate", "observe", "paper", "pilot", "live", "approved"}


def _load_json(path: Path) -> Any:
    """Load a JSON file, returning None when missing or malformed."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _get_factors() -> list[dict[str, Any]]:
    """Return factor summaries from the persistent store."""
    data = _load_json(_LIBRARY_PATH)
    if data is None:
        return []
    entries = data.get("entries", []) or []
    result: list[dict[str, Any]] = []
    for entry in entries:
        spec = entry.get("factor_spec", {})
        report = entry.get("latest_report", {})
        scorecard = report.get("scorecard", {})

        # Derive created_at from panel file modification time
        created_at = ""
        panel_path = entry.get("panel_snapshot_path", "")
        if panel_path:
            panel_file = Path(panel_path)
            if panel_file.exists():
                try:
                    mtime = panel_file.stat().st_mtime
                    created_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                except OSError:
                    pass

        result.append({
            "factor_id": spec.get("factor_id", ""),
            "name": spec.get("name", ""),
            "status": spec.get("status", "draft"),
            "ic_mean": scorecard.get("ic_mean"),
            "rank_ic_mean": scorecard.get("rank_ic_mean"),
            "ic_ir": scorecard.get("ic_ir"),
            "direction": spec.get("direction", "unknown"),
            "family": spec.get("family", ""),
            "created_at": created_at,
        })
    return result


def _get_run_records(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent run records."""
    data = _load_json(_RUNS_PATH)
    if data is None or not isinstance(data, list):
        return []
    return data[-limit:]


def _get_alerts() -> list[dict[str, Any]]:
    """Return recent alerts sorted by severity."""
    all_alerts: list[dict[str, Any]] = []
    for path in _ALERT_PATHS:
        data = _load_json(path)
        if isinstance(data, list) and data:
            all_alerts.extend(data)
            break  # use the first file that has data

    # Sort by severity (critical first, then warning, then info), then by timestamp desc
    _SEV = _SEVERITY_ORDER
    all_alerts.sort(
        key=lambda a: (_SEV.get(a.get("level", "info"), 99), a.get("timestamp", "")),
        reverse=False,
    )
    return all_alerts


def _count_active_factors(factors: list[dict[str, Any]]) -> int:
    return sum(1 for f in factors if f["status"] in _ACTIVE_STATUSES)


def _stages_completed_from_run(record: dict[str, Any]) -> list[str]:
    """Infer completed stages from a run record."""
    stage_keys = [
        "data_refresh", "decay_monitor", "evolution", "oos_validation",
        "combination", "screening", "paper_trading", "governance",
    ]
    completed: list[str] = []
    for key in stage_keys:
        stage = record.get(key, {})
        if isinstance(stage, dict) and stage.get("status") == "success":
            completed.append(key)
    # delivery_reports is a list, treat non-empty as success
    reports = record.get("delivery_reports", [])
    if reports:
        completed.append("delivery_reports")
    return completed


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app() -> "Flask":
    """Create and configure the Flask dashboard application."""
    try:
        from flask import Flask, jsonify, render_template_string
    except ImportError:
        raise ImportError(
            "Flask is not installed. Install it with: pip install flask"
        )

    app = Flask(__name__)

    # -- Routes ----------------------------------------------------------

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/api/status")
    def api_status():
        factors = _get_factors()
        runs = _get_run_records(limit=1)
        alerts = _get_alerts()

        last_run_time = ""
        if runs:
            last_run_time = runs[-1].get("end_time", "") or runs[-1].get("start_time", "")

        return jsonify({
            "factor_count": len(factors),
            "active_factors": _count_active_factors(factors),
            "pipeline_runs": len(_get_run_records(limit=999)),
            "recent_alerts": len(alerts),
            "last_run_time": last_run_time,
        })

    @app.route("/api/factors")
    def api_factors():
        factors = _get_factors()
        return jsonify(factors)

    @app.route("/api/runs")
    def api_runs():
        runs = _get_run_records(limit=20)
        # Augment with derived fields for easier consumption
        for r in runs:
            r["stages_completed"] = _stages_completed_from_run(r)
        return jsonify(runs)

    @app.route("/api/alerts")
    def api_alerts():
        alerts = _get_alerts()
        # Return last 50 alerts
        return jsonify(alerts[:50])

    return app


# ---------------------------------------------------------------------------
# Inline HTML dashboard (dark theme, no external deps)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quant Agent - Factor Factory Dashboard</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #0d1117; color: #c9d1d9; line-height: 1.5;
  }
  .header {
    background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 24px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .header h1 { font-size: 20px; font-weight: 600; color: #58a6ff; }
  .header .refresh-badge { font-size: 12px; color: #8b949e; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px;
  }
  .card .label { font-size: 13px; color: #8b949e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { font-size: 28px; font-weight: 700; color: #f0f6fc; }
  .card .sub { font-size: 12px; color: #8b949e; margin-top: 4px; }
  .section { margin-bottom: 32px; }
  .section h2 {
    font-size: 16px; font-weight: 600; color: #f0f6fc; margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 1px solid #30363d;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; color: #8b949e; font-weight: 500; border-bottom: 1px solid #30363d; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
  tr:hover { background: #1c2128; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;
  }
  .badge-critical { background: rgba(255,68,68,0.15); color: #ff4444; }
  .badge-warning { background: rgba(255,170,0,0.15); color: #ffaa00; }
  .badge-info { background: rgba(68,170,255,0.15); color: #44aaff; }
  .badge-success { background: rgba(63,185,80,0.15); color: #3fb950; }
  .badge-draft { background: rgba(139,148,158,0.15); color: #8b949e; }
  .status-icon { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .status-icon.ok { background: #3fb950; }
  .status-icon.warn { background: #ffaa00; }
  .status-icon.fail { background: #ff4444; }
  .empty-state {
    text-align: center; padding: 40px 20px; color: #484f58; font-size: 15px;
  }
  .ic-positive { color: #3fb950; }
  .ic-negative { color: #ff4444; }
  .stage-tags { display: flex; flex-wrap: wrap; gap: 4px; }
  .stage-tag {
    display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px;
    background: rgba(63,185,80,0.12); color: #3fb950;
  }
  .last-update { font-size: 12px; color: #484f58; text-align: center; padding: 16px 0; }
</style>
</head>
<body>

<div class="header">
  <h1>Factor Factory Dashboard</h1>
  <span class="refresh-badge" id="refreshBadge">auto-refresh 10s</span>
</div>

<div class="container">

  <!-- Section 1: Factor Library Overview -->
  <div class="section">
    <h2>Factor Library Overview</h2>

    <div class="cards">
      <div class="card">
        <div class="label">Total Factors</div>
        <div class="value" id="factorCount">-</div>
      </div>
      <div class="card">
        <div class="label">Active Factors</div>
        <div class="value" id="activeCount">-</div>
        <div class="sub">status != draft/rejected/archived/retired</div>
      </div>
      <div class="card">
        <div class="label">Pipeline Runs</div>
        <div class="value" id="pipelineRuns">-</div>
        <div class="sub">total executions</div>
      </div>
      <div class="card">
        <div class="label">Recent Alerts</div>
        <div class="value" id="recentAlerts">-</div>
      </div>
    </div>

    <div id="topIcSection">
      <h3 style="font-size:14px; color:#8b949e; margin-bottom:8px;">Top IC Factors</h3>
      <table>
        <thead>
          <tr><th>Factor</th><th>Status</th><th>Rank IC Mean</th><th>IC IR</th><th>Direction</th></tr>
        </thead>
        <tbody id="topIcBody"><tr><td colspan="5" class="empty-state">Loading...</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- Section 2: Pipeline Status -->
  <div class="section">
    <h2>Pipeline Status</h2>
    <div id="pipelineSection">
      <div class="empty-state">Loading...</div>
    </div>
  </div>

  <!-- Section 3: Alerts -->
  <div class="section">
    <h2>Alerts</h2>
    <table>
      <thead>
        <tr><th>Severity</th><th>Time</th><th>Title</th><th>Message</th><th>Source</th></tr>
      </thead>
      <tbody id="alertsBody"><tr><td colspan="5" class="empty-state">Loading...</td></tr></tbody>
    </table>
  </div>

  <div class="last-update" id="lastUpdate"></div>
</div>

<script>
function badgeClass(level) {
  if (level === 'critical') return 'badge badge-critical';
  if (level === 'warning' || level === 'warn') return 'badge badge-warning';
  if (level === 'info') return 'badge badge-info';
  return 'badge';
}

function statusBadge(status) {
  var s = (status || 'draft').toLowerCase();
  var map = {
    live: 'badge badge-success', pilot: 'badge badge-success',
    paper: 'badge badge-info', approved: 'badge badge-info',
    observe: 'badge badge-warning', candidate: 'badge badge-warning',
    draft: 'badge badge-draft', rejected: 'badge badge-critical',
    archived: 'badge badge-draft', retired: 'badge badge-draft'
  };
  return map[s] || 'badge badge-draft';
}

function icClass(val) {
  if (val == null) return '';
  return val > 0 ? 'ic-positive' : 'ic-negative';
}

function formatIc(val) {
  if (val == null) return '-';
  return val.toFixed(4);
}

function refresh() {
  Promise.all([
    fetch('/api/status').then(function(r) { return r.json(); }),
    fetch('/api/factors').then(function(r) { return r.json(); }),
    fetch('/api/runs').then(function(r) { return r.json(); }),
    fetch('/api/alerts').then(function(r) { return r.json(); })
  ]).then(function(results) {
    var status = results[0], factors = results[1], runs = results[2], alerts = results[3];

    // Status cards
    document.getElementById('factorCount').textContent = status.factor_count;
    document.getElementById('activeCount').textContent = status.active_factors;
    document.getElementById('pipelineRuns').textContent = status.pipeline_runs;
    document.getElementById('recentAlerts').textContent = status.recent_alerts;

    // Top IC factors (top 10 by abs rank_ic_mean)
    var sorted = factors.slice().sort(function(a, b) {
      var ia = Math.abs(a.rank_ic_mean || 0), ib = Math.abs(b.rank_ic_mean || 0);
      return ib - ia;
    }).slice(0, 10);

    var topIcHtml = '';
    if (sorted.length === 0) {
      topIcHtml = '<tr><td colspan="5" class="empty-state">因子库为空</td></tr>';
    } else {
      sorted.forEach(function(f) {
        topIcHtml += '<tr>' +
          '<td><strong>' + esc(f.name || f.factor_id) + '</strong></td>' +
          '<td><span class="' + statusBadge(f.status) + '">' + esc(f.status) + '</span></td>' +
          '<td class="' + icClass(f.rank_ic_mean) + '">' + formatIc(f.rank_ic_mean) + '</td>' +
          '<td class="' + icClass(f.ic_ir) + '">' + formatIc(f.ic_ir) + '</td>' +
          '<td>' + esc(f.direction || '-') + '</td>' +
          '</tr>';
      });
    }
    document.getElementById('topIcBody').innerHTML = topIcHtml;

    // Pipeline status
    var pipelineHtml = '';
    if (runs.length === 0) {
      pipelineHtml = '<div class="empty-state">No pipeline runs recorded</div>';
    } else {
      var last = runs[runs.length - 1];
      var statusIcon = 'ok';
      if (last.status === 'failed') statusIcon = 'fail';
      else if (last.status === 'partial') statusIcon = 'warn';

      pipelineHtml += '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">' +
        '<div class="card" style="flex:1;min-width:200px;">' +
        '<div class="label">Last Run</div>' +
        '<div class="value" style="font-size:16px;">' +
        '<span class="status-icon ' + statusIcon + '"></span>' + esc(last.run_date || '-') +
        '</div>' +
        '<div class="sub">Status: ' + esc(last.status) + '</div>' +
        '</div>' +
        '<div class="card" style="flex:1;min-width:200px;">' +
        '<div class="label">Start / End</div>' +
        '<div class="value" style="font-size:16px;">' + esc((last.start_time || '').substring(11, 19)) +
        ' ~ ' + esc((last.end_time || '').substring(11, 19)) + '</div>' +
        '<div class="sub">Next: scheduled daily 18:30</div>' +
        '</div>' +
        '</div>';

      var stages = last.stages_completed || [];
      if (stages.length > 0) {
        pipelineHtml += '<div style="margin-top:8px;"><span style="color:#8b949e;font-size:12px;">Completed stages: </span>' +
          '<div class="stage-tags">';
        stages.forEach(function(s) {
          pipelineHtml += '<span class="stage-tag">' + esc(s) + '</span>';
        });
        pipelineHtml += '</div></div>';
      }
    }
    document.getElementById('pipelineSection').innerHTML = pipelineHtml;

    // Alerts table
    var alertsHtml = '';
    if (alerts.length === 0) {
      alertsHtml = '<tr><td colspan="5" class="empty-state">No alerts</td></tr>';
    } else {
      alerts.forEach(function(a) {
        alertsHtml += '<tr>' +
          '<td><span class="' + badgeClass(a.level) + '">' + esc(a.level || 'info') + '</span></td>' +
          '<td style="white-space:nowrap;">' + esc((a.timestamp || '').substring(0, 19)) + '</td>' +
          '<td>' + esc(a.title || '') + '</td>' +
          '<td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(a.message || '') + '</td>' +
          '<td>' + esc(a.source || '') + '</td>' +
          '</tr>';
      });
    }
    document.getElementById('alertsBody').innerHTML = alertsHtml;

    // Update timestamp
    document.getElementById('lastUpdate').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
  }).catch(function(err) {
    document.getElementById('lastUpdate').textContent = 'Update failed: ' + err;
  });
}

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the Flask dashboard server."""
    try:
        app = create_app()
    except ImportError as exc:
        print(f"ERROR: {exc}")
        print("Install Flask with: pip install flask")
        raise SystemExit(1) from exc

    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
