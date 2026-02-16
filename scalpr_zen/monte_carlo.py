from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np

from scalpr_zen.types import (
    BacktestResult,
    MonteCarloResult,
    MonteCarloStats,
)


def _simulate_equity_curves(
    pnl_array: np.ndarray, n_simulations: int, rng: np.random.Generator
) -> np.ndarray:
    n_trades = len(pnl_array)
    curves = np.empty((n_simulations, n_trades + 1), dtype=np.float64)
    curves[:, 0] = 0.0
    for i in range(n_simulations):
        shuffled = rng.permutation(pnl_array)
        curves[i, 1:] = np.cumsum(shuffled)
    return curves


def _compute_max_drawdowns(curves: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(curves, axis=1)
    drawdowns = running_max - curves
    return np.max(drawdowns, axis=1)


def _compute_stats(
    curves: np.ndarray, original_pnl: np.ndarray
) -> MonteCarloStats:
    n_simulations = curves.shape[0]
    final_pnls = curves[:, -1]
    max_dds = _compute_max_drawdowns(curves)

    original_cum = np.concatenate(([0.0], np.cumsum(original_pnl)))
    original_final = float(original_cum[-1])
    original_peak = np.maximum.accumulate(original_cum)
    original_max_dd = float(np.max(original_peak - original_cum))

    return MonteCarloStats(
        n_simulations=n_simulations,
        probability_of_profit=float(np.mean(final_pnls > 0)),
        median_final_pnl=float(np.percentile(final_pnls, 50)),
        final_pnl_5th=float(np.percentile(final_pnls, 5)),
        final_pnl_25th=float(np.percentile(final_pnls, 25)),
        final_pnl_75th=float(np.percentile(final_pnls, 75)),
        final_pnl_95th=float(np.percentile(final_pnls, 95)),
        median_max_drawdown=float(np.percentile(max_dds, 50)),
        max_drawdown_5th=float(np.percentile(max_dds, 5)),
        max_drawdown_25th=float(np.percentile(max_dds, 25)),
        max_drawdown_75th=float(np.percentile(max_dds, 75)),
        max_drawdown_95th=float(np.percentile(max_dds, 95)),
        original_final_pnl=original_final,
        original_max_drawdown=original_max_dd,
    )


def _downsample(arr: np.ndarray, max_points: int) -> list[float]:
    if len(arr) <= max_points:
        return arr.tolist()
    indices = np.linspace(0, len(arr) - 1, max_points, dtype=int)
    return arr[indices].tolist()


def _extract_percentile_curves(
    curves: np.ndarray, original_pnl: np.ndarray, max_points: int = 2000
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    p5 = np.percentile(curves, 5, axis=0)
    p25 = np.percentile(curves, 25, axis=0)
    p50 = np.percentile(curves, 50, axis=0)
    p75 = np.percentile(curves, 75, axis=0)
    p95 = np.percentile(curves, 95, axis=0)
    original = np.concatenate(([0.0], np.cumsum(original_pnl)))

    return (
        _downsample(p5, max_points),
        _downsample(p25, max_points),
        _downsample(p50, max_points),
        _downsample(p75, max_points),
        _downsample(p95, max_points),
        _downsample(original, max_points),
    )


def run_monte_carlo(
    result: BacktestResult, n_simulations: int = 1000, seed: int | None = None
) -> MonteCarloResult:
    if not result.success or not result.fills:
        return MonteCarloResult(
            success=False,
            error="Backtest has no fills to simulate",
            strategy_name=result.strategy_name,
            params=result.params,
            stats=None,
            curve_5th=[],
            curve_25th=[],
            curve_50th=[],
            curve_75th=[],
            curve_95th=[],
            original_curve=[],
        )

    if len(result.fills) < 2:
        return MonteCarloResult(
            success=False,
            error="Need at least 2 trades for Monte Carlo simulation",
            strategy_name=result.strategy_name,
            params=result.params,
            stats=None,
            curve_5th=[],
            curve_25th=[],
            curve_50th=[],
            curve_75th=[],
            curve_95th=[],
            original_curve=[],
        )

    pnl_array = np.array([f.pnl_dollars for f in result.fills], dtype=np.float64)
    rng = np.random.default_rng(seed)

    curves = _simulate_equity_curves(pnl_array, n_simulations, rng)
    stats = _compute_stats(curves, pnl_array)
    c5, c25, c50, c75, c95, orig = _extract_percentile_curves(curves, pnl_array)

    return MonteCarloResult(
        success=True,
        error=None,
        strategy_name=result.strategy_name,
        params=result.params,
        stats=stats,
        curve_5th=c5,
        curve_25th=c25,
        curve_50th=c50,
        curve_75th=c75,
        curve_95th=c95,
        original_curve=orig,
    )


def _fmt_dollars(val: float) -> str:
    if val >= 0:
        return f"${val:,.0f}"
    return f"-${abs(val):,.0f}"


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>scalpr_zen — Monte Carlo</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0d1117;
            color: #c9d1d9;
            font-family: "Cascadia Code", "Fira Code", "JetBrains Mono", monospace;
            font-size: 14px;
            padding: 24px;
        }}

        .header {{
            margin-bottom: 20px;
            border-bottom: 1px solid #21262d;
            padding-bottom: 16px;
        }}
        .header h1 {{
            font-size: 20px;
            font-weight: 600;
            color: #e6edf3;
            margin-bottom: 6px;
        }}
        .header .subtitle {{
            font-size: 13px;
            color: #7d8590;
        }}
        .header .subtitle span {{
            color: #c9d1d9;
        }}

        .stats {{
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 12px 18px;
            flex: 1;
            min-width: 140px;
        }}
        .stat-card .label {{
            font-size: 11px;
            color: #7d8590;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .stat-card .value {{
            font-size: 20px;
            font-weight: 600;
            color: #e6edf3;
        }}
        .stat-card .value.positive {{ color: #3fb950; }}
        .stat-card .value.negative {{ color: #f85149; }}

        .section {{
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 20px;
        }}
        .section h2 {{
            font-size: 14px;
            font-weight: 600;
            color: #e6edf3;
            margin-bottom: 12px;
        }}
        .chart-container {{
            position: relative;
            width: 100%;
            height: 420px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 8px 16px;
            text-align: right;
            border-bottom: 1px solid #21262d;
        }}
        th {{
            color: #7d8590;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        th:first-child, td:first-child {{
            text-align: left;
        }}
        td {{
            color: #c9d1d9;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>SCALPR ZEN v0.1 — Monte Carlo</h1>
    <div class="subtitle">
        <span>{strategy_name}</span>
        &nbsp;|&nbsp; <span>{instrument}</span>
        &nbsp;|&nbsp; <span>{n_simulations:,}</span> simulations
        &nbsp;|&nbsp; <span>{n_trades:,}</span> trades
    </div>
</div>

<div class="stats">
    <div class="stat-card">
        <div class="label">P(Profit)</div>
        <div class="value {p_profit_class}">{p_profit}</div>
    </div>
    <div class="stat-card">
        <div class="label">Median P&amp;L</div>
        <div class="value {median_pnl_class}">{median_pnl}</div>
    </div>
    <div class="stat-card">
        <div class="label">90% CI Range</div>
        <div class="value">{ci_range}</div>
    </div>
    <div class="stat-card">
        <div class="label">Median Max DD</div>
        <div class="value negative">{median_dd}</div>
    </div>
    <div class="stat-card">
        <div class="label">Worst DD (95th)</div>
        <div class="value negative">{worst_dd}</div>
    </div>
</div>

<div class="section">
    <h2>Equity Curve Confidence Bands
        <span style="font-size:12px;color:#7d8590;font-weight:400">—
            <span style="color:rgba(88,166,255,0.3)">5th–95th</span> |
            <span style="color:rgba(88,166,255,0.6)">25th–75th</span> |
            <span style="color:#58a6ff">Median</span> |
            <span style="color:#3fb950">Original</span>
        </span>
    </h2>
    <div class="chart-container">
        <canvas id="mc-chart"></canvas>
    </div>
</div>

<div class="section">
    <h2>Confidence Intervals</h2>
    <table>
        <thead>
            <tr>
                <th>Percentile</th>
                <th>Final P&amp;L</th>
                <th>Max Drawdown</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>5th</td><td>{tbl_pnl_5th}</td><td>{tbl_dd_5th}</td></tr>
            <tr><td>25th</td><td>{tbl_pnl_25th}</td><td>{tbl_dd_25th}</td></tr>
            <tr><td>50th (Median)</td><td>{tbl_pnl_50th}</td><td>{tbl_dd_50th}</td></tr>
            <tr><td>75th</td><td>{tbl_pnl_75th}</td><td>{tbl_dd_75th}</td></tr>
            <tr><td>95th</td><td>{tbl_pnl_95th}</td><td>{tbl_dd_95th}</td></tr>
            <tr style="border-top:2px solid #30363d">
                <td style="color:#3fb950">Original</td>
                <td style="color:#3fb950">{tbl_pnl_orig}</td>
                <td style="color:#3fb950">{tbl_dd_orig}</td>
            </tr>
        </tbody>
    </table>
</div>

<script>
const MC_DATA = {{
    labels: {labels_json},
    curve_5th: {curve_5th_json},
    curve_25th: {curve_25th_json},
    curve_50th: {curve_50th_json},
    curve_75th: {curve_75th_json},
    curve_95th: {curve_95th_json},
    original: {original_json}
}};

(function() {{
    const ctx = document.getElementById('mc-chart').getContext('2d');

    new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: MC_DATA.labels,
            datasets: [
                {{
                    label: '95th percentile',
                    data: MC_DATA.curve_95th,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(88, 166, 255, 0.08)',
                    fill: '+4',
                    pointRadius: 0,
                    tension: 0.1,
                    order: 5,
                }},
                {{
                    label: '75th percentile',
                    data: MC_DATA.curve_75th,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(88, 166, 255, 0.15)',
                    fill: '+2',
                    pointRadius: 0,
                    tension: 0.1,
                    order: 4,
                }},
                {{
                    label: '50th percentile (Median)',
                    data: MC_DATA.curve_50th,
                    borderColor: '#58a6ff',
                    borderWidth: 2,
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0,
                    tension: 0.1,
                    order: 2,
                }},
                {{
                    label: '25th percentile',
                    data: MC_DATA.curve_25th,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0,
                    tension: 0.1,
                    order: 4,
                }},
                {{
                    label: '5th percentile',
                    data: MC_DATA.curve_5th,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0,
                    tension: 0.1,
                    order: 5,
                }},
                {{
                    label: 'Original',
                    data: MC_DATA.original,
                    borderColor: '#3fb950',
                    borderWidth: 2,
                    borderDash: [6, 3],
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0,
                    tension: 0.1,
                    order: 1,
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{
                mode: 'index',
                intersect: false,
            }},
            plugins: {{
                legend: {{ display: false }},
                tooltip: {{
                    backgroundColor: '#1c2128',
                    borderColor: '#30363d',
                    borderWidth: 1,
                    titleFont: {{ family: '"Cascadia Code", monospace', size: 12 }},
                    bodyFont: {{ family: '"Cascadia Code", monospace', size: 12 }},
                    callbacks: {{
                        title: function(items) {{
                            return 'Trade #' + items[0].label;
                        }},
                        label: function(ctx) {{
                            const val = ctx.parsed.y;
                            const sign = val >= 0 ? '+' : '';
                            return ctx.dataset.label + ': ' + sign + '$' + val.toLocaleString(undefined, {{maximumFractionDigits: 0}});
                        }}
                    }},
                    filter: function(item) {{
                        return item.dataset.borderColor !== 'transparent';
                    }}
                }}
            }},
            scales: {{
                x: {{
                    grid: {{ color: '#21262d' }},
                    ticks: {{
                        color: '#7d8590',
                        font: {{ family: '"Cascadia Code", monospace' }},
                        maxTicksLimit: 10,
                    }},
                    title: {{
                        display: true,
                        text: 'Trade #',
                        color: '#7d8590',
                        font: {{ family: '"Cascadia Code", monospace', size: 12 }}
                    }}
                }},
                y: {{
                    grid: {{ color: '#21262d' }},
                    ticks: {{
                        color: '#7d8590',
                        font: {{ family: '"Cascadia Code", monospace' }},
                        callback: function(v) {{
                            const sign = v >= 0 ? '' : '-';
                            return sign + '$' + Math.abs(v).toLocaleString();
                        }}
                    }},
                    title: {{
                        display: true,
                        text: 'Cumulative P&L ($)',
                        color: '#7d8590',
                        font: {{ family: '"Cascadia Code", monospace', size: 12 }}
                    }}
                }}
            }}
        }}
    }});
}})();
</script>

</body>
</html>
"""


def _build_labels(n_points: int) -> list[int]:
    if n_points <= 2000:
        return list(range(n_points))
    return np.linspace(0, n_points - 1, 2000, dtype=int).tolist()


def write_monte_carlo_html(
    mc_result: MonteCarloResult, output_dir: str = "results"
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    s = mc_result.stats
    n_trades = len(mc_result.original_curve)
    labels = list(range(len(mc_result.curve_50th)))

    # Map labels to actual trade numbers if downsampled
    if n_trades > 2000:
        labels = np.linspace(0, n_trades - 1, len(mc_result.curve_50th), dtype=int).tolist()

    p_profit_pct = f"{s.probability_of_profit:.1%}"
    p_profit_class = "positive" if s.probability_of_profit >= 0.5 else "negative"

    html = _HTML_TEMPLATE.format(
        strategy_name=mc_result.strategy_name,
        instrument=mc_result.params.get("instrument", "N/A"),
        n_simulations=s.n_simulations,
        n_trades=n_trades - 1,  # subtract the initial 0
        p_profit=p_profit_pct,
        p_profit_class=p_profit_class,
        median_pnl=_fmt_dollars(s.median_final_pnl),
        median_pnl_class="positive" if s.median_final_pnl >= 0 else "negative",
        ci_range=f"{_fmt_dollars(s.final_pnl_5th)} to {_fmt_dollars(s.final_pnl_95th)}",
        median_dd=_fmt_dollars(s.median_max_drawdown),
        worst_dd=_fmt_dollars(s.max_drawdown_95th),
        # Table values
        tbl_pnl_5th=_fmt_dollars(s.final_pnl_5th),
        tbl_pnl_25th=_fmt_dollars(s.final_pnl_25th),
        tbl_pnl_50th=_fmt_dollars(s.median_final_pnl),
        tbl_pnl_75th=_fmt_dollars(s.final_pnl_75th),
        tbl_pnl_95th=_fmt_dollars(s.final_pnl_95th),
        tbl_dd_5th=_fmt_dollars(s.max_drawdown_5th),
        tbl_dd_25th=_fmt_dollars(s.max_drawdown_25th),
        tbl_dd_50th=_fmt_dollars(s.median_max_drawdown),
        tbl_dd_75th=_fmt_dollars(s.max_drawdown_75th),
        tbl_dd_95th=_fmt_dollars(s.max_drawdown_95th),
        tbl_pnl_orig=_fmt_dollars(s.original_final_pnl),
        tbl_dd_orig=_fmt_dollars(s.original_max_drawdown),
        # Chart data
        labels_json=labels,
        curve_5th_json=_round_list(mc_result.curve_5th),
        curve_25th_json=_round_list(mc_result.curve_25th),
        curve_50th_json=_round_list(mc_result.curve_50th),
        curve_75th_json=_round_list(mc_result.curve_75th),
        curve_95th_json=_round_list(mc_result.curve_95th),
        original_json=_round_list(mc_result.original_curve),
    )

    now = datetime.now(tz=timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    safe_name = mc_result.strategy_name.lower().replace(" ", "_")
    filename = f"{safe_name}_monte_carlo_{timestamp_str}.html"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath


def _round_list(values: list[float]) -> list[float]:
    return [round(v, 2) for v in values]
