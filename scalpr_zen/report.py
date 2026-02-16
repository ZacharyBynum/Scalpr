from __future__ import annotations

import os
from datetime import datetime, timezone

from scalpr_zen.types import BacktestResult, Direction


def _fmt_ns_timestamp(ns: int) -> str:
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_dollars(val: float) -> str:
    if val >= 0:
        return f"+${val:,.2f}"
    return f"-${abs(val):,.2f}"


def format_report(result: BacktestResult, run_timestamp: datetime) -> str:
    lines: list[str] = []
    w = lines.append

    w("=" * 80)
    w("SCALPR ZEN v0.1 — Backtest Report")
    w("=" * 80)
    w("")

    p = result.params
    w(f"Strategy:         {result.strategy_name}")
    w(f"Run timestamp:    {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    w("")

    w("── Parameters " + "─" * 65)
    w(f"Instrument:       {p.get('instrument', 'N/A')}")
    w(f"Point value:      ${p.get('point_value', 0):.2f}")
    w(f"Tick size:        {p.get('tick_size', 0)}")
    w(f"Fast EMA:         {p.get('fast_ema', 'N/A')}")
    w(f"Slow EMA:         {p.get('slow_ema', 'N/A')}")
    w(f"TP (points):      {p.get('tp_points', 0):.2f}")
    w(f"SL (points):      {p.get('sl_points', 0):.2f}")
    w(f"Warmup ticks:     {p.get('warmup_ticks', 'N/A')}")
    w(f"Data range:       {p.get('data_range', 'N/A')}")
    if result.summary:
        w(f"Ticks processed:  {result.summary.total_ticks_processed:,}")
    w("")

    w("── Trade Log " + "─" * 66)
    header = f"{'#':<8}{'Dir':<7}{'Entry Time':<24}{'Entry Px':<12}{'Exit Time':<24}{'Exit Px':<12}{'P&L ($)':<12}{'Exit'}"
    w(header)

    for fill in result.fills:
        dir_str = "LONG" if fill.direction == Direction.LONG else "SHORT"
        entry_t = _fmt_ns_timestamp(fill.entry_time)
        exit_t = _fmt_ns_timestamp(fill.exit_time)
        pnl_str = _fmt_dollars(fill.pnl_dollars)
        w(
            f"{fill.trade_number:<8}"
            f"{dir_str:<7}"
            f"{entry_t:<24}"
            f"{fill.entry_price:<12.2f}"
            f"{exit_t:<24}"
            f"{fill.exit_price:<12.2f}"
            f"{pnl_str:<12}"
            f"{fill.exit_reason.value}"
        )

    w("")
    w("── Summary " + "─" * 68)

    if result.summary:
        s = result.summary
        w(f"Total trades:        {s.total_trades}")
        w(f"Win rate:            {s.win_rate:.1%} ({s.winning_trades}W / {s.losing_trades}L)")
        w(f"Total P&L:           {_fmt_dollars(s.total_pnl_dollars)}")
        w(f"Profit factor:       {s.profit_factor:.3f}")
        w(f"Avg win / Avg loss:  {_fmt_dollars(s.avg_win)} / {_fmt_dollars(s.avg_loss)}")
        w(f"Max drawdown:        {_fmt_dollars(s.max_drawdown_dollars)}")
        w(f"Max consec W/L:      {s.max_consecutive_wins} / {s.max_consecutive_losses}")
        w("")
        w("── Validation " + "─" * 65)
        w(f"Expectancy:          {_fmt_dollars(s.expectancy_per_trade)} / trade        target: > $0")
        w(f"t-statistic:         {s.t_stat:.2f}                      target: ≥ 2.0")
        w(f"p-value:             {s.p_value:.6f}                  target: < 0.05")
        w(f"SQN:                 {s.sqn:.2f}                      target: ≥ 2.0")
        w(f"Days profitable:     {s.pct_days_profitable:.1%}                     target: ≥ 50%")
    elif result.error:
        w(f"ERROR: {result.error}")

    w("=" * 80)
    return "\n".join(lines)


def write_report(result: BacktestResult, output_dir: str = "results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now(tz=timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    safe_name = result.strategy_name.lower().replace(" ", "_")
    filename = f"{safe_name}_{timestamp_str}.txt"
    filepath = os.path.join(output_dir, filename)

    content = format_report(result, now)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath
