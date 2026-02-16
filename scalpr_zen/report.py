from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

from scalpr_zen.types import BacktestResult, Direction

CST = timezone(timedelta(hours=-6))


def _fmt_ns_timestamp(ns: int) -> str:
    dt = datetime.fromtimestamp(ns / 1e9, tz=CST)
    return dt.strftime("%Y-%m-%d %I:%M:%S %p")


def _fmt_dollars(val: float) -> str:
    if val >= 0:
        return f"+${val:,.2f}"
    return f"-${abs(val):,.2f}"


def _utc_to_cst_12h(utc_hour: float) -> str:
    """Convert a UTC decimal hour to CST 12-hour string."""
    cst_hour = (utc_hour - 6) % 24
    h = int(cst_hour)
    m = int((cst_hour % 1) * 60)
    period = "AM" if h < 12 else "PM"
    display_h = h % 12 or 12
    return f"{display_h}:{m:02d} {period}"


def _grade(value: float, target: float, higher_is_better: bool) -> str:
    if higher_is_better:
        if value > target * 1.1:
            return "PASS"
        elif value >= target:
            return "NEUTRAL"
        else:
            return "FAIL"
    else:
        if value < target * 0.9:
            return "PASS"
        elif value <= target:
            return "NEUTRAL"
        else:
            return "FAIL"


def format_report(result: BacktestResult, run_timestamp: datetime) -> str:
    lines: list[str] = []
    w = lines.append

    w("=" * 80)
    w("SCALPR v0.2 — Backtest Report")
    w("=" * 80)
    w("")

    p = result.params
    run_cst = run_timestamp.astimezone(CST)
    w(f"Strategy:         {result.strategy_name}")
    w(f"Run timestamp:    {run_cst.strftime('%Y-%m-%d %I:%M:%S %p')} CST")
    w("")

    # Strategy description
    fast = p.get('fast_ema', '?')
    slow = p.get('slow_ema', '?')
    tp = p.get('tp_points', 0)
    sl = p.get('sl_points', 0)
    start_utc = p.get('entry_start_utc')
    end_utc = p.get('entry_end_utc')
    time_desc = ""
    if start_utc is not None and end_utc is not None:
        start_cst = _utc_to_cst_12h(start_utc)
        end_cst = _utc_to_cst_12h(end_utc)
        time_desc = f" Entries are restricted to {start_cst}–{end_cst} CST."
    w("── Description " + "─" * 64)
    w(f"A long entry is triggered when the {fast}-period EMA crosses above the "
      f"{slow}-period EMA; a short entry is triggered on the inverse crossover. "
      f"Each trade targets a {tp:.0f}-point take-profit and is protected by a "
      f"{sl:.0f}-point stop-loss, with the stop checked before the target on each "
      f"tick (worst-case assumption).{time_desc} Only one position is open at a "
      f"time; new signals are ignored until the current trade exits.")
    w("")

    w("── Parameters " + "─" * 65)
    w(f"Initial capital:  ${p.get('initial_capital', 0):,.2f}")
    w(f"Instrument:       {p.get('instrument', 'N/A')}")
    w(f"Point value:      ${p.get('point_value', 0):.2f}")
    w(f"Tick size:        {p.get('tick_size', 0)}")
    w(f"Fast EMA:         {fast}")
    w(f"Slow EMA:         {slow}")
    w(f"TP (points):      {tp:.2f}")
    w(f"SL (points):      {sl:.2f}")
    if start_utc is not None and end_utc is not None:
        w(f"Entry window:     {start_cst}–{end_cst} CST")
    w(f"Warmup ticks:     {p.get('warmup_ticks', 'N/A')}")
    w(f"Data range:       {p.get('data_range', 'N/A')}")
    if result.summary:
        w(f"Ticks processed:  {result.summary.total_ticks_processed:,}")
    w("")

    if result.summary:
        s = result.summary
        initial = p.get('initial_capital', 0)

        w("── Summary " + "─" * 68)
        pnl_grade = "PASS" if s.total_pnl_dollars > 0 else "FAIL"
        w(f"Total trades:        {s.total_trades}")
        w(f"Win rate:            {s.win_rate:.1%} ({s.winning_trades}W / {s.losing_trades}L)    [{_grade(s.win_rate, 0.40, True)}]")
        w(f"Total P&L:           {_fmt_dollars(s.total_pnl_dollars)}    [{pnl_grade}]")
        pf_grade = _grade(s.profit_factor, 1.5, True)
        w(f"Profit factor:       {s.profit_factor:.3f}    [{pf_grade}]")
        w(f"Avg win / Avg loss:  {_fmt_dollars(s.avg_win)} / {_fmt_dollars(s.avg_loss)}")
        if initial > 0:
            dd_pct = abs(s.max_drawdown_dollars) / initial * 100
            dd_grade = _grade(dd_pct, 20.0, False)
            w(f"Max drawdown:        {_fmt_dollars(s.max_drawdown_dollars)} ({dd_pct:.1f}%)    [{dd_grade}]")
        else:
            w(f"Max drawdown:        {_fmt_dollars(s.max_drawdown_dollars)}")
        w(f"Max consec W/L:      {s.max_consecutive_wins} / {s.max_consecutive_losses}")
        if initial > 0:
            roi = s.total_pnl_dollars / initial * 100
            roi_grade = _grade(roi, 20.0, True)
            w(f"ROI:                 {roi:.1f}%    [{roi_grade}]")
        sharpe_grade = _grade(s.sharpe_ratio, 1.0, True)
        w(f"Sharpe ratio:        {s.sharpe_ratio:.2f}    [{sharpe_grade}]")
        w("")
        w("── Validation " + "─" * 65)
        exp_grade = "PASS" if s.expectancy_per_trade > 0 else "FAIL"
        w(f"Expectancy:          {_fmt_dollars(s.expectancy_per_trade)} / trade    [{exp_grade}]")
        t_grade = _grade(s.t_stat, 2.0, True)
        w(f"t-statistic:         {s.t_stat:.2f}    [{t_grade}]")
        p_grade = _grade(s.p_value, 0.05, False)
        w(f"p-value:             {s.p_value:.6f}    [{p_grade}]")
        sqn_grade = _grade(s.sqn, 2.0, True)
        w(f"SQN:                 {s.sqn:.2f}    [{sqn_grade}]")
        days_grade = _grade(s.pct_days_profitable, 0.50, True)
        w(f"Days profitable:     {s.pct_days_profitable:.1%}    [{days_grade}]")
        w("")
    elif result.error:
        w(f"ERROR: {result.error}")

    w("── Trade Log " + "─" * 66)
    header = f"{'#':<8}{'Dir':<7}{'Entry Time':<26}{'Entry Px':<12}{'Exit Time':<26}{'Exit Px':<12}{'P&L ($)':<12}{'Exit'}"
    w(header)

    for fill in result.fills:
        dir_str = "LONG" if fill.direction == Direction.LONG else "SHORT"
        entry_t = _fmt_ns_timestamp(fill.entry_time)
        exit_t = _fmt_ns_timestamp(fill.exit_time)
        pnl_str = _fmt_dollars(fill.pnl_dollars)
        w(
            f"{fill.trade_number:<8}"
            f"{dir_str:<7}"
            f"{entry_t:<26}"
            f"{fill.entry_price:<12.2f}"
            f"{exit_t:<26}"
            f"{fill.exit_price:<12.2f}"
            f"{pnl_str:<12}"
            f"{fill.exit_reason.value}"
        )

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
