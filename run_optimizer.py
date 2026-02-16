"""scalpr_zen — sweep TP/SL parameters with signals precomputed once."""

import csv
import itertools
import os
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from scalpr_zen.engine import EngineConfig, precompute_signals, run_summary_from_signals
from scalpr_zen.types import InstrumentSpec

CST = timezone(timedelta(hours=-6))

# ═══════════════════════════════════════════════════════
# FIXED PARAMETERS (same as run_backtest.py)
# ═══════════════════════════════════════════════════════
CACHE_PATH = "cache/nq_ticks.npz"
INSTRUMENT_SYMBOL = "NQ"
TICK_SIZE = 0.25
POINT_VALUE = 20.00

FAST_EMA_PERIOD = 2000
SLOW_EMA_PERIOD = 8000

INITIAL_CAPITAL = 50000.0
COMMISSION_PER_TRADE = 4.12
SLIPPAGE_TICKS = 1.0

ENTRY_START_UTC = 14.5
ENTRY_END_UTC = 16.5

# ═══════════════════════════════════════════════════════
# PARAMETER GRID  (start, stop_inclusive, step)
# ═══════════════════════════════════════════════════════
TP_LONG_RANGE = (5.0, 40.0, 5.0)
SL_LONG_RANGE = (10.0, 60.0, 2.5)
TP_SHORT_RANGE = (5.0, 40.0, 5.0)
SL_SHORT_RANGE = (10.0, 60.0, 2.5)
OBJECTIVE = "sharpe"  # column to sort by (descending)
TOP_N = 20
# ═══════════════════════════════════════════════════════


def _frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range."""
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 4))
        v += step
    return vals


def _d(val: float) -> str:
    return f"+${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"


def _cst12(utc_hour: float) -> str:
    h = (utc_hour - 6) % 24
    period = "AM" if int(h) < 12 else "PM"
    return f"{int(h) % 12 or 12}:{int((h % 1) * 60):02d} {period}"


def _range_str(rng: tuple[float, float, float]) -> str:
    vals = _frange(*rng)
    if len(vals) == 1:
        return f"{vals[0]:.1f} (fixed)"
    return f"{rng[0]:.1f} → {rng[1]:.1f}, step {rng[2]:.1f} ({len(vals)} values)"


def format_optimize_report(
    rows: list[dict],
    objective: str,
    n_signals: int,
    total_ticks: int,
    n_trading_days: int,
    precompute_time: float,
    sweep_time: float,
    run_timestamp: datetime,
) -> str:
    L: list[str] = []
    w = L.append
    run_cst = run_timestamp.astimezone(CST)

    # Header
    w("SCALPR — Optimization Report")
    w(f"Version: v0.2")
    w(f"Run: {run_cst.strftime('%Y-%m-%d %I:%M:%S %p')} CST")
    w(f"Objective: {objective} (descending)")
    w("")

    # Fixed parameters
    w("FIXED PARAMETERS")
    w(f"Instrument: {INSTRUMENT_SYMBOL} | Point value: ${POINT_VALUE:.2f} | Tick: {TICK_SIZE}")
    w(f"EMA: {FAST_EMA_PERIOD}/{SLOW_EMA_PERIOD}")
    su, eu = ENTRY_START_UTC, ENTRY_END_UTC
    w(f"Entry window: {_cst12(su)}–{_cst12(eu)} CST")
    w(f"Capital: ${INITIAL_CAPITAL:,.2f}")
    slip_dollars = SLIPPAGE_TICKS * TICK_SIZE * POINT_VALUE
    w(f"Commission: ${COMMISSION_PER_TRADE:.2f}/RT | Slippage: {SLIPPAGE_TICKS:.0f} tick(s) = ${slip_dollars:.2f}/trade")
    w(f"Total cost/trade: ${COMMISSION_PER_TRADE + slip_dollars:.2f}")
    w("")

    # Sweep ranges
    w("PARAMETER GRID")
    w(f"TP Long:  {_range_str(TP_LONG_RANGE)}")
    w(f"SL Long:  {_range_str(SL_LONG_RANGE)}")
    w(f"TP Short: {_range_str(TP_SHORT_RANGE)}")
    w(f"SL Short: {_range_str(SL_SHORT_RANGE)}")
    w(f"Total combinations: {len(rows)}")
    w("")

    # Data summary
    w("DATA")
    w(f"Ticks: {total_ticks:,} | Signals: {n_signals:,} | Trading days: {n_trading_days}")
    w(f"Precompute: {precompute_time:.1f}s | Sweep: {sweep_time:.1f}s ({len(rows) / sweep_time:.0f} combos/s)")
    w("")

    # Best result highlight
    if rows:
        best = rows[0]
        w("BEST RESULT")
        w(f"TP Long: {best['tp_long']:.1f} | SL Long: {best['sl_long']:.1f} | "
          f"TP Short: {best['tp_short']:.1f} | SL Short: {best['sl_short']:.1f}")
        w(f"Trades: {best['trades']:,} | Win rate: {best['win_rate']:.1%} | "
          f"P&L: {_d(best['pnl'])} | PF: {best['profit_factor']:.3f}")
        w(f"Sharpe: {best['sharpe']:.4f} | Max DD: {_d(best['max_dd'])} | "
          f"SQN: {best['sqn']:.4f} | Expectancy: {_d(best['expectancy'])}/trade")
        if INITIAL_CAPITAL > 0:
            roi = best['pnl'] / INITIAL_CAPITAL * 100
            dd_pct = abs(best['max_dd']) / INITIAL_CAPITAL * 100
            w(f"ROI: {roi:.1f}% | Drawdown: {dd_pct:.1f}% of capital")
        w("")

    # Full ranked table
    w(f"ALL RESULTS (ranked by {objective})")
    header = f"{'#':>4}  {'TP_L':>6} {'SL_L':>6} {'TP_S':>6} {'SL_S':>6}  "
    header += f"{'Trades':>6} {'WinR':>6} {'PnL':>12} {'PF':>7} {'Sharpe':>8} {'MaxDD':>12} {'Expect':>10} {'SQN':>8}"
    w(header)
    w("─" * len(header))

    for i, r in enumerate(rows):
        line = f"{i+1:>4}  {r['tp_long']:>6.1f} {r['sl_long']:>6.1f} {r['tp_short']:>6.1f} {r['sl_short']:>6.1f}  "
        line += f"{r['trades']:>6} {r['win_rate']:>5.1%} {r['pnl']:>12,.2f} "
        line += f"{r['profit_factor']:>7.3f} {r['sharpe']:>8.4f} {r['max_dd']:>12,.2f} "
        line += f"{r['expectancy']:>10.2f} {r['sqn']:>8.4f}"
        w(line)
    w("")

    # Insights
    profitable = [r for r in rows if r['pnl'] > 0]
    negative = [r for r in rows if r['pnl'] <= 0]
    w("SUMMARY")
    w(f"Profitable combos: {len(profitable)}/{len(rows)} ({len(profitable)/len(rows):.0%})")
    if profitable:
        best_pnl = max(rows, key=lambda r: r['pnl'])
        w(f"Highest P&L: {_d(best_pnl['pnl'])} (TP_L={best_pnl['tp_long']:.1f}, SL_L={best_pnl['sl_long']:.1f}, "
          f"TP_S={best_pnl['tp_short']:.1f}, SL_S={best_pnl['sl_short']:.1f})")
    if negative:
        worst = min(rows, key=lambda r: r['pnl'])
        w(f"Worst P&L: {_d(worst['pnl'])} (TP_L={worst['tp_long']:.1f}, SL_L={worst['sl_long']:.1f}, "
          f"TP_S={worst['tp_short']:.1f}, SL_S={worst['sl_short']:.1f})")
    best_sharpe = max(rows, key=lambda r: r['sharpe'])
    best_pf = max(rows, key=lambda r: r['profit_factor'])
    best_sqn = max(rows, key=lambda r: r['sqn'])
    w(f"Best Sharpe: {best_sharpe['sharpe']:.4f} (SL_L={best_sharpe['sl_long']:.1f}, SL_S={best_sharpe['sl_short']:.1f})")
    w(f"Best PF: {best_pf['profit_factor']:.3f} (SL_L={best_pf['sl_long']:.1f}, SL_S={best_pf['sl_short']:.1f})")
    w(f"Best SQN: {best_sqn['sqn']:.4f} (SL_L={best_sqn['sl_long']:.1f}, SL_S={best_sqn['sl_short']:.1f})")
    w("")

    return "\n".join(L)


if __name__ == "__main__":
    instrument = InstrumentSpec(
        symbol=INSTRUMENT_SYMBOL,
        tick_size=TICK_SIZE,
        point_value=POINT_VALUE,
    )

    # Base config (TP/SL values will be overridden per combo)
    base_config = EngineConfig(
        instrument=instrument,
        tp_points=0.0,  # placeholder, overridden by per-direction values
        sl_points=0.0,
        fast_period=FAST_EMA_PERIOD,
        slow_period=SLOW_EMA_PERIOD,
        cache_path=CACHE_PATH,
        initial_capital=INITIAL_CAPITAL,
        commission_per_trade=COMMISSION_PER_TRADE,
        slippage_ticks=SLIPPAGE_TICKS,
        entry_start_utc=ENTRY_START_UTC,
        entry_end_utc=ENTRY_END_UTC,
    )

    # 1. Precompute signals once
    print(f"Precomputing signals: EMA({FAST_EMA_PERIOD}/{SLOW_EMA_PERIOD}), "
          f"window {ENTRY_START_UTC}-{ENTRY_END_UTC} UTC")
    t0 = time.perf_counter()
    pre = precompute_signals(base_config)
    precompute_time = time.perf_counter() - t0
    print(f"  {len(pre.all_indices)} signals in {precompute_time:.1f}s "
          f"({pre.total_ticks:,} ticks, {pre.n_trading_days} trading days)")

    # 2. Build parameter grid
    tp_longs = _frange(*TP_LONG_RANGE)
    sl_longs = _frange(*SL_LONG_RANGE)
    tp_shorts = _frange(*TP_SHORT_RANGE)
    sl_shorts = _frange(*SL_SHORT_RANGE)
    grid = list(itertools.product(tp_longs, sl_longs, tp_shorts, sl_shorts))
    print(f"  {len(grid)} parameter combinations")

    # 3. Sweep
    columns = [
        "tp_long", "sl_long", "tp_short", "sl_short",
        "trades", "win_rate", "pnl", "profit_factor",
        "sharpe", "max_dd", "expectancy", "sqn",
    ]
    rows: list[dict] = []
    t1 = time.perf_counter()

    for idx, (tp_l, sl_l, tp_s, sl_s) in enumerate(grid):
        cfg = replace(
            base_config,
            tp_points=tp_l,  # base fallback (unused when per-direction set)
            sl_points=sl_l,
            tp_points_long=tp_l,
            sl_points_long=sl_l,
            tp_points_short=tp_s,
            sl_points_short=sl_s,
        )
        s = run_summary_from_signals(pre, cfg)

        rows.append({
            "tp_long": tp_l,
            "sl_long": sl_l,
            "tp_short": tp_s,
            "sl_short": sl_s,
            "trades": s.total_trades,
            "win_rate": round(s.win_rate, 4),
            "pnl": round(s.total_pnl_dollars, 2),
            "profit_factor": round(s.profit_factor, 4),
            "sharpe": round(s.sharpe_ratio, 4),
            "max_dd": round(s.max_drawdown_dollars, 2),
            "expectancy": round(s.expectancy_per_trade, 2),
            "sqn": round(s.sqn, 4),
        })

        if (idx + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t1
            rate = (idx + 1) / elapsed
            remaining = (len(grid) - idx - 1) / rate
            print(f"  {idx + 1}/{len(grid)} combos ({rate:.0f}/s, ~{remaining:.0f}s remaining)")

    sweep_time = time.perf_counter() - t1
    print(f"Sweep completed: {len(grid)} combos in {sweep_time:.1f}s "
          f"({len(grid) / sweep_time:.0f} combos/s)")

    # 4. Sort results
    rows.sort(key=lambda r: r.get(OBJECTIVE, 0), reverse=True)

    # 5. Write report and CSV to timestamped folder
    now = datetime.now(tz=timezone.utc)
    now_cst = now.astimezone(CST)
    folder_name = f"optimize {now_cst.strftime('%Y-%m-%d %I.%M.%S %p CST')}"
    run_dir = os.path.join("results", folder_name)
    os.makedirs(run_dir, exist_ok=True)

    report_text = format_optimize_report(
        rows=rows,
        objective=OBJECTIVE,
        n_signals=len(pre.all_indices),
        total_ticks=pre.total_ticks,
        n_trading_days=pre.n_trading_days,
        precompute_time=precompute_time,
        sweep_time=sweep_time,
        run_timestamp=now,
    )
    report_path = os.path.join(run_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport: {report_path}")
    print(f"CSV: {csv_path}")

    # 6. Print top N
    print(f"\nTop {TOP_N} by {OBJECTIVE}:")
    header = f"{'#':>3}  {'TP_L':>6} {'SL_L':>6} {'TP_S':>6} {'SL_S':>6}  "
    header += f"{'Trades':>6} {'WinR':>6} {'PnL':>10} {'PF':>6} {'Sharpe':>7} {'MaxDD':>10} {'SQN':>7}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows[:TOP_N]):
        line = f"{i+1:>3}  {r['tp_long']:>6.1f} {r['sl_long']:>6.1f} {r['tp_short']:>6.1f} {r['sl_short']:>6.1f}  "
        line += f"{r['trades']:>6} {r['win_rate']:>5.1%} {r['pnl']:>10,.2f} "
        line += f"{r['profit_factor']:>6.2f} {r['sharpe']:>7.4f} {r['max_dd']:>10,.2f} {r['sqn']:>7.4f}"
        print(line)

    os.startfile(report_path)
