from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from scalpr_zen.types import BacktestResult, Direction, MonteCarloResult

CST = timezone(timedelta(hours=-6))


def _fmt_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=CST).strftime("%Y-%m-%d %I:%M:%S %p")


def _d(val: float) -> str:
    return f"+${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"


def _cst12(utc_hour: float) -> str:
    h = (utc_hour - 6) % 24
    period = "AM" if int(h) < 12 else "PM"
    return f"{int(h) % 12 or 12}:{int((h % 1) * 60):02d} {period}"


def _grade(value: float, target: float, higher: bool) -> str:
    if higher:
        return "PASS" if value > target * 1.1 else ("NEUTRAL" if value >= target else "FAIL")
    return "PASS" if value < target * 0.9 else ("NEUTRAL" if value <= target else "FAIL")


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = p / 100 * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _dur(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    return f"{secs / 60:.1f}m" if secs < 3600 else f"{secs / 3600:.1f}h"


def format_report(
    result: BacktestResult,
    run_timestamp: datetime,
    mc: MonteCarloResult | None = None,
) -> str:
    L: list[str] = []
    w = L.append
    p = result.params
    run_cst = run_timestamp.astimezone(CST)

    # Header
    w("SCALPR — Backtest Report")
    w(f"Version: v0.2")
    w(f"Strategy: {result.strategy_name}")
    w(f"Run: {run_cst.strftime('%Y-%m-%d %I:%M:%S %p')} CST")
    w("")

    # Description
    fast, slow = p.get('fast_ema', '?'), p.get('slow_ema', '?')
    tp, sl_ = p.get('tp_points', 0), p.get('sl_points', 0)
    su, eu = p.get('entry_start_utc'), p.get('entry_end_utc')
    td = f" Entries restricted to {_cst12(su)}–{_cst12(eu)} CST." if su is not None and eu is not None else ""
    w("DESCRIPTION")
    w(f"Long on {fast}/{slow} EMA bullish cross, short on bearish cross. "
      f"TP {tp:.0f} pts, SL {sl_:.0f} pts (SL checked first = worst case).{td} "
      f"One position at a time.")
    w("")

    # Parameters
    comm = p.get('commission_per_trade', 0)
    slip = p.get('slippage_ticks', 0)
    tick_sz = p.get('tick_size', 0.25)
    pv = p.get('point_value', 0)
    slip_dollars = slip * tick_sz * pv

    w("PARAMETERS")
    w(f"Capital: ${p.get('initial_capital', 0):,.2f}")
    w(f"Instrument: {p.get('instrument', 'N/A')} | Point value: ${pv:.2f} | Tick: {tick_sz}")
    w(f"EMA: {fast}/{slow} | TP/SL: {tp:.2f}/{sl_:.2f} pts")
    if su is not None and eu is not None:
        w(f"Entry window: {_cst12(su)}–{_cst12(eu)} CST")
    w(f"Commission: ${comm:.2f}/RT | Slippage: {slip:.0f} tick(s) = ${slip_dollars:.2f}/trade")
    w(f"Total cost/trade: ${comm + slip_dollars:.2f}")
    if result.summary:
        w(f"Data: {p.get('data_range', 'N/A')} | {result.summary.total_ticks_processed:,} ticks")
    w("")

    if not (result.summary and result.fills):
        w("No trades generated." if result.summary else f"ERROR: {result.error}")
        return "\n".join(L)

    s = result.summary
    fills = result.fills
    initial = p.get('initial_capital', 0)
    c = 48  # value column | target column at 64 | grade column at 76
    tc = 64

    def _r(label: str, val: str, target: str | None = None, grade: str | None = None) -> str:
        base = f"{label:<22}{val}"
        if target is None:
            return base
        return f"{base:<{c}}{target:<{tc - c}}{grade}"

    # Performance
    w("PERFORMANCE")
    pnl_g = "PASS" if s.total_pnl_dollars > 0 else "FAIL"
    w(_r("Trades", f"{s.total_trades:,}"))
    w(_r("Win/Loss", f"{s.winning_trades:,}W / {s.losing_trades:,}L ({s.win_rate:.1%})", ">= 40%", _grade(s.win_rate, 0.40, True)))
    w(_r("Net P&L", _d(s.total_pnl_dollars), "> $0", pnl_g))
    total_costs = s.total_trades * (comm + slip_dollars)
    gross_pnl = s.total_pnl_dollars + total_costs
    w(_r("Gross P&L", _d(gross_pnl)))
    w(_r("Total costs", f"-${total_costs:,.2f} ({s.total_trades:,} x ${comm + slip_dollars:.2f})"))
    w(_r("Profit factor", f"{s.profit_factor:.3f}", ">= 1.50", _grade(s.profit_factor, 1.5, True)))
    w(_r("Avg win/loss", f"{_d(s.avg_win)} / {_d(s.avg_loss)}"))
    if initial > 0:
        dd_pct = abs(s.max_drawdown_dollars) / initial * 100
        w(_r("Max drawdown", f"{_d(s.max_drawdown_dollars)} ({dd_pct:.1f}%)", "<= 20%", _grade(dd_pct, 20.0, False)))
        roi = s.total_pnl_dollars / initial * 100
        w(_r("ROI", f"{roi:.1f}%", ">= 20%", _grade(roi, 20.0, True)))
    else:
        w(_r("Max drawdown", _d(s.max_drawdown_dollars)))
    w(_r("Sharpe", f"{s.sharpe_ratio:.2f}", ">= 1.00", _grade(s.sharpe_ratio, 1.0, True)))
    w(_r("Consec W/L", f"{s.max_consecutive_wins} / {s.max_consecutive_losses}"))
    w("")

    # Validation
    w("VALIDATION")
    w(_r("Expectancy", f"{_d(s.expectancy_per_trade)}/trade", "> $0", "PASS" if s.expectancy_per_trade > 0 else "FAIL"))
    w(_r("t-stat", f"{s.t_stat:.2f}", ">= 2.00", _grade(s.t_stat, 2.0, True)))
    w(_r("p-value", f"{s.p_value:.6f}", "< 0.05", _grade(s.p_value, 0.05, False)))
    w(_r("SQN", f"{s.sqn:.2f}", ">= 2.00", _grade(s.sqn, 2.0, True)))
    w(_r("Days profitable", f"{s.pct_days_profitable:.1%}", ">= 50%", _grade(s.pct_days_profitable, 0.50, True)))
    w("")

    # Directional
    longs = [f for f in fills if f.direction == Direction.LONG]
    shorts = [f for f in fills if f.direction == Direction.SHORT]
    lw = sum(1 for f in longs if f.pnl_dollars > 0)
    sw = sum(1 for f in shorts if f.pnl_dollars > 0)
    lp = sum(f.pnl_dollars for f in longs)
    sp = sum(f.pnl_dollars for f in shorts)

    w("DIRECTIONAL")
    w(f"{'':22}{'LONG':<18}SHORT")
    w(f"{'Trades':<22}{len(longs):<18,}{len(shorts):,}")
    w(f"{'Win rate':<22}{lw / len(longs) if longs else 0:<18.1%}{sw / len(shorts) if shorts else 0:.1%}")
    w(f"{'P&L':<22}{_d(lp):<18}{_d(sp)}")
    w("")

    # Duration
    durs = sorted([(f.exit_time - f.entry_time) / 1e9 for f in fills])
    wd = sorted([(f.exit_time - f.entry_time) / 1e9 for f in fills if f.pnl_dollars > 0])
    ld = sorted([(f.exit_time - f.entry_time) / 1e9 for f in fills if f.pnl_dollars <= 0])

    w("DURATION")
    w(f"Avg: {_dur(sum(durs) / len(durs))} | Median: {_dur(_pct(durs, 50))}")
    w(f"Avg winner: {_dur(sum(wd) / len(wd) if wd else 0)} | Avg loser: {_dur(sum(ld) / len(ld) if ld else 0)}")
    w(f"Range: {_dur(durs[0])} – {_dur(durs[-1])}")
    w("")

    # MFE/MAE
    w("EXCURSION (MFE/MAE)")
    w(f"Avg MFE: {s.avg_mfe_points:.2f} pts | Avg MAE: {s.avg_mae_points:.2f} pts")
    wmfe = [f.mfe_points for f in fills if f.pnl_dollars > 0]
    lmae = [f.mae_points for f in fills if f.pnl_dollars <= 0]
    if wmfe:
        w(f"Winner avg MFE: {sum(wmfe) / len(wmfe):.2f} pts")
    if lmae:
        w(f"Loser avg MAE: {sum(lmae) / len(lmae):.2f} pts")
    w("")

    # P&L distribution
    pnls = sorted([f.pnl_dollars for f in fills])
    w("P&L DISTRIBUTION")
    w(f"P5: {_d(_pct(pnls, 5))} | P25: {_d(_pct(pnls, 25))} | P50: {_d(_pct(pnls, 50))} | P75: {_d(_pct(pnls, 75))} | P95: {_d(_pct(pnls, 95))}")
    w("")

    # Hourly
    ht: defaultdict[int, int] = defaultdict(int)
    hp: defaultdict[int, float] = defaultdict(float)
    hw: defaultdict[int, int] = defaultdict(int)
    for f in fills:
        h = datetime.fromtimestamp(f.entry_time / 1e9, tz=CST).hour
        ht[h] += 1
        hp[h] += f.pnl_dollars
        if f.pnl_dollars > 0:
            hw[h] += 1

    w("HOURLY (CST)")
    w(f"{'Hour':<8}{'Trades':<10}{'Win%':<8}{'P&L':<16}{'Avg'}")
    for h in sorted(ht):
        pr = "AM" if h < 12 else "PM"
        n = ht[h]
        w(f"{h % 12 or 12:>2} {pr}   {n:<10,}{hw[h] / n:<8.1%}{_d(hp[h]):<16}{_d(hp[h] / n)}")
    w("")

    # DOW
    dn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dt_: defaultdict[int, int] = defaultdict(int)
    dp: defaultdict[int, float] = defaultdict(float)
    dw: defaultdict[int, int] = defaultdict(int)
    for f in fills:
        d = datetime.fromtimestamp(f.entry_time / 1e9, tz=CST).weekday()
        dt_[d] += 1
        dp[d] += f.pnl_dollars
        if f.pnl_dollars > 0:
            dw[d] += 1

    w("DAY OF WEEK")
    w(f"{'Day':<8}{'Trades':<10}{'Win%':<8}{'P&L':<16}{'Avg'}")
    for d in sorted(dt_):
        n = dt_[d]
        w(f"{dn[d]:<8}{n:<10,}{dw[d] / n:<8.1%}{_d(dp[d]):<16}{_d(dp[d] / n)}")
    w("")

    # Buy & hold
    w("BUY & HOLD COMPARISON")
    w(f"Strategy: {_d(s.total_pnl_dollars)} | Sharpe: {s.sharpe_ratio:.2f}")
    w(f"Buy&Hold: {_d(s.buy_hold_pnl_dollars)} | Sharpe: {s.buy_hold_sharpe:.2f} | Max DD: {_d(s.buy_hold_max_dd)}")
    w("")

    # Monte Carlo
    if mc and mc.success and mc.stats:
        ms = mc.stats
        w("MONTE CARLO SIMULATION")
        w(f"Simulations: {ms.n_simulations:,}")
        w(f"P(Profit): {ms.probability_of_profit:.1%}")
        w(f"Final P&L  — P5: {_d(ms.final_pnl_5th)} | P25: {_d(ms.final_pnl_25th)} | P50: {_d(ms.median_final_pnl)} | P75: {_d(ms.final_pnl_75th)} | P95: {_d(ms.final_pnl_95th)}")
        w(f"Max DD     — P5: {_d(ms.max_drawdown_5th)} | P25: {_d(ms.max_drawdown_25th)} | P50: {_d(ms.median_max_drawdown)} | P75: {_d(ms.max_drawdown_75th)} | P95: {_d(ms.max_drawdown_95th)}")
        w(f"Original   — P&L: {_d(ms.original_final_pnl)} | Max DD: {_d(ms.original_max_drawdown)}")
        w("")

    return "\n".join(L)


def _format_trade_log(result: BacktestResult) -> str:
    L: list[str] = []
    w = L.append
    w("SCALPR — Trade Log")
    w(f"{'#':<8}{'Dir':<7}{'Entry Time':<26}{'Entry Px':<12}{'Exit Time':<26}{'Exit Px':<12}{'P&L ($)':<12}{'Exit'}")
    for f in result.fills:
        w(f"{f.trade_number:<8}"
          f"{'LONG' if f.direction == Direction.LONG else 'SHORT':<7}"
          f"{_fmt_ns(f.entry_time):<26}{f.entry_price:<12.2f}"
          f"{_fmt_ns(f.exit_time):<26}{f.exit_price:<12.2f}"
          f"{_d(f.pnl_dollars):<12}{f.exit_reason.value}")
    return "\n".join(L)


def write_report(
    result: BacktestResult,
    mc: MonteCarloResult | None = None,
    output_dir: str = "results",
) -> str:
    now = datetime.now(tz=timezone.utc)
    now_cst = now.astimezone(CST)
    folder_name = now_cst.strftime("%Y-%m-%d %I.%M.%S %p CST")
    run_dir = os.path.join(output_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(format_report(result, now, mc))

    if result.fills:
        with open(os.path.join(run_dir, "trades.txt"), "w", encoding="utf-8") as f:
            f.write(_format_trade_log(result))

    return os.path.join(run_dir, "report.txt")
