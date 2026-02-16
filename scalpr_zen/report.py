from __future__ import annotations

import math
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from scalpr_zen.types import BacktestResult, Direction, MonteCarloResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SOURCE_FILES = [
    "run_backtest.py",
    "scalpr_zen/types.py",
    "scalpr_zen/engine.py",
    "scalpr_zen/gpu.py",
    "scalpr_zen/data.py",
    "scalpr_zen/monte_carlo.py",
    "scalpr_zen/report.py",
]

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


def _prop_firm_section(
    fills: list[Fill],
    point_value: float,
    profit_target: float,
    drawdown_limit: float,
) -> str:
    """Simulate sequential prop firm eval attempts across all fills."""
    if not fills:
        return ""

    L: list[str] = []
    w = L.append
    w(f"PROP FIRM EVAL (${profit_target:,.0f} target / ${drawdown_limit:,.0f} trailing EOD drawdown)")

    attempts: list[dict] = []
    fi = 0

    while fi < len(fills):
        balance = 0.0
        eod_hwm = 0.0
        floor = -drawdown_limit
        worst_dd = 0.0
        start_trade = fills[fi].trade_number
        resolved = False
        daily_log: list[tuple[str, int, float, float, float, float]] = []
        day_trades = 0
        day_pnl = 0.0
        current_day: str | None = None

        while fi < len(fills):
            f = fills[fi]
            trade_day = datetime.fromtimestamp(f.exit_time / 1e9, tz=CST).strftime("%Y-%m-%d")

            # Day boundary — finalize previous day, update EOD HWM
            if current_day is not None and trade_day != current_day:
                eod_hwm = max(eod_hwm, balance)
                floor = eod_hwm - drawdown_limit
                daily_log.append((current_day, day_trades, day_pnl, balance, eod_hwm, floor))
                day_trades = 0
                day_pnl = 0.0

            current_day = trade_day

            # Mid-trade drawdown check (MAE is positive magnitude in points)
            worst_equity = balance - f.mae_points * point_value
            dd = worst_equity - eod_hwm
            if dd < worst_dd:
                worst_dd = dd

            if worst_equity < floor:
                daily_log.append((current_day, day_trades + 1, day_pnl, worst_equity, eod_hwm, floor))
                attempts.append({
                    "start": start_trade, "end": f.trade_number, "type": "bust",
                    "pnl": worst_equity, "hwm": eod_hwm, "worst_dd": worst_dd,
                    "trades": f.trade_number - start_trade + 1,
                    "days": len(daily_log), "log": daily_log,
                })
                fi += 1
                resolved = True
                break

            # Apply P&L (includes commission + slippage)
            balance += f.pnl_dollars
            day_trades += 1
            day_pnl += f.pnl_dollars

            # Post-trade check (costs make final balance worse than mid-trade)
            dd_post = balance - eod_hwm
            if dd_post < worst_dd:
                worst_dd = dd_post
            if balance < floor:
                daily_log.append((current_day, day_trades, day_pnl, balance, eod_hwm, floor))
                attempts.append({
                    "start": start_trade, "end": f.trade_number, "type": "bust",
                    "pnl": balance, "hwm": eod_hwm, "worst_dd": worst_dd,
                    "trades": f.trade_number - start_trade + 1,
                    "days": len(daily_log), "log": daily_log,
                })
                fi += 1
                resolved = True
                break

            # Target hit
            if balance >= profit_target:
                eod_hwm = max(eod_hwm, balance)
                daily_log.append((current_day, day_trades, day_pnl, balance, eod_hwm, eod_hwm - drawdown_limit))
                attempts.append({
                    "start": start_trade, "end": f.trade_number, "type": "pass",
                    "pnl": balance, "hwm": eod_hwm, "worst_dd": worst_dd,
                    "trades": f.trade_number - start_trade + 1,
                    "days": len(daily_log), "log": daily_log,
                })
                fi += 1
                resolved = True
                break

            fi += 1

        if not resolved:
            eod_hwm = max(eod_hwm, balance)
            floor = eod_hwm - drawdown_limit
            if current_day is not None:
                daily_log.append((current_day, day_trades, day_pnl, balance, eod_hwm, floor))
            attempts.append({
                "start": start_trade, "end": fills[-1].trade_number, "type": "exhausted",
                "pnl": balance, "hwm": eod_hwm, "worst_dd": worst_dd,
                "trades": fills[-1].trade_number - start_trade + 1,
                "days": len(daily_log), "log": daily_log,
            })
            break

    # First attempt detail
    first = attempts[0]
    if first["type"] == "pass":
        w(f"Result: PASS — hit target after {first['trades']} trades ({first['days']} trading days)")
    elif first["type"] == "bust":
        w(f"Result: BUST — trade #{first['end']} breached trailing limit")
    else:
        w(f"Result: INCOMPLETE — {first['trades']} trades, never hit target or limit")

    w(f"Balance: {_d(first['pnl'])}  |  Peak EOD: {_d(first['hwm'])}  |  Worst DD: {_d(first['worst_dd'])}  |  Margin: {_d(drawdown_limit + first['worst_dd'])}")
    w("")

    # Day-by-day log (first attempt only)
    log = first["log"]
    if log:
        w(f"{'Date':<13}{'#':<6}{'Day P&L':<13}{'Balance':<13}{'EOD HWM':<13}{'Floor':<13}{'DD from HWM'}")
        for date, nt, dp, bal, hwm, fl in log:
            w(f"{date:<13}{nt:<6}{_d(dp):<13}{_d(bal):<13}{_d(hwm):<13}{_d(fl):<13}{_d(bal - hwm)}")
    w("")

    # Multi-attempt summary
    n_pass = sum(1 for a in attempts if a["type"] == "pass")
    n_bust = sum(1 for a in attempts if a["type"] == "bust")
    n_inc = len(attempts) - n_pass - n_bust
    resolved_count = n_pass + n_bust
    pass_rate = n_pass / resolved_count if resolved_count > 0 else 0

    w(f"All attempts: {n_pass} PASS / {n_bust} BUST" +
      (f" / {n_inc} incomplete" if n_inc else "") +
      f" ({pass_rate:.0%} pass rate)")
    for idx, a in enumerate(attempts):
        if a["type"] == "pass":
            w(f"  #{idx+1}: PASS  {a['trades']:>4} trades / {a['days']:>3} days  {_d(a['pnl']):>12}")
        elif a["type"] == "bust":
            w(f"  #{idx+1}: BUST  trade #{a['end']:<5}  {_d(a['pnl']):>12}  (HWM {_d(a['hwm'])})")
        else:
            w(f"  #{idx+1}: INC   {a['trades']:>4} trades  {_d(a['pnl']):>12}")
    w("")

    return "\n".join(L)


def format_report(
    result: BacktestResult,
    run_timestamp: datetime,
    mc: MonteCarloResult | None = None,
    prop_target: float | None = None,
    prop_drawdown: float | None = None,
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

    # Prop firm eval
    if prop_target is not None and prop_drawdown is not None:
        w(_prop_firm_section(fills, pv, prop_target, prop_drawdown))

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

    # Monthly breakdown
    mt: defaultdict[str, int] = defaultdict(int)
    mp: defaultdict[str, float] = defaultdict(float)
    mw: defaultdict[str, int] = defaultdict(int)
    for f in fills:
        key = datetime.fromtimestamp(f.entry_time / 1e9, tz=CST).strftime("%Y-%m")
        mt[key] += 1
        mp[key] += f.pnl_dollars
        if f.pnl_dollars > 0:
            mw[key] += 1

    w("MONTHLY")
    w(f"{'Month':<10}{'Trades':<10}{'Win%':<8}{'P&L':<16}{'Avg'}")
    for key in sorted(mt):
        n = mt[key]
        w(f"{key:<10}{n:<10,}{mw[key] / n:<8.1%}{_d(mp[key]):<16}{_d(mp[key] / n)}")
    w("")

    # Top drawdowns
    equity = 0.0
    peak = 0.0
    dd_start_idx = 0
    drawdowns: list[tuple[float, int, int, int]] = []  # (depth, start_fill, trough_fill, recovery_fill)
    in_dd = False
    trough_idx = 0
    trough_equity = 0.0

    for i, f in enumerate(fills):
        equity += f.pnl_dollars
        if equity > peak:
            if in_dd:
                drawdowns.append((trough_equity - peak, dd_start_idx, trough_idx, i))
            peak = equity
            in_dd = False
        else:
            if not in_dd:
                dd_start_idx = i
                in_dd = True
                trough_equity = equity
                trough_idx = i
            if equity < trough_equity:
                trough_equity = equity
                trough_idx = i

    if in_dd:
        drawdowns.append((trough_equity - peak, dd_start_idx, trough_idx, len(fills) - 1))

    drawdowns.sort(key=lambda x: x[0])
    w("TOP DRAWDOWNS")
    w(f"{'#':<4}{'Depth':<16}{'Start':<24}{'Trough':<24}{'Recovery':<24}{'Trades'}")
    for rank, (depth, si, ti, ri) in enumerate(drawdowns[:5], 1):
        start_dt = datetime.fromtimestamp(fills[si].entry_time / 1e9, tz=CST).strftime("%Y-%m-%d %I:%M %p")
        trough_dt = datetime.fromtimestamp(fills[ti].entry_time / 1e9, tz=CST).strftime("%Y-%m-%d %I:%M %p")
        if ri < len(fills) - 1 or not in_dd or rank > 1:
            recov_dt = datetime.fromtimestamp(fills[ri].exit_time / 1e9, tz=CST).strftime("%Y-%m-%d %I:%M %p")
        else:
            recov_dt = "(ongoing)"
        w(f"{rank:<4}{_d(depth):<16}{start_dt:<24}{trough_dt:<24}{recov_dt:<24}{ri - si + 1}")
    w("")

    # AI analysis context
    payoff = abs(s.avg_win / s.avg_loss) if s.avg_loss != 0 else float("inf")
    breakeven_wr = 1.0 / (1.0 + payoff) if payoff != float("inf") else 0.0
    wr_gap = s.win_rate - breakeven_wr
    kelly = s.win_rate - (1.0 - s.win_rate) / payoff if payoff > 0 and payoff != float("inf") else 0.0
    recovery = s.total_pnl_dollars / abs(s.max_drawdown_dollars) if s.max_drawdown_dollars != 0 else 0.0

    # Annualized return for Calmar
    first_ts = fills[0].entry_time / 1e9
    last_ts = fills[-1].exit_time / 1e9
    years = (last_ts - first_ts) / (365.25 * 86400)
    ann_return = s.total_pnl_dollars / years if years > 0 else 0.0
    calmar = ann_return / abs(s.max_drawdown_dollars) if s.max_drawdown_dollars != 0 else 0.0

    # Skewness and kurtosis
    n_t = len(pnls)
    mean_pnl = sum(pnls) / n_t
    var_pnl = sum((x - mean_pnl) ** 2 for x in pnls) / n_t
    std_pnl = math.sqrt(var_pnl) if var_pnl > 0 else 0.0
    if std_pnl > 0 and n_t > 2:
        skew = sum((x - mean_pnl) ** 3 for x in pnls) / (n_t * std_pnl ** 3)
        kurt = sum((x - mean_pnl) ** 4 for x in pnls) / (n_t * std_pnl ** 4) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    w("AI ANALYSIS CONTEXT")
    w(f"Payoff ratio         {payoff:.2f}:1")
    w(f"Breakeven win rate   {breakeven_wr:.1%}")
    w(f"Win rate gap         {wr_gap:+.1%} ({'above' if wr_gap > 0 else 'below'} breakeven)")
    w(f"Kelly fraction       {kelly:.3f}" + (" (do not trade)" if kelly <= 0 else f" ({kelly:.1%} of capital)"))
    w(f"Recovery factor      {recovery:.2f}" + (" (negative P&L)" if s.total_pnl_dollars < 0 else ""))
    w(f"Calmar ratio         {calmar:.2f}")
    w(f"Skewness             {skew:.3f}")
    w(f"Kurtosis             {kurt:.3f}")
    w(f"Annualized return    {_d(ann_return)}")
    w(f"Test duration        {years:.2f} years ({n_t:,} trades)")
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


def _format_source_bundle() -> str:
    L: list[str] = []
    w = L.append
    w("SCALPR — Source Code")
    w(f"Generated: {datetime.now(tz=CST).strftime('%Y-%m-%d %I:%M:%S %p')} CST")
    w("")
    for rel in _SOURCE_FILES:
        path = _PROJECT_ROOT / rel
        w(f"{'═' * 80}")
        w(f"FILE: {rel}")
        w(f"{'═' * 80}")
        if path.exists():
            w(path.read_text(encoding="utf-8"))
        else:
            w(f"(file not found)")
        w("")
    return "\n".join(L)


def write_report(
    result: BacktestResult,
    mc: MonteCarloResult | None = None,
    output_dir: str = "results",
    prop_target: float | None = None,
    prop_drawdown: float | None = None,
) -> str:
    now = datetime.now(tz=timezone.utc)
    now_cst = now.astimezone(CST)
    folder_name = now_cst.strftime("%Y-%m-%d %I.%M.%S %p CST")
    run_dir = os.path.join(output_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(format_report(result, now, mc, prop_target, prop_drawdown))

    if result.fills:
        with open(os.path.join(run_dir, "trades.txt"), "w", encoding="utf-8") as f:
            f.write(_format_trade_log(result))

    with open(os.path.join(run_dir, "source.txt"), "w", encoding="utf-8") as f:
        f.write(_format_source_bundle())

    return os.path.join(run_dir, "report.txt")
