"""scalpr_zen — run a backtest with all parameters visible at the top."""

import hashlib
import os
import pickle
import time

from scalpr_zen.engine import EngineConfig, run
from scalpr_zen.monte_carlo import run_monte_carlo, write_monte_carlo_html
from scalpr_zen.report import write_report
from scalpr_zen.types import InstrumentSpec

# ═══════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════
CACHE_PATH = "cache/nq_ticks.npz"
INSTRUMENT_SYMBOL = "NQ"
TICK_SIZE = 0.25
POINT_VALUE = 20.00

# 1. Macro Trend Filter: Validates the dominant institutional flow
FAST_EMA_PERIOD = 2000
SLOW_EMA_PERIOD = 8000

# 2. Asymmetric Scalp: Tight long scalps, wide short runners
TP_POINTS = 10.0                # Fallback (overridden by per-direction)
SL_POINTS = 35.0                # Fallback (overridden by per-direction)
TP_POINTS_LONG = 10.0           # Target $200 gross per long win
SL_POINTS_LONG = 35.0           # Risk $700 gross per long loss
TP_POINTS_SHORT = 30.0          # Target $600 gross per short win
SL_POINTS_SHORT = 60.0          # Risk $1200 gross per short loss

INITIAL_CAPITAL = 50000.0
COMMISSION_PER_TRADE = 4.12     # NQ round-trip via NinjaTrader
SLIPPAGE_TICKS = 1.0            # 1 tick slippage per entry+exit

# 3. Confine entries strictly to the US Open liquidity surge
ENTRY_START_UTC = 15.0          # 15:00 UTC = 9:00 AM CST
ENTRY_END_UTC = 16.5            # 16:30 UTC = 10:30 AM CST (Morning Volume Fades)
EXCLUDED_WEEKDAYS = (0,)        # 0=Mon — toxic day-of-week removed

PROP_FIRM_TARGET = 9000.0
PROP_FIRM_DRAWDOWN = 4500.0
RUN_MONTE_CARLO = True
MONTE_CARLO_SIMS = 1000
MONTE_CARLO_SEED = 42
USE_CACHE = True
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    instrument = InstrumentSpec(
        symbol=INSTRUMENT_SYMBOL,
        tick_size=TICK_SIZE,
        point_value=POINT_VALUE,
    )

    config = EngineConfig(
        instrument=instrument,
        tp_points=TP_POINTS,
        sl_points=SL_POINTS,
        fast_period=FAST_EMA_PERIOD,
        slow_period=SLOW_EMA_PERIOD,
        cache_path=CACHE_PATH,
        initial_capital=INITIAL_CAPITAL,
        commission_per_trade=COMMISSION_PER_TRADE,
        slippage_ticks=SLIPPAGE_TICKS,
        entry_start_utc=ENTRY_START_UTC,
        entry_end_utc=ENTRY_END_UTC,
        tp_points_long=TP_POINTS_LONG,
        sl_points_long=SL_POINTS_LONG,
        tp_points_short=TP_POINTS_SHORT,
        sl_points_short=SL_POINTS_SHORT,
        excluded_weekdays=EXCLUDED_WEEKDAYS,
    )

    # Build cache key from backtest parameters
    npz_mtime = os.path.getmtime(CACHE_PATH) if os.path.exists(CACHE_PATH) else "missing"
    bt_key = (
        f"{CACHE_PATH}:{npz_mtime}:{INSTRUMENT_SYMBOL}:{TICK_SIZE}:{POINT_VALUE}"
        f":{FAST_EMA_PERIOD}:{SLOW_EMA_PERIOD}:{TP_POINTS}:{SL_POINTS}:{INITIAL_CAPITAL}"
        f":{COMMISSION_PER_TRADE}:{SLIPPAGE_TICKS}"
        f":{ENTRY_START_UTC}:{ENTRY_END_UTC}"
        f":{TP_POINTS_LONG}:{SL_POINTS_LONG}:{TP_POINTS_SHORT}:{SL_POINTS_SHORT}"
        f":{EXCLUDED_WEEKDAYS}"
    )
    bt_hash = hashlib.sha256(bt_key.encode()).hexdigest()[:16]
    cache_file = f"cache/backtest_{bt_hash}.pkl"

    result = None

    if USE_CACHE and os.path.exists(cache_file):
        t0 = time.perf_counter()
        with open(cache_file, "rb") as f:
            result = pickle.load(f)
        elapsed = time.perf_counter() - t0
        print(f"Loaded backtest from cache in {elapsed:.2f}s — {result.summary.total_trades} trades")
    else:
        print(f"Running backtest: EMA({FAST_EMA_PERIOD}/{SLOW_EMA_PERIOD}), TP={TP_POINTS}, SL={SL_POINTS}")
        print(f"Cache: {CACHE_PATH}")

        t0 = time.perf_counter()
        result = run(config)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"Backtest failed: {result.error}")
        else:
            print(f"Backtest completed in {elapsed:.1f}s")
            if USE_CACHE:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                print(f"Cached to {cache_file}")

    if result and result.success:
        s = result.summary
        print(f"  {s.total_trades} trades, {s.win_rate:.1%} win rate, "
              f"P&L: ${s.total_pnl_dollars:,.2f}, PF: {s.profit_factor:.2f}")

        mc_result = None
        if RUN_MONTE_CARLO:
            t1 = time.perf_counter()
            mc_result = run_monte_carlo(result, MONTE_CARLO_SIMS, MONTE_CARLO_SEED)
            mc_elapsed = time.perf_counter() - t1
            if mc_result.success:
                print(f"Monte Carlo ({MONTE_CARLO_SIMS} sims) in {mc_elapsed:.1f}s")
                print(f"  P(Profit): {mc_result.stats.probability_of_profit:.1%}, "
                      f"Median P&L: ${mc_result.stats.median_final_pnl:,.0f}, "
                      f"Median DD: ${mc_result.stats.median_max_drawdown:,.0f}")
            else:
                print(f"Monte Carlo failed: {mc_result.error}")
                mc_result = None

        report_path = write_report(result, mc_result,
                                    prop_target=PROP_FIRM_TARGET,
                                    prop_drawdown=PROP_FIRM_DRAWDOWN)
        run_dir = os.path.dirname(report_path)
        print(f"Report: {report_path}")

        if mc_result:
            mc_path = write_monte_carlo_html(mc_result, output_dir=run_dir)
            print(f"Monte Carlo chart: {mc_path}")

        os.startfile(report_path)
