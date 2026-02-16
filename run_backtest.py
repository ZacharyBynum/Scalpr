"""scalpr_zen — run a backtest with all parameters visible at the top."""

import time

from scalpr_zen.engine import EngineConfig, run
from scalpr_zen.gpu import select_device
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
FAST_EMA_PERIOD = 50
SLOW_EMA_PERIOD = 200
TP_POINTS = 10.0
SL_POINTS = 5.0
RUN_MONTE_CARLO = True
MONTE_CARLO_SIMS = 1000
MONTE_CARLO_SEED = None
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    select_device()

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
    )

    print(f"Running backtest: EMA({FAST_EMA_PERIOD}/{SLOW_EMA_PERIOD}), TP={TP_POINTS}, SL={SL_POINTS}")
    print(f"Cache: {CACHE_PATH}")

    t0 = time.perf_counter()
    result = run(config)
    elapsed = time.perf_counter() - t0

    if not result.success:
        print(f"Backtest failed: {result.error}")
    else:
        report_path = write_report(result)
        print(f"Backtest completed in {elapsed:.1f}s. Report: {report_path}")
        if result.summary:
            s = result.summary
            print(f"  {s.total_trades} trades, {s.win_rate:.1%} win rate, "
                  f"P&L: ${s.total_pnl_dollars:,.2f}, PF: {s.profit_factor:.2f}")

        if RUN_MONTE_CARLO:
            t1 = time.perf_counter()
            mc_result = run_monte_carlo(result, MONTE_CARLO_SIMS, MONTE_CARLO_SEED)
            mc_elapsed = time.perf_counter() - t1
            if mc_result.success:
                mc_path = write_monte_carlo_html(mc_result)
                print(f"Monte Carlo ({MONTE_CARLO_SIMS} sims) in {mc_elapsed:.1f}s. Report: {mc_path}")
                print(f"  P(Profit): {mc_result.stats.probability_of_profit:.1%}, "
                      f"Median P&L: ${mc_result.stats.median_final_pnl:,.0f}, "
                      f"Median DD: ${mc_result.stats.median_max_drawdown:,.0f}")
            else:
                print(f"Monte Carlo failed: {mc_result.error}")
