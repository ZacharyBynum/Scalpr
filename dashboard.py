"""scalpr_zen — launch the dashboard with all parameters visible at the top."""

import hashlib
import os
import pickle
import time
import webbrowser
from threading import Timer

from scalpr_zen.engine import EngineConfig, run
from scalpr_zen.gpu import select_device
from scalpr_zen.monte_carlo import run_monte_carlo
from scalpr_zen.types import InstrumentSpec
from scalpr_zen.web import create_app

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

    # Build cache key from all parameters that affect the result
    npz_mtime = os.path.getmtime(CACHE_PATH) if os.path.exists(CACHE_PATH) else "missing"
    key_data = (
        f"{CACHE_PATH}:{npz_mtime}:{INSTRUMENT_SYMBOL}:{TICK_SIZE}:{POINT_VALUE}"
        f":{FAST_EMA_PERIOD}:{SLOW_EMA_PERIOD}:{TP_POINTS}:{SL_POINTS}"
        f":{MONTE_CARLO_SIMS}:{MONTE_CARLO_SEED}"
    )
    params_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
    cache_file = f"cache/dashboard_{params_hash}.pkl"

    result = None
    mc_result = None

    if os.path.exists(cache_file):
        t0 = time.perf_counter()
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        result = cached["result"]
        mc_result = cached["mc_result"]
        elapsed = time.perf_counter() - t0
        print(f"Loaded from cache in {elapsed:.2f}s — {result.summary.total_trades} trades")
    else:
        print(f"Running backtest: EMA({FAST_EMA_PERIOD}/{SLOW_EMA_PERIOD}), TP={TP_POINTS}, SL={SL_POINTS}")

        t0 = time.perf_counter()
        result = run(config)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"Backtest failed: {result.error}")
        else:
            print(f"Backtest completed in {elapsed:.1f}s — {result.summary.total_trades} trades")

            t1 = time.perf_counter()
            mc_result = run_monte_carlo(result, MONTE_CARLO_SIMS, MONTE_CARLO_SEED)
            mc_elapsed = time.perf_counter() - t1
            if mc_result.success:
                print(f"Monte Carlo ({MONTE_CARLO_SIMS} sims) in {mc_elapsed:.1f}s")
            else:
                print(f"Monte Carlo skipped: {mc_result.error}")
                mc_result = None

            with open(cache_file, "wb") as f:
                pickle.dump({"result": result, "mc_result": mc_result}, f)
            print(f"Cached to {cache_file}")

    if result and result.success:
        print("Starting dashboard at http://localhost:5001")

        app = create_app(result, mc_result)
        Timer(1.0, lambda: webbrowser.open("http://localhost:5001")).start()
        app.run(host="localhost", port=5001, debug=False)
