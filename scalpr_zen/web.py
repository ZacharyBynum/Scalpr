from __future__ import annotations

from dataclasses import asdict

import numpy as np
from flask import Flask, jsonify, render_template

from scalpr_zen.types import BacktestResult, Direction, ExitReason, MonteCarloResult


def _ns_to_datetime_str(ns: int) -> str:
    s = str(np.datetime64(ns, "ns"))
    return s[:19].replace("T", " ")


def _ns_to_date_str(ns: int) -> str:
    return str(np.datetime64(ns, "ns").astype("datetime64[D]"))


def _batch_ns_to_datetime_strs(ns_arr: np.ndarray) -> list[str]:
    """Convert an int64 array of nanosecond timestamps to 'YYYY-MM-DD HH:MM:SS' strings."""
    dt_arr = ns_arr.astype("datetime64[ns]")
    # numpy datetime64 str() gives 'YYYY-MM-DDTHH:MM:SS.xxxxxxxxx'
    # Convert to second precision to get 'YYYY-MM-DDTHH:MM:SS', then replace T
    dt_sec = dt_arr.astype("datetime64[s]")
    return [str(d).replace("T", " ") for d in dt_sec]


def _batch_ns_to_date_strs(ns_arr: np.ndarray) -> np.ndarray:
    """Convert an int64 array of nanosecond timestamps to datetime64[D] array."""
    return ns_arr.astype("datetime64[ns]").astype("datetime64[D]")


def result_to_json(result: BacktestResult) -> dict:
    """Convert BacktestResult to a JSON-serializable dict for the dashboard."""
    summary = None
    if result.summary:
        s = result.summary
        summary = {
            "total_trades": s.total_trades,
            "winning_trades": s.winning_trades,
            "losing_trades": s.losing_trades,
            "win_rate": s.win_rate,
            "total_pnl_dollars": s.total_pnl_dollars,
            "gross_profit": s.gross_profit,
            "gross_loss": s.gross_loss,
            "profit_factor": s.profit_factor,
            "avg_win": s.avg_win,
            "avg_loss": s.avg_loss,
            "max_drawdown_dollars": s.max_drawdown_dollars,
            "max_consecutive_wins": s.max_consecutive_wins,
            "max_consecutive_losses": s.max_consecutive_losses,
            "total_ticks_processed": s.total_ticks_processed,
            "sharpe_ratio": s.sharpe_ratio,
            "avg_mfe_points": s.avg_mfe_points,
            "avg_mae_points": s.avg_mae_points,
            "buy_hold_pnl_dollars": s.buy_hold_pnl_dollars,
            "buy_hold_sharpe": s.buy_hold_sharpe,
            "buy_hold_max_dd": s.buy_hold_max_dd,
            "expectancy_per_trade": s.expectancy_per_trade,
            "t_stat": s.t_stat,
            "p_value": s.p_value,
            "sqn": s.sqn,
            "pct_days_profitable": s.pct_days_profitable,
        }

    fills = result.fills
    n = len(fills)

    if n == 0:
        return {
            "strategy_name": result.strategy_name,
            "params": result.params,
            "summary": summary,
            "equity_curve": [],
            "daily_pnl": [],
            "win_loss": {"tp_long": 0, "sl_long": 0, "tp_short": 0, "sl_short": 0},
            "buy_hold_curve": [
                {"time": d, "value": v} for d, v in result.buy_hold_equity
            ],
            "trades": [],
            "hourly_avg": [],
            "dow_avg": [],
        }

    # Extract fill data into numpy arrays upfront
    exit_times = np.array([f.exit_time for f in fills], dtype=np.int64)
    entry_times = np.array([f.entry_time for f in fills], dtype=np.int64)
    pnl_dollars = np.array([f.pnl_dollars for f in fills], dtype=np.float64)
    directions = np.array(
        [1 if f.direction == Direction.LONG else 0 for f in fills], dtype=np.int8
    )
    reasons = np.array(
        [0 if f.exit_reason == ExitReason.TP else 1 for f in fills], dtype=np.int8
    )

    # Win/loss breakdown by direction and exit reason (vectorized)
    is_tp = reasons == 0
    is_sl = reasons == 1
    is_long = directions == 1
    is_short = directions == 0
    win_loss = {
        "tp_long": int(np.sum(is_tp & is_long)),
        "sl_long": int(np.sum(is_sl & is_long)),
        "tp_short": int(np.sum(is_tp & is_short)),
        "sl_short": int(np.sum(is_sl & is_short)),
    }

    # Daily, hourly, and day-of-week P&L aggregation (vectorized)
    # Convert nanosecond timestamps to datetime64
    dt_arr = exit_times.astype("datetime64[ns]")
    dates = dt_arr.astype("datetime64[D]")

    # Hours: extract from nanosecond timestamps
    # seconds since midnight = (ns % ns_per_day) / ns_per_second
    ns_per_day = np.int64(86400_000_000_000)
    ns_per_hour = np.int64(3600_000_000_000)
    hours = ((exit_times % ns_per_day) // ns_per_hour).astype(np.int32)

    # Day of week: numpy epoch 1970-01-01 was Thursday (weekday=3)
    # (days_since_epoch - 4) % 7 gives 0=Mon...6=Sun
    # But more directly: (days + 3) % 7 gives 0=Mon if epoch is Thu=3
    days_i64 = dates.view("int64")
    dow = ((days_i64 - 4) % 7).astype(np.int32)

    # Daily P&L aggregation using np.unique + np.bincount
    unique_dates, date_inv = np.unique(dates, return_inverse=True)
    daily_pnl_vals = np.bincount(date_inv, weights=pnl_dollars, minlength=len(unique_dates))

    # Convert unique dates to strings
    date_strs = [str(d) for d in unique_dates]

    daily_pnl = [
        {"date": date_strs[i], "pnl": round(float(daily_pnl_vals[i]), 2)}
        for i in range(len(unique_dates))
    ]

    # Equity curve: cumulative P&L by day (vectorized)
    cumulative = np.round(np.cumsum(daily_pnl_vals), 2)
    equity_curve = [
        {"time": date_strs[i], "value": float(cumulative[i])}
        for i in range(len(unique_dates))
    ]

    # Hourly aggregation
    unique_hours, hour_inv = np.unique(hours, return_inverse=True)
    hourly_sums = np.bincount(hour_inv, weights=pnl_dollars, minlength=len(unique_hours))
    hourly_counts = np.bincount(hour_inv, minlength=len(unique_hours))
    hourly_avg = [
        {
            "hour": int(unique_hours[i]),
            "avg_pnl": round(float(hourly_sums[i] / hourly_counts[i]), 2),
            "count": int(hourly_counts[i]),
        }
        for i in range(len(unique_hours))
    ]

    # Day-of-week aggregation
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    unique_dow, dow_inv = np.unique(dow, return_inverse=True)
    dow_sums = np.bincount(dow_inv, weights=pnl_dollars, minlength=len(unique_dow))
    dow_counts = np.bincount(dow_inv, minlength=len(unique_dow))
    dow_avg = [
        {
            "day": dow_names[int(unique_dow[i])],
            "avg_pnl": round(float(dow_sums[i] / dow_counts[i]), 2),
            "count": int(dow_counts[i]),
        }
        for i in range(len(unique_dow))
    ]

    # Buy & hold equity curve
    buy_hold_curve = [
        {"time": d, "value": v} for d, v in result.buy_hold_equity
    ]

    # Trade list with batch datetime conversion
    entry_strs = _batch_ns_to_datetime_strs(entry_times)
    exit_strs = _batch_ns_to_datetime_strs(exit_times)
    pnl_rounded = np.round(pnl_dollars, 2)
    mfe_arr = np.round(np.array([f.mfe_points for f in fills], dtype=np.float64), 2)
    mae_arr = np.round(np.array([f.mae_points for f in fills], dtype=np.float64), 2)

    trades = [
        {
            "num": fills[i].trade_number,
            "dir": fills[i].direction.value,
            "entry_time": entry_strs[i],
            "entry_price": fills[i].entry_price,
            "exit_time": exit_strs[i],
            "exit_price": fills[i].exit_price,
            "pnl": float(pnl_rounded[i]),
            "exit": fills[i].exit_reason.value,
            "mfe": float(mfe_arr[i]),
            "mae": float(mae_arr[i]),
        }
        for i in range(n)
    ]

    return {
        "strategy_name": result.strategy_name,
        "params": result.params,
        "summary": summary,
        "equity_curve": equity_curve,
        "daily_pnl": daily_pnl,
        "win_loss": win_loss,
        "buy_hold_curve": buy_hold_curve,
        "trades": trades,
        "hourly_avg": hourly_avg,
        "dow_avg": dow_avg,
    }


def mc_result_to_json(mc_result: MonteCarloResult) -> dict | None:
    """Convert MonteCarloResult to a JSON-serializable dict for the dashboard."""
    if not mc_result.success or mc_result.stats is None:
        return None

    s = mc_result.stats
    n_trades = len(mc_result.original_curve)

    labels = list(range(len(mc_result.curve_50th)))
    if n_trades > 2000:
        labels = np.linspace(0, n_trades - 1, len(mc_result.curve_50th), dtype=int).tolist()

    return {
        "stats": asdict(s),
        "labels": labels,
        "curve_5th": np.round(np.asarray(mc_result.curve_5th), 2).tolist(),
        "curve_25th": np.round(np.asarray(mc_result.curve_25th), 2).tolist(),
        "curve_50th": np.round(np.asarray(mc_result.curve_50th), 2).tolist(),
        "curve_75th": np.round(np.asarray(mc_result.curve_75th), 2).tolist(),
        "curve_95th": np.round(np.asarray(mc_result.curve_95th), 2).tolist(),
        "original": np.round(np.asarray(mc_result.original_curve), 2).tolist(),
    }


def create_app(
    result: BacktestResult, mc_result: MonteCarloResult | None = None
) -> Flask:
    app = Flask(__name__, template_folder="templates")
    data = result_to_json(result)
    mc_data = mc_result_to_json(mc_result) if mc_result else None

    @app.route("/")
    def index():
        return render_template("index.html", result=data, mc=mc_data)

    @app.route("/api/result")
    def api_result():
        return jsonify(data)

    @app.route("/api/monte-carlo")
    def api_mc():
        return jsonify(mc_data)

    return app
