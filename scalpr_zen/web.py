from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template

from scalpr_zen.types import BacktestResult, Direction, ExitReason, MonteCarloResult


def _ns_to_datetime_str(ns: int) -> str:
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _ns_to_date_str(ns: int) -> str:
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


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
        }

    # Win/loss breakdown by direction and exit reason
    win_loss = {"tp_long": 0, "sl_long": 0, "tp_short": 0, "sl_short": 0}
    for f in result.fills:
        key = f"{f.exit_reason.value.lower()}_{f.direction.value.lower()}"
        win_loss[key] += 1

    # Daily P&L aggregation
    daily_pnl_map: defaultdict[str, float] = defaultdict(float)
    for f in result.fills:
        date_str = _ns_to_date_str(f.exit_time)
        daily_pnl_map[date_str] += f.pnl_dollars

    sorted_dates = sorted(daily_pnl_map.keys())
    daily_pnl = [{"date": d, "pnl": round(daily_pnl_map[d], 2)} for d in sorted_dates]

    # Equity curve: cumulative P&L by day
    cumulative = 0.0
    equity_curve = []
    for d in sorted_dates:
        cumulative += daily_pnl_map[d]
        equity_curve.append({"time": d, "value": round(cumulative, 2)})

    # Buy & hold equity curve
    buy_hold_curve = [
        {"time": d, "value": v} for d, v in result.buy_hold_equity
    ]

    # Trade list
    trades = [
        {
            "num": f.trade_number,
            "dir": f.direction.value,
            "entry_time": _ns_to_datetime_str(f.entry_time),
            "entry_price": f.entry_price,
            "exit_time": _ns_to_datetime_str(f.exit_time),
            "exit_price": f.exit_price,
            "pnl": round(f.pnl_dollars, 2),
            "exit": f.exit_reason.value,
            "mfe": round(f.mfe_points, 2),
            "mae": round(f.mae_points, 2),
        }
        for f in result.fills
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
    }


def mc_result_to_json(mc_result: MonteCarloResult) -> dict | None:
    """Convert MonteCarloResult to a JSON-serializable dict for the dashboard."""
    if not mc_result.success or mc_result.stats is None:
        return None

    s = mc_result.stats
    n_trades = len(mc_result.original_curve)

    labels = list(range(len(mc_result.curve_50th)))
    if n_trades > 2000:
        import numpy as np
        labels = np.linspace(0, n_trades - 1, len(mc_result.curve_50th), dtype=int).tolist()

    return {
        "stats": asdict(s),
        "labels": labels,
        "curve_5th": [round(v, 2) for v in mc_result.curve_5th],
        "curve_25th": [round(v, 2) for v in mc_result.curve_25th],
        "curve_50th": [round(v, 2) for v in mc_result.curve_50th],
        "curve_75th": [round(v, 2) for v in mc_result.curve_75th],
        "curve_95th": [round(v, 2) for v in mc_result.curve_95th],
        "original": [round(v, 2) for v in mc_result.original_curve],
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
