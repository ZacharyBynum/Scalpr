from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template

from scalpr_zen.types import BacktestResult, Direction, ExitReason


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

    return {
        "strategy_name": result.strategy_name,
        "params": result.params,
        "summary": summary,
        "equity_curve": equity_curve,
        "daily_pnl": daily_pnl,
        "win_loss": win_loss,
        "buy_hold_curve": buy_hold_curve,
    }


def create_app(result: BacktestResult) -> Flask:
    app = Flask(__name__, template_folder="templates")
    data = result_to_json(result)

    @app.route("/")
    def index():
        return render_template("index.html", result=data)

    @app.route("/api/result")
    def api_result():
        return jsonify(data)

    return app
