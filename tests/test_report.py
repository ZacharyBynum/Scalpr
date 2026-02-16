from datetime import datetime, timezone

from scalpr_zen.report import format_report
from scalpr_zen.types import (
    BacktestResult,
    BacktestSummary,
    Direction,
    ExitReason,
    Fill,
)


def _make_result() -> BacktestResult:
    fills = [
        Fill(
            trade_number=1,
            direction=Direction.LONG,
            entry_time=1739890472_000_000_000,
            entry_price=21450.50,
            exit_time=1739890921_000_000_000,
            exit_price=21460.50,
            pnl_points=10.0,
            pnl_dollars=200.0,
            exit_reason=ExitReason.TP,
            mfe_points=10.0,
            mae_points=0.0,
        ),
        Fill(
            trade_number=2,
            direction=Direction.SHORT,
            entry_time=1739893544_000_000_000,
            entry_price=21475.25,
            exit_time=1739893692_000_000_000,
            exit_price=21480.25,
            pnl_points=-5.0,
            pnl_dollars=-100.0,
            exit_reason=ExitReason.SL,
            mfe_points=0.0,
            mae_points=5.0,
        ),
    ]
    summary = BacktestSummary(
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=0.5,
        total_pnl_dollars=100.0,
        gross_profit=200.0,
        gross_loss=-100.0,
        profit_factor=2.0,
        avg_win=200.0,
        avg_loss=-100.0,
        max_drawdown_dollars=-100.0,
        max_consecutive_wins=1,
        max_consecutive_losses=1,
        total_ticks_processed=500000,
        sharpe_ratio=1.0,
        avg_mfe_points=5.0,
        avg_mae_points=2.5,
        buy_hold_pnl_dollars=50.0,
        buy_hold_sharpe=0.8,
        buy_hold_max_dd=-50.0,
        expectancy_per_trade=50.0,
        t_stat=1.0,
        p_value=0.32,
        sqn=0.5,
        pct_days_profitable=0.5,
    )
    return BacktestResult(
        success=True,
        error=None,
        strategy_name="EMA Crossover",
        params={
            "instrument": "NQ (E-mini Nasdaq 100)",
            "point_value": 20.0,
            "tick_size": 0.25,
            "fast_ema": 50,
            "slow_ema": 200,
            "tp_points": 10.0,
            "sl_points": 5.0,
            "warmup_ticks": 200,
            "data_range": "2025-02-18 to 2026-02-13",
        },
        fills=fills,
        summary=summary,
        buy_hold_equity=[],
    )


def test_format_report_contains_header():
    result = _make_result()
    now = datetime(2026, 2, 16, 14, 30, 0, tzinfo=timezone.utc)
    text = format_report(result, now)
    assert "SCALPR ZEN v0.1" in text
    assert "Backtest Report" in text


def test_format_report_contains_parameters():
    result = _make_result()
    now = datetime(2026, 2, 16, 14, 30, 0, tzinfo=timezone.utc)
    text = format_report(result, now)
    assert "NQ (E-mini Nasdaq 100)" in text
    assert "$20.00" in text
    assert "Fast EMA:         50" in text
    assert "Slow EMA:         200" in text
    assert "10.00" in text  # TP
    assert "5.00" in text   # SL


def test_format_report_contains_trades():
    result = _make_result()
    now = datetime(2026, 2, 16, 14, 30, 0, tzinfo=timezone.utc)
    text = format_report(result, now)
    assert "LONG" in text
    assert "SHORT" in text
    assert "21450.50" in text
    assert "21460.50" in text
    assert "+$200.00" in text
    assert "-$100.00" in text
    assert "TP" in text
    assert "SL" in text


def test_format_report_contains_summary():
    result = _make_result()
    now = datetime(2026, 2, 16, 14, 30, 0, tzinfo=timezone.utc)
    text = format_report(result, now)
    assert "Total trades:        2" in text
    assert "50.0%" in text
    assert "1W / 1L" in text
    assert "+$100.00" in text  # total P&L
    assert "2.000" in text     # profit factor (3dp)
    assert "Validation" in text
    assert "t-statistic" in text
    assert "target:" in text


def test_format_report_failure():
    result = BacktestResult(
        success=False,
        error="No data for date range",
        strategy_name="EMA Crossover",
        params={"instrument": "NQ"},
        fills=[],
        summary=None,
        buy_hold_equity=[],
    )
    now = datetime(2026, 2, 16, 14, 30, 0, tzinfo=timezone.utc)
    text = format_report(result, now)
    assert "ERROR: No data for date range" in text
