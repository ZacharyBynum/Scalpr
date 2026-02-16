from scalpr_zen.types import (
    BacktestResult,
    BacktestSummary,
    ContractPeriod,
    Direction,
    ExitReason,
    Fill,
    InstrumentSpec,
)


def test_direction_values():
    assert Direction.LONG.value == "LONG"
    assert Direction.SHORT.value == "SHORT"


def test_exit_reason_values():
    assert ExitReason.TP.value == "TP"
    assert ExitReason.SL.value == "SL"


def test_instrument_spec_frozen():
    spec = InstrumentSpec(symbol="NQ", tick_size=0.25, point_value=20.0)
    assert spec.symbol == "NQ"
    assert spec.tick_size == 0.25
    assert spec.point_value == 20.0


def test_contract_period():
    cp = ContractPeriod(
        symbol="NQH5",
        instrument_id="42288528",
        start_date="2025-02-18",
        end_date="2025-03-14",
    )
    assert cp.symbol == "NQH5"
    assert cp.instrument_id == "42288528"


def test_fill_pnl():
    fill = Fill(
        trade_number=1,
        direction=Direction.LONG,
        entry_time=1000000,
        entry_price=21450.25,
        exit_time=2000000,
        exit_price=21460.25,
        pnl_points=10.0,
        pnl_dollars=200.0,
        exit_reason=ExitReason.TP,
        mfe_points=10.0,
        mae_points=0.0,
    )
    assert fill.pnl_points == 10.0
    assert fill.pnl_dollars == 200.0
    assert fill.exit_reason == ExitReason.TP


def test_backtest_result_success():
    summary = BacktestSummary(
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=0.6,
        total_pnl_dollars=500.0,
        gross_profit=1200.0,
        gross_loss=-700.0,
        profit_factor=1200.0 / 700.0,
        avg_win=200.0,
        avg_loss=-175.0,
        max_drawdown_dollars=-300.0,
        max_consecutive_wins=3,
        max_consecutive_losses=2,
        total_ticks_processed=1000000,
        sharpe_ratio=1.5,
        avg_mfe_points=8.0,
        avg_mae_points=3.0,
        buy_hold_pnl_dollars=400.0,
        buy_hold_sharpe=1.0,
        buy_hold_max_dd=-200.0,
        expectancy_per_trade=50.0,
        t_stat=2.5,
        p_value=0.01,
        sqn=2.0,
        pct_days_profitable=0.55,
    )
    result = BacktestResult(
        success=True,
        error=None,
        strategy_name="EMA Crossover",
        params={"fast": 50, "slow": 200},
        fills=[],
        summary=summary,
        buy_hold_equity=[],
    )
    assert result.success is True
    assert result.summary is not None
    assert result.summary.win_rate == 0.6


def test_backtest_result_failure():
    result = BacktestResult(
        success=False,
        error="No data for date range",
        strategy_name="EMA Crossover",
        params={},
        fills=[],
        summary=None,
        buy_hold_equity=[],
    )
    assert result.success is False
    assert result.error == "No data for date range"
    assert result.summary is None
