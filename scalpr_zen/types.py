from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(Enum):
    TP = "TP"
    SL = "SL"


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    tick_size: float
    point_value: float


@dataclass(frozen=True)
class ContractPeriod:
    symbol: str
    instrument_id: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class Fill:
    trade_number: int
    direction: Direction
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    pnl_points: float
    pnl_dollars: float
    exit_reason: ExitReason
    mfe_points: float
    mae_points: float


@dataclass(frozen=True)
class BacktestSummary:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_dollars: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_drawdown_dollars: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_ticks_processed: int
    sharpe_ratio: float
    avg_mfe_points: float
    avg_mae_points: float
    buy_hold_pnl_dollars: float


@dataclass(frozen=True)
class BacktestResult:
    success: bool
    error: str | None
    strategy_name: str
    params: dict[str, object]
    fills: list[Fill]
    summary: BacktestSummary | None
    buy_hold_equity: list[tuple[str, float]]


@dataclass(frozen=True)
class MonteCarloStats:
    n_simulations: int
    probability_of_profit: float
    median_final_pnl: float
    final_pnl_5th: float
    final_pnl_25th: float
    final_pnl_75th: float
    final_pnl_95th: float
    median_max_drawdown: float
    max_drawdown_5th: float
    max_drawdown_25th: float
    max_drawdown_75th: float
    max_drawdown_95th: float
    original_final_pnl: float
    original_max_drawdown: float


@dataclass(frozen=True)
class MonteCarloResult:
    success: bool
    error: str | None
    strategy_name: str
    params: dict[str, object]
    stats: MonteCarloStats | None
    curve_5th: list[float]
    curve_25th: list[float]
    curve_50th: list[float]
    curve_75th: list[float]
    curve_95th: list[float]
    original_curve: list[float]


def snap_to_tick(price: float, tick_size: float = 0.25) -> float:
    return round(round(price / tick_size) * tick_size, 10)
