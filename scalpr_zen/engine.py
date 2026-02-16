from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np
from numba import njit

from scalpr_zen.data import load_cache
from scalpr_zen.gpu import compute_ema, find_signals, simulate_fills_gpu
from scalpr_zen.types import (
    BacktestResult,
    BacktestSummary,
    Direction,
    ExitReason,
    Fill,
    InstrumentSpec,
)


@dataclass(frozen=True)
class EngineConfig:
    instrument: InstrumentSpec
    tp_points: float
    sl_points: float
    fast_period: int
    slow_period: int
    cache_path: str
    initial_capital: float = 0.0
    commission_per_trade: float = 0.0  # dollars per round-trip
    slippage_ticks: float = 0.0       # ticks of slippage per entry+exit
    entry_start_utc: float | None = None
    entry_end_utc: float | None = None


@njit(cache=True)
def filter_overlapping(
    signal_indices: np.ndarray,
    exit_indices: np.ndarray,
    signal_dirs: np.ndarray,
    exit_prices: np.ndarray,
    exit_reasons: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    n = len(signal_indices)
    keep = np.zeros(n, dtype=np.bool_)
    last_exit_idx = np.int64(-1)

    for i in range(n):
        if not valid[i]:
            continue
        if signal_indices[i] > last_exit_idx:
            keep[i] = True
            last_exit_idx = exit_indices[i]

    return keep


@njit(cache=True)
def invalidate_cross_rollover(
    signal_indices: np.ndarray,
    exit_indices: np.ndarray,
    valid: np.ndarray,
    rollover_indices: np.ndarray,
) -> None:
    for i in range(len(signal_indices)):
        if not valid[i]:
            continue
        entry_idx = signal_indices[i]
        exit_idx = exit_indices[i]
        for r in rollover_indices:
            if entry_idx < r <= exit_idx:
                valid[i] = False
                break


def _compute_summary(
    fills: list[Fill],
    total_ticks: int,
    n_trading_days: int,
) -> BacktestSummary:
    if not fills:
        return BacktestSummary(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl_dollars=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown_dollars=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            total_ticks_processed=total_ticks,
            sharpe_ratio=0.0,
            avg_mfe_points=0.0,
            avg_mae_points=0.0,
            buy_hold_pnl_dollars=0.0,
            buy_hold_sharpe=0.0,
            buy_hold_max_dd=0.0,
            expectancy_per_trade=0.0,
            t_stat=0.0,
            p_value=1.0,
            sqn=0.0,
            pct_days_profitable=0.0,
        )

    # Extract fill data into numpy arrays upfront
    pnl_arr = np.array([f.pnl_dollars for f in fills], dtype=np.float64)
    exit_times = np.array([f.exit_time for f in fills], dtype=np.int64)
    mfe_points = np.array([f.mfe_points for f in fills], dtype=np.float64)
    mae_points = np.array([f.mae_points for f in fills], dtype=np.float64)

    total = len(pnl_arr)
    wins_mask = pnl_arr > 0
    n_wins = int(np.sum(wins_mask))
    n_losses = total - n_wins
    gross_profit = float(np.sum(pnl_arr[wins_mask]))
    gross_loss = float(np.sum(pnl_arr[~wins_mask]))

    # Max drawdown (vectorized)
    equity = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity - peak))

    # Consecutive streaks
    max_consec_w = 0
    max_consec_l = 0
    cur_w = 0
    cur_l = 0
    for w in wins_mask:
        if w:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        if cur_w > max_consec_w:
            max_consec_w = cur_w
        if cur_l > max_consec_l:
            max_consec_l = cur_l

    # Daily P&L aggregation (vectorized day index computation)
    day_indices = exit_times // 86_400_000_000_000
    unique_days, inverse = np.unique(day_indices, return_inverse=True)
    daily_sums = np.bincount(inverse, weights=pnl_arr).astype(np.float64)

    n_active_days = len(unique_days)
    n_zero_days = n_trading_days - n_active_days
    if n_zero_days > 0:
        daily_values = np.concatenate([daily_sums, np.zeros(n_zero_days, dtype=np.float64)])
    else:
        daily_values = daily_sums

    # Sharpe ratio
    if len(daily_values) > 1:
        mean_d = float(np.mean(daily_values))
        std_d = float(np.std(daily_values, ddof=1))
        sharpe = (mean_d / std_d) * math.sqrt(252) if std_d > 0 else 0.0
    else:
        sharpe = 0.0

    avg_mfe = float(np.mean(mfe_points))
    avg_mae = float(np.mean(mae_points))

    # Validation metrics
    mean_pnl = float(np.mean(pnl_arr))
    if total > 1:
        std_pnl = float(np.std(pnl_arr, ddof=1))
    else:
        std_pnl = 0.0
    t_stat_val = (mean_pnl / (std_pnl / math.sqrt(total))) if std_pnl > 0 else 0.0
    p_val = 2.0 * (1.0 - 0.5 * math.erfc(-abs(t_stat_val) / math.sqrt(2))) if t_stat_val != 0 else 1.0
    sqn_val = math.sqrt(min(total, 100)) * mean_pnl / std_pnl if std_pnl > 0 else 0.0
    pct_days_prof = float(np.sum(daily_values > 0)) / len(daily_values) if len(daily_values) > 0 else 0.0

    return BacktestSummary(
        total_trades=total,
        winning_trades=n_wins,
        losing_trades=n_losses,
        win_rate=n_wins / total if total > 0 else 0.0,
        total_pnl_dollars=gross_profit + gross_loss,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf"),
        avg_win=gross_profit / n_wins if n_wins > 0 else 0.0,
        avg_loss=gross_loss / n_losses if n_losses > 0 else 0.0,
        max_drawdown_dollars=max_dd,
        max_consecutive_wins=max_consec_w,
        max_consecutive_losses=max_consec_l,
        total_ticks_processed=total_ticks,
        sharpe_ratio=sharpe,
        avg_mfe_points=avg_mfe,
        avg_mae_points=avg_mae,
        buy_hold_pnl_dollars=0.0,  # computed in run() and overridden
        buy_hold_sharpe=0.0,
        buy_hold_max_dd=0.0,
        expectancy_per_trade=mean_pnl,
        t_stat=t_stat_val,
        p_value=p_val,
        sqn=sqn_val,
        pct_days_profitable=pct_days_prof,
    )


def run(config: EngineConfig) -> BacktestResult:
    params = {
        "instrument": f"{config.instrument.symbol} (E-mini Nasdaq 100)",
        "point_value": config.instrument.point_value,
        "tick_size": config.instrument.tick_size,
        "fast_ema": config.fast_period,
        "slow_ema": config.slow_period,
        "tp_points": config.tp_points,
        "sl_points": config.sl_points,
        "warmup_ticks": config.slow_period,
        "initial_capital": config.initial_capital,
        "commission_per_trade": config.commission_per_trade,
        "slippage_ticks": config.slippage_ticks,
        "entry_start_utc": config.entry_start_utc,
        "entry_end_utc": config.entry_end_utc,
    }

    # 1. Load cached data
    try:
        prices, timestamps, rollover_indices = load_cache(config.cache_path)
    except FileNotFoundError:
        return BacktestResult(
            success=False,
            error=f"Cache file not found: {config.cache_path}",
            strategy_name="EMA Crossover",
            params=params,
            fills=[],
            summary=None,
            buy_hold_equity=[],
        )

    total_ticks = len(prices)
    if total_ticks == 0:
        return BacktestResult(
            success=False,
            error="Cache contains no data",
            strategy_name="EMA Crossover",
            params=params,
            fills=[],
            summary=None,
            buy_hold_equity=[],
        )

    # Add data range to params
    first_ts = datetime.fromtimestamp(timestamps[0] / 1e9, tz=timezone.utc)
    last_ts = datetime.fromtimestamp(timestamps[-1] / 1e9, tz=timezone.utc)
    params["data_range"] = f"{first_ts.strftime('%Y-%m-%d')} to {last_ts.strftime('%Y-%m-%d')}"

    # Count trading days for Sharpe ratio
    day_ns = np.int64(86400_000_000_000)
    days = timestamps // day_ns
    n_trading_days = len(np.unique(days))

    # 2. Compute EMAs
    fast_ema = compute_ema(prices, config.fast_period)
    slow_ema = compute_ema(prices, config.slow_period)

    # 3. Find crossover signals
    warmup = config.slow_period
    long_indices, short_indices = find_signals(fast_ema, slow_ema, warmup)

    if len(long_indices) == 0 and len(short_indices) == 0:
        summary = _compute_summary([], total_ticks, n_trading_days)
        return BacktestResult(
            success=True,
            error=None,
            strategy_name="EMA Crossover",
            params=params,
            fills=[],
            summary=summary,
            buy_hold_equity=[],
        )

    # 4. Merge and sort signals
    all_indices = np.concatenate([long_indices, short_indices])
    all_dirs = np.concatenate([
        np.ones(len(long_indices), dtype=np.int64),
        -np.ones(len(short_indices), dtype=np.int64),
    ])
    sort_order = np.argsort(all_indices)
    all_indices = all_indices[sort_order]
    all_dirs = all_dirs[sort_order]

    # 4b. Time-of-day filter (discard signals outside allowed window)
    if config.entry_start_utc is not None and config.entry_end_utc is not None:
        signal_ts = timestamps[all_indices]
        hour_of_day = (signal_ts % 86_400_000_000_000).astype(np.float64) / 3_600_000_000_000
        time_mask = (hour_of_day >= config.entry_start_utc) & (hour_of_day < config.entry_end_utc)
        all_indices = all_indices[time_mask]
        all_dirs = all_dirs[time_mask]

    if len(all_indices) == 0:
        summary = _compute_summary([], total_ticks, n_trading_days)
        return BacktestResult(
            success=True, error=None, strategy_name="EMA Crossover",
            params=params, fills=[], summary=summary, buy_hold_equity=[],
        )

    # 5. Simulate potential fills (parallel)
    exit_indices, exit_prices, exit_reasons, valid, mfe_arr, mae_arr = simulate_fills_gpu(
        prices, timestamps, all_indices, all_dirs,
        config.tp_points, config.sl_points, config.instrument.tick_size,
    )

    # 6. Invalidate trades spanning contract rollovers
    invalidate_cross_rollover(all_indices, exit_indices, valid, rollover_indices)

    # 7. Sequential overlap filter
    keep = filter_overlapping(
        all_indices, exit_indices, all_dirs,
        exit_prices, exit_reasons, valid,
    )

    # 8. Build Fill objects
    point_value = config.instrument.point_value
    slippage_cost = config.slippage_ticks * config.instrument.tick_size * point_value
    commission = config.commission_per_trade
    fills: list[Fill] = []
    trade_num = 0
    for i in range(len(all_indices)):
        if not keep[i]:
            continue
        trade_num += 1
        entry_idx = all_indices[i]
        direction = Direction.LONG if all_dirs[i] == 1 else Direction.SHORT
        entry_price = float(prices[entry_idx])
        exit_price = float(exit_prices[i])
        exit_reason = ExitReason.TP if exit_reasons[i] == 0 else ExitReason.SL

        if direction == Direction.LONG:
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        fills.append(Fill(
            trade_number=trade_num,
            direction=direction,
            entry_time=int(timestamps[entry_idx]),
            entry_price=entry_price,
            exit_time=int(timestamps[exit_indices[i]]),
            exit_price=exit_price,
            pnl_points=pnl_points,
            pnl_dollars=pnl_points * point_value - commission - slippage_cost,
            exit_reason=exit_reason,
            mfe_points=float(mfe_arr[i]),
            mae_points=float(mae_arr[i]),
        ))

    # 9. Compute summary
    summary = _compute_summary(fills, total_ticks, n_trading_days)

    # 10. Buy & hold per-segment (avoids phantom P&L from rollover gaps)
    seg_boundaries = np.concatenate([
        np.array([0], dtype=np.int64),
        rollover_indices,
        np.array([len(prices)], dtype=np.int64),
    ])

    buy_hold_total = 0.0
    for seg in range(len(seg_boundaries) - 1):
        seg_start = seg_boundaries[seg]
        seg_end = seg_boundaries[seg + 1]
        buy_hold_total += (float(prices[seg_end - 1]) - float(prices[seg_start])) * point_value

    # Buy & hold equity curve (daily closing prices, per-segment)
    day_changes = np.where(np.diff(days))[0]
    close_indices = np.append(day_changes, len(days) - 1)
    close_timestamps = timestamps[close_indices]

    # Batch-convert timestamps to date strings via numpy datetime64
    date_strings = (
        close_timestamps.astype("datetime64[ns]")
        .astype("datetime64[D]")
        .astype(str)
        .tolist()
    )

    running_pnl = 0.0
    seg_idx = 0
    seg_first_price = float(prices[seg_boundaries[0]])
    buy_hold_equity: list[tuple[str, float]] = []
    bh_daily_values: list[float] = []
    prev_equity = 0.0

    for j, ci in enumerate(close_indices):
        # Advance to the correct segment for this close index
        while seg_idx < len(seg_boundaries) - 2 and ci >= seg_boundaries[seg_idx + 1]:
            seg_end = seg_boundaries[seg_idx + 1]
            running_pnl += (float(prices[seg_end - 1]) - seg_first_price) * point_value
            seg_idx += 1
            seg_first_price = float(prices[seg_boundaries[seg_idx]])

        equity = running_pnl + (float(prices[ci]) - seg_first_price) * point_value
        buy_hold_equity.append((date_strings[j], round(equity, 2)))
        bh_daily_values.append(equity - prev_equity)
        prev_equity = equity

    # Buy & hold Sharpe and max drawdown
    if len(bh_daily_values) > 1:
        bh_daily_pnl = np.array(bh_daily_values, dtype=np.float64)
        bh_mean = float(np.mean(bh_daily_pnl))
        bh_std = float(np.std(bh_daily_pnl, ddof=1))
        bh_sharpe = (bh_mean / bh_std) * math.sqrt(252) if bh_std > 0 else 0.0
        bh_cumulative = np.cumsum(np.insert(bh_daily_pnl, 0, 0.0))
        bh_peak = np.maximum.accumulate(bh_cumulative)
        bh_max_dd = float(np.min(bh_cumulative - bh_peak))
    else:
        bh_sharpe = 0.0
        bh_max_dd = 0.0

    summary = replace(summary,
        buy_hold_pnl_dollars=round(buy_hold_total, 2),
        buy_hold_sharpe=round(bh_sharpe, 4),
        buy_hold_max_dd=round(bh_max_dd, 2),
    )

    return BacktestResult(
        success=True,
        error=None,
        strategy_name="EMA Crossover",
        params=params,
        fills=fills,
        summary=summary,
        buy_hold_equity=buy_hold_equity,
    )
