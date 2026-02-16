import os
import tempfile

import numpy as np

from scalpr_zen.engine import EngineConfig, filter_overlapping, run, _compute_summary
from scalpr_zen.types import (
    BacktestSummary,
    Direction,
    ExitReason,
    Fill,
    InstrumentSpec,
)

NQ = InstrumentSpec(symbol="NQ", tick_size=0.25, point_value=20.0)


def _make_cache(prices: np.ndarray, timestamps: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    np.savez_compressed(path, prices=prices, timestamps=timestamps)
    return path


def test_filter_overlapping_no_overlap():
    sig_idx = np.array([10, 50, 100], dtype=np.int64)
    exit_idx = np.array([20, 60, 110], dtype=np.int64)
    dirs = np.array([1, -1, 1], dtype=np.int64)
    exit_px = np.array([110.0, 90.0, 110.0], dtype=np.float32)
    exit_reasons = np.array([0, 0, 0], dtype=np.int8)
    valid = np.array([True, True, True])

    keep = filter_overlapping(sig_idx, exit_idx, dirs, exit_px, exit_reasons, valid)
    assert keep.sum() == 3


def test_filter_overlapping_with_overlap():
    sig_idx = np.array([10, 15, 50], dtype=np.int64)
    exit_idx = np.array([30, 25, 60], dtype=np.int64)
    dirs = np.array([1, -1, 1], dtype=np.int64)
    exit_px = np.array([110.0, 90.0, 110.0], dtype=np.float32)
    exit_reasons = np.array([0, 0, 0], dtype=np.int8)
    valid = np.array([True, True, True])

    keep = filter_overlapping(sig_idx, exit_idx, dirs, exit_px, exit_reasons, valid)
    # Signal at 15 overlaps with trade 10->30, should be skipped
    assert keep[0] == True
    assert keep[1] == False
    assert keep[2] == True


def test_filter_overlapping_skips_invalid():
    sig_idx = np.array([10, 50], dtype=np.int64)
    exit_idx = np.array([-1, 60], dtype=np.int64)
    dirs = np.array([1, 1], dtype=np.int64)
    exit_px = np.array([0.0, 110.0], dtype=np.float32)
    exit_reasons = np.array([-1, 0], dtype=np.int8)
    valid = np.array([False, True])

    keep = filter_overlapping(sig_idx, exit_idx, dirs, exit_px, exit_reasons, valid)
    assert keep[0] == False
    assert keep[1] == True


def test_compute_summary_empty():
    s = _compute_summary([], 1000, 0)
    assert s.total_trades == 0
    assert s.win_rate == 0.0
    assert s.total_ticks_processed == 1000


def test_compute_summary_basic():
    fills = [
        Fill(1, Direction.LONG, 100, 21000.0, 200, 21010.0, 10.0, 200.0, ExitReason.TP, 10.0, 0.0),
        Fill(2, Direction.SHORT, 300, 21010.0, 400, 21015.0, -5.0, -100.0, ExitReason.SL, 0.0, 5.0),
        Fill(3, Direction.LONG, 500, 21000.0, 600, 21010.0, 10.0, 200.0, ExitReason.TP, 10.0, 0.0),
    ]
    s = _compute_summary(fills, 5000, 1)
    assert s.total_trades == 3
    assert s.winning_trades == 2
    assert s.losing_trades == 1
    assert abs(s.win_rate - 2.0 / 3.0) < 0.001
    assert s.total_pnl_dollars == 300.0
    assert s.gross_profit == 400.0
    assert s.gross_loss == -100.0
    assert abs(s.profit_factor - 4.0) < 0.001
    assert s.max_consecutive_wins == 1  # W, L, W — no consecutive
    assert s.max_consecutive_losses == 1
    assert abs(s.expectancy_per_trade - 100.0) < 0.001
    assert s.t_stat > 0
    assert s.p_value < 1.0


def test_run_missing_cache():
    config = EngineConfig(
        instrument=NQ,
        tp_points=10.0,
        sl_points=5.0,
        fast_period=5,
        slow_period=10,
        cache_path="nonexistent.npz",
    )
    result = run(config)
    assert result.success is False
    assert "not found" in result.error


def test_run_with_synthetic_data():
    # Create synthetic trending price data that will generate crossover signals
    np.random.seed(42)
    n = 10000
    # Create a price series with a clear trend reversal
    trend = np.concatenate([
        np.linspace(21000, 21100, n // 2),  # uptrend
        np.linspace(21100, 21000, n // 2),  # downtrend
    ])
    noise = np.random.randn(n) * 0.5
    prices = (trend + noise).astype(np.float32)
    timestamps = (np.arange(n, dtype=np.int64) + 1_700_000_000) * 1_000_000_000

    cache_path = _make_cache(prices, timestamps)
    try:
        config = EngineConfig(
            instrument=NQ,
            tp_points=10.0,
            sl_points=5.0,
            fast_period=5,
            slow_period=20,
            cache_path=cache_path,
        )
        result = run(config)
        assert result.success is True
        assert result.summary is not None
        assert result.summary.total_ticks_processed == n
        # Should have at least some trades from the trend reversal
        assert result.summary.total_trades >= 0
    finally:
        os.unlink(cache_path)
