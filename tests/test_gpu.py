import numpy as np

from scalpr_zen.gpu import compute_ema, find_signals, simulate_fills_gpu


def test_ema_constant_price():
    prices = np.full(100, 21000.0, dtype=np.float32)
    ema = compute_ema(prices, 10)
    assert len(ema) == 100
    np.testing.assert_allclose(ema[-1], 21000.0, atol=0.01)


def test_ema_trending_up():
    prices = np.arange(1, 101, dtype=np.float32)
    ema = compute_ema(prices, 10)
    # EMA lags behind price in a trend
    assert ema[-1] < prices[-1]
    assert ema[-1] > prices[-1] - 10


def test_ema_period_1_tracks_price():
    prices = np.array([10.0, 20.0, 15.0, 25.0], dtype=np.float32)
    ema = compute_ema(prices, 1)
    np.testing.assert_allclose(ema, prices, atol=0.01)


def test_find_signals_basic_crossover():
    n = 200
    fast = np.zeros(n, dtype=np.float32)
    slow = np.zeros(n, dtype=np.float32)
    # Before crossover: fast below slow
    fast[:100] = 10.0
    slow[:100] = 20.0
    # After crossover: fast above slow (long signal at index 100)
    fast[100:] = 20.0
    slow[100:] = 10.0

    long_idx, short_idx = find_signals(fast, slow, warmup=10)
    assert len(long_idx) == 1
    assert long_idx[0] == 100
    assert len(short_idx) == 0


def test_find_signals_short_crossover():
    n = 200
    fast = np.zeros(n, dtype=np.float32)
    slow = np.zeros(n, dtype=np.float32)
    # Before: fast above slow
    fast[:100] = 20.0
    slow[:100] = 10.0
    # After: fast below slow (short signal at index 100)
    fast[100:] = 10.0
    slow[100:] = 20.0

    long_idx, short_idx = find_signals(fast, slow, warmup=10)
    assert len(short_idx) == 1
    assert short_idx[0] == 100
    assert len(long_idx) == 0


def test_find_signals_respects_warmup():
    n = 200
    fast = np.zeros(n, dtype=np.float32)
    slow = np.zeros(n, dtype=np.float32)
    fast[:50] = 10.0
    slow[:50] = 20.0
    fast[50:] = 20.0
    slow[50:] = 10.0

    # Warmup=100 means skip indices <= 100
    long_idx, _ = find_signals(fast, slow, warmup=100)
    assert len(long_idx) == 0


def test_simulate_fills_long_tp():
    prices = np.array([100.0, 100.5, 101.0, 110.5, 111.0], dtype=np.float32)
    timestamps = np.arange(5, dtype=np.int64) * 1_000_000_000
    signal_indices = np.array([0], dtype=np.int64)
    signal_dirs = np.array([1], dtype=np.int64)  # LONG

    exit_idx, exit_px, exit_reason, valid, _, _ = simulate_fills_gpu(
        prices, timestamps, signal_indices, signal_dirs,
        tp_points=10.0, sl_points=5.0, tick_size=0.25,
    )
    assert valid[0]
    assert exit_reason[0] == 0  # TP
    assert exit_px[0] == 110.0  # 100 + 10


def test_simulate_fills_long_sl():
    prices = np.array([100.0, 99.0, 95.0, 94.0], dtype=np.float32)
    timestamps = np.arange(4, dtype=np.int64) * 1_000_000_000
    signal_indices = np.array([0], dtype=np.int64)
    signal_dirs = np.array([1], dtype=np.int64)  # LONG

    exit_idx, exit_px, exit_reason, valid, _, _ = simulate_fills_gpu(
        prices, timestamps, signal_indices, signal_dirs,
        tp_points=10.0, sl_points=5.0, tick_size=0.25,
    )
    assert valid[0]
    assert exit_reason[0] == 1  # SL
    assert exit_px[0] == 95.0  # 100 - 5


def test_simulate_fills_short_tp():
    prices = np.array([100.0, 99.0, 90.5, 89.0], dtype=np.float32)
    timestamps = np.arange(4, dtype=np.int64) * 1_000_000_000
    signal_indices = np.array([0], dtype=np.int64)
    signal_dirs = np.array([-1], dtype=np.int64)  # SHORT

    exit_idx, exit_px, exit_reason, valid, _, _ = simulate_fills_gpu(
        prices, timestamps, signal_indices, signal_dirs,
        tp_points=10.0, sl_points=5.0, tick_size=0.25,
    )
    assert valid[0]
    assert exit_reason[0] == 0  # TP
    assert exit_px[0] == 90.0  # 100 - 10


def test_simulate_fills_sl_checked_before_tp():
    """When price hits both SL and TP on same tick, SL wins (worst case)."""
    # LONG from 100, TP=110, SL=95
    # Price jumps to 95 (SL hit) — even though TP might also be reachable
    prices = np.array([100.0, 95.0], dtype=np.float32)
    timestamps = np.arange(2, dtype=np.int64) * 1_000_000_000
    signal_indices = np.array([0], dtype=np.int64)
    signal_dirs = np.array([1], dtype=np.int64)

    exit_idx, exit_px, exit_reason, valid, _, _ = simulate_fills_gpu(
        prices, timestamps, signal_indices, signal_dirs,
        tp_points=10.0, sl_points=5.0, tick_size=0.25,
    )
    assert valid[0]
    assert exit_reason[0] == 1  # SL


def test_simulate_fills_no_exit():
    prices = np.array([100.0, 100.5, 101.0], dtype=np.float32)
    timestamps = np.arange(3, dtype=np.int64) * 1_000_000_000
    signal_indices = np.array([0], dtype=np.int64)
    signal_dirs = np.array([1], dtype=np.int64)

    exit_idx, exit_px, exit_reason, valid, _, _ = simulate_fills_gpu(
        prices, timestamps, signal_indices, signal_dirs,
        tp_points=100.0, sl_points=100.0, tick_size=0.25,
    )
    assert not valid[0]
