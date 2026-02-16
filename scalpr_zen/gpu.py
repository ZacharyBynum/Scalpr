from __future__ import annotations

import numpy as np
from numba import njit, prange


def select_device() -> None:
    pass


@njit(cache=True)
def compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    out = np.empty(n, dtype=np.float32)
    alpha = np.float32(2.0 / (period + 1))
    one_minus_alpha = np.float32(1.0) - alpha
    out[0] = prices[0]
    for i in range(1, n):
        out[i] = alpha * prices[i] + one_minus_alpha * out[i - 1]
    return out


@njit(cache=True)
def find_signals(
    fast_ema: np.ndarray,
    slow_ema: np.ndarray,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(fast_ema)
    # Pre-allocate max possible size
    long_buf = np.empty(n, dtype=np.int64)
    short_buf = np.empty(n, dtype=np.int64)
    n_long = 0
    n_short = 0

    for i in range(warmup + 1, n):
        prev_diff = fast_ema[i - 1] - slow_ema[i - 1]
        curr_diff = fast_ema[i] - slow_ema[i]
        # Long signal: fast crosses above slow
        if prev_diff <= 0.0 and curr_diff > 0.0:
            long_buf[n_long] = i
            n_long += 1
        # Short signal: fast crosses below slow
        elif prev_diff >= 0.0 and curr_diff < 0.0:
            short_buf[n_short] = i
            n_short += 1

    return long_buf[:n_long].copy(), short_buf[:n_short].copy()


@njit(parallel=True, cache=True)
def simulate_fills_gpu(
    prices: np.ndarray,
    timestamps: np.ndarray,
    signal_indices: np.ndarray,
    signal_dirs: np.ndarray,
    tp_points: float,
    sl_points: float,
    tick_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_signals = len(signal_indices)
    n_prices = len(prices)
    exit_indices = np.empty(n_signals, dtype=np.int64)
    exit_prices = np.empty(n_signals, dtype=np.float32)
    exit_reasons = np.empty(n_signals, dtype=np.int8)
    valid = np.ones(n_signals, dtype=np.bool_)
    mfe_points = np.zeros(n_signals, dtype=np.float32)
    mae_points = np.zeros(n_signals, dtype=np.float32)

    tp_f32 = np.float32(tp_points)
    sl_f32 = np.float32(sl_points)

    for s in prange(n_signals):
        entry_idx = signal_indices[s]
        entry_px = prices[entry_idx]
        d = signal_dirs[s]

        if d == 1:
            tp_px = entry_px + tp_f32
            sl_px = entry_px - sl_f32
        else:
            tp_px = entry_px - tp_f32
            sl_px = entry_px + sl_f32

        # Snap to tick
        tp_px = np.float32(round(tp_px / tick_size) * tick_size)
        sl_px = np.float32(round(sl_px / tick_size) * tick_size)

        mfe = np.float32(0.0)
        mae = np.float32(0.0)
        found = False
        for i in range(entry_idx + 1, n_prices):
            px = prices[i]

            # Track MFE/MAE
            if d == 1:
                fav = px - entry_px
                adv = entry_px - px
            else:
                fav = entry_px - px
                adv = px - entry_px
            if fav > mfe:
                mfe = fav
            if adv > mae:
                mae = adv

            # Check SL first (worst case assumption)
            if d == 1:
                if px <= sl_px:
                    exit_indices[s] = i
                    exit_prices[s] = sl_px
                    exit_reasons[s] = 1  # SL
                    found = True
                    break
                if px >= tp_px:
                    exit_indices[s] = i
                    exit_prices[s] = tp_px
                    exit_reasons[s] = 0  # TP
                    found = True
                    break
            else:
                if px >= sl_px:
                    exit_indices[s] = i
                    exit_prices[s] = sl_px
                    exit_reasons[s] = 1  # SL
                    found = True
                    break
                if px <= tp_px:
                    exit_indices[s] = i
                    exit_prices[s] = tp_px
                    exit_reasons[s] = 0  # TP
                    found = True
                    break

        if not found:
            valid[s] = False
            exit_indices[s] = -1
            exit_prices[s] = np.float32(0.0)
            exit_reasons[s] = -1

        mfe_points[s] = mfe
        mae_points[s] = mae

    return exit_indices, exit_prices, exit_reasons, valid, mfe_points, mae_points
