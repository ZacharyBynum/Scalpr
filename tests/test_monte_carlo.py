from __future__ import annotations

import numpy as np
import pytest

from scalpr_zen.monte_carlo import (
    _compute_max_drawdowns,
    _compute_stats,
    _downsample,
    _extract_percentile_curves,
    _simulate_equity_curves,
    run_monte_carlo,
    write_monte_carlo_html,
)
from scalpr_zen.types import (
    BacktestResult,
    Direction,
    ExitReason,
    Fill,
    MonteCarloResult,
)


def _make_fill(trade_number: int, pnl_dollars: float) -> Fill:
    return Fill(
        trade_number=trade_number,
        direction=Direction.LONG,
        entry_time=1_000_000_000 * trade_number,
        entry_price=20000.0,
        exit_time=1_000_000_000 * trade_number + 500_000_000,
        exit_price=20000.0 + (pnl_dollars / 20.0),
        pnl_points=pnl_dollars / 20.0,
        pnl_dollars=pnl_dollars,
        exit_reason=ExitReason.TP if pnl_dollars > 0 else ExitReason.SL,
        mfe_points=abs(pnl_dollars / 20.0),
        mae_points=0.0,
    )


def _make_result(pnls: list[float]) -> BacktestResult:
    fills = [_make_fill(i + 1, p) for i, p in enumerate(pnls)]
    return BacktestResult(
        success=True,
        error=None,
        strategy_name="Test Strategy",
        params={"instrument": "NQ", "tp_points": 10.0, "sl_points": 5.0},
        fills=fills,
        summary=None,
        buy_hold_equity=[],
    )


class TestSimulateEquityCurves:
    def test_shape(self):
        pnl = np.array([100.0, -50.0, 200.0, -30.0, 80.0])
        rng = np.random.default_rng(42)
        curves = _simulate_equity_curves(pnl, 100, rng)
        assert curves.shape == (100, 6)

    def test_starts_at_zero(self):
        pnl = np.array([100.0, -50.0, 200.0])
        rng = np.random.default_rng(42)
        curves = _simulate_equity_curves(pnl, 50, rng)
        assert np.all(curves[:, 0] == 0.0)

    def test_final_pnl_preserved(self):
        pnl = np.array([100.0, -50.0, 200.0, -30.0])
        expected_total = pnl.sum()
        rng = np.random.default_rng(42)
        curves = _simulate_equity_curves(pnl, 200, rng)
        np.testing.assert_allclose(curves[:, -1], expected_total, atol=1e-8)

    def test_seed_reproducibility(self):
        pnl = np.array([100.0, -50.0, 200.0, -30.0, 80.0])
        curves_a = _simulate_equity_curves(pnl, 100, np.random.default_rng(123))
        curves_b = _simulate_equity_curves(pnl, 100, np.random.default_rng(123))
        np.testing.assert_array_equal(curves_a, curves_b)


class TestComputeMaxDrawdowns:
    def test_no_drawdown(self):
        curves = np.array([[0.0, 100.0, 200.0, 300.0]])
        dds = _compute_max_drawdowns(curves)
        assert dds[0] == 0.0

    def test_full_drawdown(self):
        curves = np.array([[0.0, 100.0, -50.0]])
        dds = _compute_max_drawdowns(curves)
        assert dds[0] == 150.0

    def test_multiple_sims(self):
        curves = np.array([
            [0.0, 100.0, 50.0, 200.0],
            [0.0, -100.0, -200.0, -300.0],
        ])
        dds = _compute_max_drawdowns(curves)
        assert dds[0] == 50.0
        assert dds[1] == 300.0


class TestComputeStats:
    def test_all_winners(self):
        pnl = np.array([100.0, 200.0, 50.0, 150.0])
        rng = np.random.default_rng(42)
        curves = _simulate_equity_curves(pnl, 500, rng)
        stats = _compute_stats(curves, pnl)
        assert stats.probability_of_profit == 1.0
        assert stats.median_final_pnl == 500.0
        assert stats.original_final_pnl == 500.0

    def test_all_losers(self):
        pnl = np.array([-100.0, -200.0, -50.0])
        rng = np.random.default_rng(42)
        curves = _simulate_equity_curves(pnl, 500, rng)
        stats = _compute_stats(curves, pnl)
        assert stats.probability_of_profit == 0.0
        assert stats.median_final_pnl == -350.0

    def test_percentile_ordering(self):
        pnl = np.array([100.0, -50.0, 200.0, -30.0, 80.0, -10.0, 50.0, -40.0])
        rng = np.random.default_rng(42)
        curves = _simulate_equity_curves(pnl, 1000, rng)
        stats = _compute_stats(curves, pnl)
        assert stats.final_pnl_5th <= stats.final_pnl_25th
        assert stats.final_pnl_25th <= stats.median_final_pnl
        assert stats.median_final_pnl <= stats.final_pnl_75th
        assert stats.final_pnl_75th <= stats.final_pnl_95th
        # Drawdown percentiles should also be ordered
        assert stats.max_drawdown_5th <= stats.max_drawdown_25th
        assert stats.max_drawdown_25th <= stats.median_max_drawdown
        assert stats.median_max_drawdown <= stats.max_drawdown_75th
        assert stats.max_drawdown_75th <= stats.max_drawdown_95th


class TestDownsample:
    def test_no_downsample_needed(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _downsample(arr, 10)
        assert result == [1.0, 2.0, 3.0]

    def test_downsample(self):
        arr = np.arange(5000, dtype=np.float64)
        result = _downsample(arr, 100)
        assert len(result) == 100
        assert result[0] == 0.0
        assert result[-1] == 4999.0


class TestExtractPercentileCurves:
    def test_curve_lengths_match(self):
        pnl = np.array([100.0, -50.0, 200.0, -30.0, 80.0])
        rng = np.random.default_rng(42)
        curves = _simulate_equity_curves(pnl, 100, rng)
        c5, c25, c50, c75, c95, orig = _extract_percentile_curves(curves, pnl)
        assert len(c5) == len(c25) == len(c50) == len(c75) == len(c95) == len(orig)
        # n_trades + 1 (includes starting 0)
        assert len(c50) == 6

    def test_percentile_curve_ordering(self):
        pnl = np.array([100.0, -50.0, 200.0, -30.0, 80.0, -10.0] * 10)
        rng = np.random.default_rng(42)
        curves = _simulate_equity_curves(pnl, 500, rng)
        c5, c25, c50, c75, c95, _ = _extract_percentile_curves(curves, pnl)
        for i in range(len(c5)):
            assert c5[i] <= c25[i] + 1e-8
            assert c25[i] <= c50[i] + 1e-8
            assert c50[i] <= c75[i] + 1e-8
            assert c75[i] <= c95[i] + 1e-8


class TestRunMonteCarlo:
    def test_success(self):
        result = _make_result([100.0, -50.0, 200.0, -30.0, 80.0])
        mc = run_monte_carlo(result, n_simulations=100, seed=42)
        assert mc.success is True
        assert mc.error is None
        assert mc.stats is not None
        assert mc.stats.n_simulations == 100

    def test_failed_backtest(self):
        result = BacktestResult(
            success=False,
            error="No data",
            strategy_name="Test",
            params={},
            fills=[],
            summary=None,
            buy_hold_equity=[],
        )
        mc = run_monte_carlo(result)
        assert mc.success is False
        assert "no fills" in mc.error.lower()

    def test_empty_fills(self):
        result = BacktestResult(
            success=True,
            error=None,
            strategy_name="Test",
            params={},
            fills=[],
            summary=None,
            buy_hold_equity=[],
        )
        mc = run_monte_carlo(result)
        assert mc.success is False

    def test_single_fill(self):
        result = _make_result([100.0])
        mc = run_monte_carlo(result)
        assert mc.success is False
        assert "at least 2" in mc.error.lower()

    def test_seed_reproducibility(self):
        result = _make_result([100.0, -50.0, 200.0, -30.0, 80.0])
        mc_a = run_monte_carlo(result, n_simulations=100, seed=42)
        mc_b = run_monte_carlo(result, n_simulations=100, seed=42)
        assert mc_a.curve_50th == mc_b.curve_50th
        assert mc_a.stats.median_final_pnl == mc_b.stats.median_final_pnl

    def test_curve_length(self):
        pnls = [100.0, -50.0, 200.0, -30.0, 80.0]
        result = _make_result(pnls)
        mc = run_monte_carlo(result, n_simulations=50, seed=42)
        assert len(mc.original_curve) == len(pnls) + 1
        assert len(mc.curve_50th) == len(pnls) + 1

    def test_original_curve_values(self):
        pnls = [100.0, -50.0, 200.0]
        result = _make_result(pnls)
        mc = run_monte_carlo(result, n_simulations=50, seed=42)
        assert mc.original_curve[0] == 0.0
        assert mc.original_curve[1] == 100.0
        assert mc.original_curve[2] == 50.0
        assert mc.original_curve[3] == 250.0


class TestWriteHtml:
    def test_writes_file(self, tmp_path):
        result = _make_result([100.0, -50.0, 200.0, -30.0, 80.0])
        mc = run_monte_carlo(result, n_simulations=50, seed=42)
        path = write_monte_carlo_html(mc, output_dir=str(tmp_path))
        assert path.endswith(".html")
        content = open(path, encoding="utf-8").read()
        assert "Monte Carlo" in content
        assert "Chart" in content
        assert "Test Strategy" in content
