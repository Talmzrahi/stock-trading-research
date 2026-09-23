"""Regression tests for research/mechanics.py.

Each test encodes an error actually made in this project between
2026-09-21 and 2026-09-23. They are written so that reintroducing the
original mistake fails the suite, not so that they read prettily.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))

from mechanics import (  # noqa: E402
    hold_from_month_end, levered_return, month_end_mask, performance,
    position_book, trailing_stats, years_to_significance,
)


class TestWeightTiming(unittest.TestCase):
    """Error 1/2 class: a weight must never earn the return it was computed from."""

    def setUp(self):
        self.idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31",
                                     "2024-02-01", "2024-02-02"])

    def test_month_end_mask_uses_the_index_not_the_calendar(self):
        # 2024-01-31 is the last trading day present for January; 2024-02-02
        # is the last present for February, though it is not a calendar end.
        self.assertEqual(list(month_end_mask(self.idx)), [False, True, False, True])

    def test_weight_applies_from_the_day_after_it_is_set(self):
        w = hold_from_month_end([1.0, 2.0, 3.0, 4.0], self.idx)
        self.assertTrue(np.isnan(w.iloc[1]))          # not on the day it is set
        self.assertEqual(w.iloc[2], 2.0)              # the month-end value, next day
        self.assertEqual(w.iloc[3], 2.0)              # held until the next month end

    def test_weight_never_equals_its_own_day_value(self):
        raw = [10.0, 20.0, 30.0, 40.0]
        w = hold_from_month_end(raw, self.idx)
        for i, v in enumerate(raw):
            if not np.isnan(w.iloc[i]):
                self.assertNotEqual(w.iloc[i], v)


class TestTrailingStats(unittest.TestCase):
    """Error 2: full-sample statistics let a past decision consult the future."""

    def test_same_day_values_are_excluded(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
        mean, _ = trailing_stats(dates, [1.0, 1.0, 1.0, 99.0], min_obs=3)
        self.assertAlmostEqual(mean[3], 1.0)          # the 99 must not be in its own mean

    def test_future_values_cannot_change_the_past(self):
        dates = pd.to_datetime([f"2020-01-{d:02d}" for d in range(1, 11)])
        base = [1.0] * 10
        later = base.copy()
        later[9] = 1000.0
        m1, s1 = trailing_stats(dates, base, min_obs=3)
        m2, s2 = trailing_stats(dates, later, min_obs=3)
        np.testing.assert_allclose(m1[:9], m2[:9], equal_nan=True)
        np.testing.assert_allclose(s1[:9], s2[:9], equal_nan=True)

    def test_short_history_is_nan_not_a_guess(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
        mean, sd = trailing_stats(dates, [1.0, 2.0], min_obs=30)
        self.assertTrue(np.isnan(mean).all())
        self.assertTrue(np.isnan(sd).all())


class TestCostSign(unittest.TestCase):
    """Error 4: `side` applied AFTER the cost subtraction turned every
    short-side cost into a gain, overstating a strategy by ~0.9pp/yr."""

    def setUp(self):
        self.returns = np.zeros((20, 1))              # a flat asset: only costs move the book

    def test_cost_reduces_a_short_position(self):
        daily, _ = position_book([0], [0], self.returns, hold=5, side=-1, cost=0.001)
        self.assertLess(daily.sum(), 0.0)             # the original bug made this POSITIVE
        self.assertAlmostEqual(daily.sum(), -0.002)   # charged once on entry, once on exit

    def test_cost_reduces_a_long_position(self):
        daily, _ = position_book([0], [0], self.returns, hold=5, side=+1, cost=0.001)
        self.assertLess(daily.sum(), 0.0)
        self.assertAlmostEqual(daily.sum(), -0.002)

    def test_cost_hurts_both_sides_identically(self):
        lo, _ = position_book([0], [0], self.returns, hold=5, side=+1, cost=0.001)
        sh, _ = position_book([0], [0], self.returns, hold=5, side=-1, cost=0.001)
        self.assertAlmostEqual(lo.sum(), sh.sum())

    def test_side_flips_the_return_not_the_cost(self):
        rets = np.zeros((20, 1))
        rets[1:6, 0] = 0.01                           # asset rises while held
        lo, _ = position_book([0], [0], rets, hold=5, side=+1, cost=0.0)
        sh, _ = position_book([0], [0], rets, hold=5, side=-1, cost=0.0)
        self.assertAlmostEqual(lo.sum(), -sh.sum())
        self.assertGreater(lo.sum(), 0.0)
        self.assertLess(sh.sum(), 0.0)


class TestLeverageCost(unittest.TestCase):
    """Error 5: leverage charged nothing, which is not how borrowing works."""

    def test_borrowing_costs_money(self):
        r = levered_return(weight=2.0, asset_return=0.0, rf_daily=0.0,
                           margin_spread=0.0252)
        self.assertLess(r, 0.0)
        self.assertAlmostEqual(float(r), -0.0001, places=7)

    def test_idle_cash_earns_the_risk_free_rate(self):
        r = levered_return(weight=0.5, asset_return=0.0, rf_daily=0.0001,
                           margin_spread=0.015)
        self.assertAlmostEqual(float(r), 0.00005, places=9)

    def test_more_leverage_costs_more(self):
        kw = dict(asset_return=0.0, rf_daily=0.0, margin_spread=0.0252)
        self.assertLess(levered_return(weight=3.0, **kw), levered_return(weight=2.0, **kw))


class TestPerformance(unittest.TestCase):
    """Raw Sharpe flatters whichever strategy holds more cash."""

    def test_excess_sharpe_is_below_raw_when_cash_pays(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        r = pd.Series(0.0004, index=idx)
        rf = pd.Series(0.0001, index=idx)
        p = performance(r, rf_daily=rf)
        self.assertLess(p["sharpe"], p["sharpe_raw"])

    def test_both_sharpes_reported_so_neither_can_be_chosen_quietly(self):
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        p = performance(pd.Series(np.random.default_rng(0).normal(0, 0.01, 300), index=idx))
        self.assertIn("sharpe", p)
        self.assertIn("sharpe_raw", p)

    def test_drawdown_is_negative_after_a_fall(self):
        r = pd.Series([0.1, -0.5, 0.1])
        self.assertLess(performance(r)["max_dd"], -0.4)


class TestYearsToSignificance(unittest.TestCase):
    """Quoted before any shadow book is proposed as a 'test'."""

    def test_a_weak_strategy_needs_a_career(self):
        self.assertGreater(years_to_significance(0.36), 25)

    def test_a_strong_strategy_resolves_quickly(self):
        self.assertLess(years_to_significance(1.5), 2)

    def test_a_dead_strategy_never_resolves(self):
        self.assertEqual(years_to_significance(0.0), float("inf"))


if __name__ == "__main__":
    unittest.main()
