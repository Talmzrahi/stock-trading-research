import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
from inference_check import HOLD, calendar_time, quarter_clustered  # noqa: E402


class QuarterClusteredTest(unittest.TestCase):
    def test_matches_cr1_by_hand(self):
        rng = np.random.default_rng(1)
        dates = pd.bdate_range("2015-01-01", periods=400)
        means = pd.Series(rng.normal(0.01, 0.05, len(dates)), index=dates)
        est, p, G = quarter_clustered(means)

        q = pd.PeriodIndex(dates, freq="Q")
        u = means - means.mean()
        sums = u.groupby(q).sum()
        n = len(means)
        se = np.sqrt(G / (G - 1) * (sums ** 2).sum() / n ** 2)
        self.assertAlmostEqual(est, means.mean(), places=12)
        self.assertEqual(G, sums.size)
        self.assertAlmostEqual(p, 2 * stats.t.sf(abs(means.mean() / se), G - 1), places=10)


class CalendarTimeTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2020-01-01", periods=200)
        rng = np.random.default_rng(2)
        self.returns = rng.normal(0, 0.01, (len(self.dates), 3))   # cols: A, B, bench

    def test_single_event_is_its_own_returns_less_cost(self):
        monthly = calendar_time([10], [0], [2], self.returns, self.dates, cost=0.002)
        window = slice(11, 11 + HOLD)
        port = self.returns[window, 0].copy()
        port[0] -= 0.002
        daily = pd.DataFrame({"port": port, "bench": self.returns[window, 2]},
                             index=self.dates[window])
        month = daily.index.to_period("M")
        expected = (1 + daily).groupby(month).prod() - 1
        expected = expected[daily.groupby(month).size() >= 5]
        pd.testing.assert_frame_equal(monthly, expected)

    def test_overlapping_events_are_equal_weighted(self):
        monthly = calendar_time([10, 30], [0, 1], [2, 2], self.returns, self.dates, cost=0.0)
        both = self.returns[31:71, :2].mean(axis=1)          # days both are open
        only_a = self.returns[11:31, 0]
        only_b = self.returns[71:91, 1]
        port = np.concatenate([only_a, both, only_b])
        daily = pd.Series(port, index=self.dates[11:91])
        month = daily.index.to_period("M")
        expected = (1 + daily).groupby(month).prod() - 1
        expected = expected[daily.groupby(month).size() >= 5]
        np.testing.assert_allclose(monthly.port.to_numpy(), expected.to_numpy())

    def test_months_with_under_five_open_days_are_dropped(self):
        monthly = calendar_time([30], [0], [2], self.returns, self.dates, cost=0.0)
        last = self.dates[30 + HOLD]
        in_last_month = sum(d.to_period("M") == last.to_period("M") for d in self.dates[31:91])
        self.assertEqual(last.to_period("M") in monthly.index, in_last_month >= 5)

    def test_days_with_nothing_open_are_left_out(self):
        monthly = calendar_time([10], [0], [2], self.returns, self.dates, cost=0.0)
        self.assertEqual(monthly.index.min(), self.dates[11].to_period("M"))
        self.assertEqual(monthly.index.max(), self.dates[10 + HOLD].to_period("M"))


if __name__ == "__main__":
    unittest.main()
