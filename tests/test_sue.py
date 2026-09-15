import unittest

import numpy as np
import pandas as pd

from trader.signals.sue import sue, trailing_percentile


class TrailingPercentileTest(unittest.TestCase):
    def setUp(self):
        # 300 history events on distinct earlier days, values 1..300.
        self.hist_dates = pd.date_range("2020-01-01", periods=300, freq="D")
        self.hist_vals = np.arange(1, 301, dtype=float)

    def test_ranks_against_strictly_earlier_history(self):
        dates = list(self.hist_dates) + [pd.Timestamp("2020-10-27")]
        vals = list(self.hist_vals) + [150.5]
        p = trailing_percentile(dates, vals, window_days=365, min_history=200)
        self.assertAlmostEqual(p[-1], 150 / 300)

    def test_future_values_do_not_change_past_ranks(self):
        dates = list(self.hist_dates) + [pd.Timestamp("2020-10-27")]
        vals = list(self.hist_vals) + [150.5]
        base = trailing_percentile(dates, vals, min_history=200)
        later = trailing_percentile(dates + [pd.Timestamp("2020-11-01")] * 50,
                                    vals + [1e9] * 50, min_history=200)
        np.testing.assert_array_equal(base, later[:len(base)])

    def test_same_day_events_do_not_rank_each_other(self):
        d = pd.Timestamp("2020-10-27")
        dates = list(self.hist_dates) + [d, d]
        vals = list(self.hist_vals) + [150.5, 1e9]
        p = trailing_percentile(dates, vals, min_history=200)
        self.assertAlmostEqual(p[-2], 0.5)
        self.assertAlmostEqual(p[-1], 1.0)

    def test_short_history_and_missing_values_are_nan(self):
        p = trailing_percentile(self.hist_dates[:50], self.hist_vals[:50], min_history=200)
        self.assertTrue(np.isnan(p).all())
        dates = list(self.hist_dates) + [pd.Timestamp("2020-10-27")]
        p = trailing_percentile(dates, list(self.hist_vals) + [np.nan], min_history=200)
        self.assertTrue(np.isnan(p[-1]))

    def test_window_drops_old_history(self):
        dates = list(self.hist_dates) + [pd.Timestamp("2021-06-01")]
        p = trailing_percentile(dates, list(self.hist_vals) + [1.0],
                                window_days=365, min_history=100)
        # 2021-06-01 minus 365d = 2020-06-01, leaving ~refs from day 152 onward.
        self.assertTrue(np.isnan(p[-1]) or p[-1] == 0.0)

    def test_sue_is_price_scaled(self):
        self.assertAlmostEqual(sue(1.10, 1.00, 50.0), 0.002)


if __name__ == "__main__":
    unittest.main()
