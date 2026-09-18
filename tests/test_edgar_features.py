import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
from edgar_features import WINSOR, own_history_z  # noqa: E402


def history(values, cik="1"):
    return pd.DataFrame({"cik": cik, "value": values,
                         "filed_date": pd.date_range("2015-01-01", periods=len(values),
                                                     freq="91D")})


class OwnHistoryZTest(unittest.TestCase):
    def test_scored_from_the_min_prior_th_filing(self):
        # Regression: a NaN from shift() left the first 12 unscored.
        z = own_history_z(history([1.0, 2, 3, 4, 5, 6, 7, 8]), ["value"],
                          min_prior=6, window=12)["value_z"]
        self.assertTrue(z.iloc[:6].isna().all())
        self.assertTrue(z.iloc[6:].notna().all())

    def test_uses_only_earlier_filings(self):
        # Prior six are 1..6: median 3.5, MAD 1.5 -> scale 2.2239.
        z = own_history_z(history([1.0, 2, 3, 4, 5, 6, 10]), ["value"],
                          min_prior=6, window=12)["value_z"]
        self.assertAlmostEqual(z.iloc[6], (10 - 3.5) / (1.5 * 1.4826), places=6)

    def test_constant_history_is_a_norm_not_missing(self):
        z = own_history_z(history([0.5] * 8), ["value"], min_prior=6, window=12)["value_z"]
        self.assertEqual(z.iloc[6], 0.0)
        self.assertEqual(z.iloc[7], 0.0)

    def test_departure_from_a_never_varying_norm_hits_the_cap(self):
        z = own_history_z(history([0.0] * 6 + [0.01, -0.01]), ["value"],
                          min_prior=6, window=12)["value_z"]
        self.assertEqual(z.iloc[6], WINSOR)

    def test_zero_mad_falls_back_to_mean_absolute_deviation(self):
        # 91-day gaps with one 98: MAD is 0, mean abs deviation 7/6.
        z = own_history_z(history([91.0] * 5 + [98, 92]), ["value"],
                          min_prior=6, window=12)["value_z"]
        self.assertAlmostEqual(z.iloc[6], 1 / (7 / 6 * 1.2533), places=6)

    def test_companies_do_not_share_history(self):
        df = pd.concat([history([100.0] * 8, cik="A"), history([1.0] * 8, cik="B")],
                       ignore_index=True)
        z = own_history_z(df, ["value"], min_prior=6, window=12)["value_z"]
        self.assertTrue((z.dropna() == 0).all())


if __name__ == "__main__":
    unittest.main()
