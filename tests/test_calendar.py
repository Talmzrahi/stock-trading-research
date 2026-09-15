import sqlite3
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from trader import market_calendar as cal

DB = Path(__file__).resolve().parent.parent / "data" / "research.db"


class CalendarTest(unittest.TestCase):
    def test_known_days(self):
        self.assertFalse(cal.is_trading_day(date(2026, 7, 3)))    # July 4 on Saturday
        self.assertFalse(cal.is_trading_day(date(2026, 11, 26)))  # Thanksgiving
        self.assertFalse(cal.is_trading_day(date(2026, 4, 3)))    # Good Friday
        self.assertTrue(cal.is_trading_day(date(2026, 9, 15)))
        self.assertTrue(cal.is_early_close(date(2026, 11, 27)))
        self.assertFalse(cal.is_trading_day(date(2022, 1, 1)))    # Saturday, not observed Friday
        self.assertTrue(cal.is_trading_day(date(2021, 12, 31)))

    @unittest.skipUnless(DB.exists(), "research.db not present")
    def test_matches_every_spy_session(self):
        conn = sqlite3.connect(DB)
        spy = pd.to_datetime([r[0] for r in conn.execute(
            "SELECT date FROM prices WHERE symbol='SPY' ORDER BY date")])
        conn.close()
        rule = cal.trading_days(spy.min(), spy.max())
        missing = spy.difference(rule)
        extra = rule.difference(spy)
        self.assertEqual(list(missing), [], "SPY traded on days the calendar calls closed")
        self.assertEqual(list(extra), [], "calendar has sessions SPY has no bar for")


if __name__ == "__main__":
    unittest.main()
