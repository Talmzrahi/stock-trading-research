import tempfile
import unittest
from pathlib import Path

from trader import state as st
from trader.config import Config
from trader.engine import EngineState
from trader.exits import Position
from trader.market_calendar import trading_days


class StateRoundTripTest(unittest.TestCase):
    def test_ledger_and_book_survive_a_restart(self):
        cfg = Config()
        cal = trading_days("2026-01-02", "2026-12-31")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trader.db"

            conn = st.open_state(path)
            led = st.load_ledger(conn, cfg)
            self.assertEqual(led.cash(), cfg.initial_capital)
            led.submit("2026-03-02", "SPY", "buy", notional=500, tag="benchmark")
            led.settle("2026-03-02", lambda s: 100.0)
            led.submit("2026-03-03", "AAA", "buy", notional=100, tag="entry:AAA|2026-03-03")
            book = EngineState()
            book.positions["BBB"] = Position("BBB", "BBB|2026-02-10", 30, 95, 0.99, 1, 2.0, 50.0, 1)
            book.pending_entries["AAA"] = Position("AAA", "AAA|2026-03-03", 40, 100, 0.985)
            book.processed = {"AAA|2026-03-03", "BBB|2026-02-10"}
            st.save_ledger(conn, led)
            st.save_engine_state(conn, book, cal, "2026-03-03", [])
            st.record_run(conn, "2026-03-03", "t1", "decided")
            st.record_run(conn, "2026-03-03", "t2", "settled")
            conn.commit()
            conn.close()

            conn = st.open_state(path)
            led2 = st.load_ledger(conn, cfg)
            self.assertAlmostEqual(led2.cash(), led.cash())
            self.assertAlmostEqual(led2.holdings()["SPY"], led.holdings()["SPY"])
            self.assertEqual([o.symbol for o in led2.pending()], ["AAA"])
            self.assertEqual(led2._next_id, 3)
            book2 = st.load_engine_state(conn, cal)
            b = book2.positions["BBB"]
            self.assertEqual((b.entry_idx, b.exit_due_idx, b.resets, b.late_days), (30, 95, 1, 1))
            self.assertIn("AAA", book2.pending_entries)
            self.assertEqual(book2.processed, book.processed)
            self.assertEqual(st.run_status(conn, "2026-03-03"), "decided")   # never downgraded
            conn.close()


if __name__ == "__main__":
    unittest.main()
