import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from trader.backtest import Market, run
from trader.broker import Ledger
from trader.config import Config
from trader.engine import Engine, EngineState
from trader.events import build_events
from trader.fusion import Fusion
from trader.market_calendar import trading_days
from trader.monitor import bootstrap_band, first_fire, tripwire_status
from trader.portfolio import benchmark_trade, size_entries, slot_count
from trader.signals.base import Signal

N = 200
CFG = Config(hold_days=10, min_slots=5, initial_capital=1000.0)


def make_market(prices, events):
    cal = trading_days("2020-01-01", "2021-12-31")[:N]
    closes = pd.DataFrame({s: (v if isinstance(v, np.ndarray) else np.full(N, float(v)))
                           for s, v in prices.items()}, index=cal)
    ev = pd.DataFrame([dict(symbol=s, key=f"{s}|{i}", entry_idx=i, entry_date=cal[i],
                            pit=True, conviction=c) for s, i, c in events])
    return Market(cal, closes, ev)


class EngineTest(unittest.TestCase):
    def test_entry_then_time_exit(self):
        m = make_market({"SPY": 100, "AAA": 50}, [("AAA", 20, 0.95)])
        res = run(m, CFG, start=5, end=60)
        self.assertEqual(len(res.trades), 1)
        t = res.trades.iloc[0]
        self.assertEqual(t.entry_date, m.cal[20])
        self.assertEqual(t.exit_date, m.cal[30])
        self.assertEqual(t.reason, "time")
        # 1/5 of equity went into the stock on entry day.
        stock_share = res.equity.stocks.iloc[20 - 5] / res.equity.equity.iloc[20 - 5]
        self.assertAlmostEqual(stock_share, 0.2, delta=0.01)
        # flat prices: only costs are lost
        self.assertTrue(990 < res.equity.equity.iloc[-1] < 1000)

    def test_below_cutoff_is_not_traded(self):
        m = make_market({"SPY": 100, "AAA": 50}, [("AAA", 20, 0.50)])
        res = run(m, CFG, start=5, end=60)
        self.assertEqual(len(res.trades), 0)
        self.assertEqual(res.equity.n_positions.max(), 0)

    def test_retrigger_resets_the_clock(self):
        m = make_market({"SPY": 100, "AAA": 50}, [("AAA", 20, 0.95), ("AAA", 25, 0.99)])
        res = run(m, CFG, start=5, end=80)
        self.assertEqual(len(res.trades), 1)
        self.assertEqual(res.trades.iloc[0].exit_date, m.cal[35])
        self.assertEqual(res.trades.iloc[0].resets, 1)

    def test_vol_trailing_stop_exits_day_after_the_break(self):
        # ±1% zigzag: ~1% daily vol but never a 3-sd drawdown until the break
        aaa = 50 * np.cumprod(np.where(np.arange(N) % 2, 0.99, 1.0101))
        aaa[31:] = aaa[30] * 0.7
        m = make_market({"SPY": 100, "AAA": aaa}, [("AAA", 20, 0.95)])
        res = run(m, replace(CFG, hold_days=100, stop_k=3), start=5, end=60)
        t = res.trades.iloc[0]
        self.assertEqual(t.reason, "stop")
        self.assertEqual(t.exit_date, m.cal[32])   # break seen at close 31, sold at close 32

    def test_idle_cash_sits_in_benchmark_and_never_goes_negative(self):
        evs = [(s, 20 + i, 0.95) for i, s in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"])]
        prices = {"SPY": np.linspace(100, 130, N)} | {s: 40 for s, _, _ in evs}
        m = make_market(prices, evs)
        res = run(m, CFG, start=5, end=100)
        self.assertTrue((res.equity.cash >= -1e-9).all())
        self.assertGreater(res.equity.bench.iloc[0] / res.equity.equity.iloc[0], 0.98)
        # 7 entries at 1/5 each can't all be funded: exposure capped near 100%
        self.assertLessEqual((res.equity.stocks / res.equity.equity).max(), 1.0 + 1e-6)

    def test_late_events_are_taken_within_limit_and_reported_beyond(self):
        m = make_market({"SPY": 100, "AAA": 50, "BBB": 50},
                        [("AAA", 28, 0.95), ("BBB", 25, 0.95)])
        eng = Engine(CFG, m.cal, m.closes, m.events)
        ledger = Ledger(1000, lambda s: 5.0)
        state = EngineState()
        decisions = eng.step(30, state, ledger, lambda s: 50.0)
        acts = {d.symbol: d.action for d in decisions}
        self.assertEqual(acts["AAA"], "enter")       # 2 sessions late: allowed
        self.assertEqual(acts["BBB"], "too_late")    # 5 sessions late: skipped


class PartsTest(unittest.TestCase):
    def test_ledger_sells_fund_buys_and_buys_never_overdraw(self):
        led = Ledger(0, lambda s: 0.0)
        led._pos["SPY"] = 10.0
        led.submit("d", "AAA", "buy", notional=2000, tag="entry")
        led.submit("d", "SPY", "sell", notional=500)
        fills = led.settle("d", lambda s: 100.0)
        self.assertEqual([f.side for f in fills], ["sell", "buy"])
        self.assertAlmostEqual(led.cash(), 0.0)
        self.assertAlmostEqual(led.holdings()["AAA"], 5.0)

    def test_sizing(self):
        self.assertEqual(slot_count(0, CFG), 5)
        self.assertEqual(slot_count(137, Config()), 33)
        self.assertAlmostEqual(size_entries(1000, 2, 5, 1000, CFG), 200)
        self.assertAlmostEqual(size_entries(1000, 10, 5, 1000, CFG), 100)
        self.assertEqual(size_entries(1000, 3, 5, 1.5, CFG), 0.0)   # below $1 each
        self.assertAlmostEqual(benchmark_trade(100, 1000, 0, CFG), 95)
        self.assertAlmostEqual(benchmark_trade(-50, 1000, 30, CFG), -30)
        self.assertEqual(benchmark_trade(7, 1000, 500, CFG), 0.0)

    def test_events_timing_and_membership(self):
        cal = trading_days("2024-01-02", "2024-01-31")
        closes = pd.DataFrame({"AAA": 10.0, "BBB": 20.0}, index=cal)
        earn = pd.DataFrame({
            "symbol": ["AAA", "BBB"],
            "announced_at": pd.to_datetime(["2024-01-10 07:00", "2024-01-10 16:00"])
                              .tz_localize("America/New_York").tz_convert("UTC"),
            "eps_estimate": [1.0, 1.0], "eps_actual": [1.2, 0.8]})
        uni = pd.DataFrame({"as_of": pd.to_datetime(["2024-01-01"]), "symbol": ["AAA"]})
        ev = build_events(earn, cal, closes, uni).set_index("symbol")
        jan10 = cal.get_loc(pd.Timestamp("2024-01-10"))
        self.assertEqual(ev.loc["AAA", "entry_idx"], jan10)        # before open: same day
        self.assertEqual(ev.loc["BBB", "entry_idx"], jan10 + 1)    # after close: next day
        self.assertEqual(ev.loc["AAA", "signal_idx"], jan10 - 1)
        self.assertTrue(ev.loc["AAA", "pit"])
        self.assertFalse(ev.loc["BBB", "pit"])

    def test_fusion_single_signal_passthrough(self):
        class Const(Signal):
            name = "c"
            def score(self, events):
                return pd.Series([0.2, np.nan, 0.9], index=events.index)
        out = Fusion([Const()]).score(pd.DataFrame(index=[0, 1, 2]))
        np.testing.assert_array_equal(out.conviction.to_numpy(), [0.2, np.nan, 0.9])
        self.assertTrue(out.agree.all())

    def test_tripwire(self):
        rng = np.random.default_rng(1)
        dates = pd.date_range("2020-01-01", periods=400)
        trades = pd.DataFrame({"entry_date": dates, "alpha": rng.normal(0.01, 0.05, 400)})
        band = bootstrap_band(trades, n_max=100, draws=500)
        self.assertEqual(tripwire_status([0.0] * 5, band)["state"], "warming_up")
        self.assertEqual(tripwire_status([-0.05] * 40, band)["state"], "FIRED")
        self.assertEqual(tripwire_status([0.01] * 40, band)["state"], "ok")
        self.assertIsNone(first_fire([0.02] * 50, band))


if __name__ == "__main__":
    unittest.main()
