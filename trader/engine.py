# ═══════════════════════════════════════════════════════════════════════
#  The daily decision step, shared by the backtest and the live run.
#
#  step(t) runs "before the close" of session t: it sees closes through
#  t-1 and every event whose entry session has arrived, and submits
#  market-on-close orders for session t. on_fills(t) books the results.
#  The backtest calls both every session; the live run calls step today
#  and on_fills tomorrow, once today's close is known.
# ═══════════════════════════════════════════════════════════════════════

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .exits import Position, rules_for
from .portfolio import benchmark_trade, size_entries, slot_count

LOOKBACK = 10        # sessions of unprocessed events to scan (reports too-late ones)


@dataclass
class Decision:
    date: str
    symbol: str
    action: str
    key: str = ""
    conviction: float = float("nan")
    detail: str = ""


@dataclass
class Trade:
    symbol: str
    key: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    hold_days: int
    conviction: float
    reason: str
    late_days: int
    resets: int
    entry_fill: float
    exit_fill: float
    ret: float           # close-to-close, adjusted
    bench_ret: float
    alpha: float         # ret - bench_ret - round-trip cost


@dataclass
class EngineState:
    positions: dict = field(default_factory=dict)
    processed: set = field(default_factory=set)
    pending_entries: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)


def _v(x):
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else x


class Engine:
    def __init__(self, cfg, calendar, closes, events):
        self.cfg = cfg
        self.cal = pd.DatetimeIndex(calendar)
        self.dates = list(self.cal.strftime("%Y-%m-%d"))
        closes = closes.reindex(self.cal)
        if cfg.benchmark not in closes.columns:
            raise ValueError(f"benchmark {cfg.benchmark} has no prices")
        self.col = {s: i for i, s in enumerate(closes.columns)}
        self.px = closes.to_numpy()
        self.ff = closes.ffill().to_numpy()
        self.exits = rules_for(cfg)
        for rule in self.exits:
            rule.prepare(closes)

        ev = events[events.pit & events.conviction.notna() & events.symbol.isin(list(self.col))]
        ev = ev.sort_values(["entry_idx", "conviction"], ascending=[True, False])
        self.ev_entry = ev.entry_idx.to_numpy()
        self.ev_symbol = ev.symbol.to_numpy()
        self.ev_key = ev.key.to_numpy()
        self.ev_conv = ev.conviction.to_numpy()
        self.q_dates = np.sort(ev.entry_date[ev.conviction >= cfg.cutoff].to_numpy())

    def default_start(self):
        """First session with a full trailing year of qualifying signals, so
        slot sizing is estimated rather than defaulted."""
        if not len(self.q_dates):
            raise ValueError("no qualifying events")
        first = pd.Timestamp(self.q_dates[0]) + pd.Timedelta(days=self.cfg.window_days)
        return int(self.cal.searchsorted(first))

    def recent_signals(self, t):
        d = self.cal[t].to_datetime64()
        lo = np.searchsorted(self.q_dates, d - np.timedelta64(self.cfg.window_days, "D"), "left")
        return int(np.searchsorted(self.q_dates, d, "left") - lo)

    def step(self, t, state, broker, mark):
        """Decide and submit orders for the close of session t.
        mark(symbol) values holdings at the last known close."""
        cfg, ds, bench = self.cfg, self.dates[t], self.cfg.benchmark
        out = []
        holdings, cash = broker.holdings(), broker.cash()
        marks = {s: _v(mark(s)) for s in holdings}
        equity = cash + sum(q * marks[s] for s, q in holdings.items())

        exiting = {}
        for sym, pos in state.positions.items():
            for rule in self.exits:
                if rule.check(pos, t, self):
                    exiting[sym] = rule.name
                    break

        entries = []
        lo = np.searchsorted(self.ev_entry, t - cfg.max_late_days - LOOKBACK, "left")
        hi = np.searchsorted(self.ev_entry, t, "right")
        for i in range(lo, hi):
            key = self.ev_key[i]
            if key in state.processed:
                continue
            state.processed.add(key)
            sym, conv, late = self.ev_symbol[i], float(self.ev_conv[i]), int(t - self.ev_entry[i])
            if conv < cfg.cutoff:
                out.append(Decision(ds, sym, "below_cutoff", key, conv))
            elif late > cfg.max_late_days:
                out.append(Decision(ds, sym, "too_late", key, conv, f"{late} sessions late"))
            elif sym == bench or any(e[0] == sym for e in entries):
                continue
            elif sym in state.positions:
                if exiting.get(sym) in ("stop", "no_data"):
                    out.append(Decision(ds, sym, "retrigger_ignored", key, conv,
                                        f"exiting on {exiting[sym]}"))
                    continue
                exiting.pop(sym, None)
                pos = state.positions[sym]
                pos.exit_due_idx = t + cfg.hold_days
                pos.resets += 1
                out.append(Decision(ds, sym, "reset_clock", key, conv,
                                    f"hold restarts: {cfg.hold_days} sessions from today"))
            else:
                entries.append((sym, key, conv, late))

        bench_value = holdings.get(bench, 0.0) * _v(mark(bench))
        proceeds = sum(holdings.get(s, 0.0) * marks.get(s, 0.0) for s in exiting)
        spendable = cash + bench_value + proceeds - cfg.cash_buffer * equity
        slots = slot_count(self.recent_signals(t), cfg)
        per = size_entries(equity, len(entries), slots, spendable, cfg)

        for sym, reason in exiting.items():
            broker.submit(ds, sym, "sell", tag=f"exit:{reason}")
            out.append(Decision(ds, sym, f"exit_{reason}", state.positions[sym].event_key,
                                state.positions[sym].conviction))
        spent = 0.0
        for sym, key, conv, late in entries:
            if per <= 0:
                out.append(Decision(ds, sym, "no_cash", key, conv))
                continue
            broker.submit(ds, sym, "buy", notional=per, tag=f"entry:{key}")
            state.pending_entries[sym] = Position(sym, key, t, t + cfg.hold_days, conv, late)
            spent += per
            note = f"${per:,.2f}, 1/{slots} of equity" + (f", {late}d late" if late else "")
            out.append(Decision(ds, sym, "enter", key, conv, note))

        trade = benchmark_trade(cash + proceeds - spent, equity, bench_value, cfg)
        if abs(trade) >= cfg.min_order:
            broker.submit(ds, bench, "buy" if trade > 0 else "sell", notional=abs(trade),
                          tag="benchmark")
        return out

    def on_fills(self, t, fills, state):
        """Book fills for session t. Rejected exits leave the position open,
        so the next step simply tries again."""
        cfg, bc = self.cfg, self.col[self.cfg.benchmark]
        for o in fills:
            if o.tag.startswith("entry:"):
                pos = state.pending_entries.pop(o.symbol, None)
                if pos is not None:
                    pos.qty, pos.entry_fill = o.fill_qty, o.fill_price
                    state.positions[o.symbol] = pos
            elif o.tag.startswith("exit:"):
                pos = state.positions.pop(o.symbol, None)
                if pos is None:
                    continue
                c = self.col[o.symbol]
                ret = self.ff[t, c] / self.ff[pos.entry_idx, c] - 1
                bret = self.ff[t, bc] / self.ff[pos.entry_idx, bc] - 1
                state.trades.append(Trade(
                    o.symbol, pos.event_key, self.cal[pos.entry_idx], self.cal[t],
                    t - pos.entry_idx, pos.conviction, o.tag[5:], pos.late_days, pos.resets,
                    pos.entry_fill, o.fill_price, ret, bret,
                    ret - bret - 2 * cfg.cost_bps_side / 1e4))
        state.pending_entries.clear()
