# ═══════════════════════════════════════════════════════════════════════
#  Layer 6 — execution.
#
#  Broker is the interface the engine talks to. Ledger is an in-memory
#  broker that fills market-on-close orders at the closing price plus a
#  per-side cost; the backtest uses it directly and SimBroker persists it
#  to SQLite for the live paper account. An Alpaca broker would implement
#  the same three methods.
# ═══════════════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Order:
    id: int
    when: str                     # session date the order fills at the close of
    symbol: str
    side: str                     # 'buy' | 'sell'
    qty: float | None = None      # sell: None with no notional = whole position
    notional: float | None = None
    tag: str = ""
    status: str = "pending"       # pending | filled | rejected
    fill_price: float | None = None
    fill_qty: float | None = None
    broker_id: str | None = None  # the external broker's order id, once sent


class Broker(ABC):
    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def holdings(self) -> dict: ...

    @abstractmethod
    def submit(self, when, symbol, side, qty=None, notional=None, tag="") -> Order: ...


class Ledger(Broker):
    def __init__(self, cash, cost_bps, min_order=1.0, keep_history=True):
        self._cash = float(cash)
        self._pos = {}
        self._pending = []
        self.history = []
        self.cost_bps = cost_bps            # symbol -> bps per side
        self.min_order = min_order
        self.keep_history = keep_history
        self._next_id = 1

    def cash(self):
        return self._cash

    def holdings(self):
        return dict(self._pos)

    def pending(self):
        return list(self._pending)

    def submit(self, when, symbol, side, qty=None, notional=None, tag=""):
        o = Order(self._next_id, when, symbol, side, qty, notional, tag)
        self._next_id += 1
        self._pending.append(o)
        return o

    def settle(self, when, price):
        """Fill every pending order for session `when` at price(symbol).

        Sells go first so their proceeds fund the buys; a buy larger than
        the cash left is cut down to it, and rejected below min_order.
        """
        todo = [o for o in self._pending if o.when == when]
        self._pending = [o for o in self._pending if o.when != when]
        filled = []
        for o in sorted(todo, key=lambda o: (o.side != "sell", o.id)):
            px = price(o.symbol)
            if px is None or not px > 0:
                self._close(o, "rejected")
                continue
            bps = self.cost_bps(o.symbol) / 1e4
            if o.side == "sell":
                held = self._pos.get(o.symbol, 0.0)
                if o.qty is None and o.notional is None:
                    qty = held
                else:
                    qty = min(o.qty if o.qty is not None else o.notional / px, held)
                if qty <= 0:
                    self._close(o, "rejected")
                    continue
                o.fill_price = px * (1 - bps)
                self._cash += qty * o.fill_price
                left = held - qty
                if left <= held * 1e-9:
                    self._pos.pop(o.symbol, None)
                else:
                    self._pos[o.symbol] = left
            else:
                notional = min(o.notional, self._cash)
                if notional < self.min_order:
                    self._close(o, "rejected")
                    continue
                o.fill_price = px * (1 + bps)
                qty = notional / o.fill_price
                self._cash -= notional
                self._pos[o.symbol] = self._pos.get(o.symbol, 0.0) + qty
            o.fill_qty = qty
            self._close(o, "filled")
            filled.append(o)
        return filled

    def apply_dividend(self, symbol, per_share):
        self._cash += self._pos.get(symbol, 0.0) * per_share

    def apply_split(self, symbol, ratio):
        if symbol in self._pos:
            self._pos[symbol] *= ratio

    def _close(self, o, status):
        o.status = status
        if self.keep_history:
            self.history.append(o)
