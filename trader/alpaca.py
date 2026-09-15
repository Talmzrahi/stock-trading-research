# ═══════════════════════════════════════════════════════════════════════
#  Layer 6 — Alpaca paper-trading broker.
#
#  Implements the Broker interface the engine uses (cash, holdings,
#  submit) plus the transmit/settle bookkeeping run_daily needs. Paper
#  only: a non-paper URL is refused — live trading is Phase 4, deferred on
#  purpose.
#
#  Order type. Alpaca accepts fractional and notional orders only with
#  time_in_force=day (no market-on-close), and a $100-$1,000 account needs
#  fractional shares. So engine orders are queued during the run and sent
#  as market day orders shortly before the close; the report measures each
#  fill against the official close. Sells go first and buys wait for them
#  to fill, so proceeds fund purchases.
#
#  Keys: ALPACA_API_KEY and ALPACA_SECRET_KEY (paper account) environment
#  variables. Alpaca books dividends and splits itself.
# ═══════════════════════════════════════════════════════════════════════

import os
import time

import requests

from .broker import Broker, Order

PAPER_URL   = "https://paper-api.alpaca.markets"
DONE_STATES = {"filled", "canceled", "expired", "rejected", "done_for_day"}


class AlpacaError(RuntimeError):
    pass


class AlpacaBroker(Broker):
    def __init__(self, key=None, secret=None, base_url=PAPER_URL, session=None,
                 dry_run=False, min_order=1.0, poll_seconds=2.0):
        key = key or os.environ.get("ALPACA_API_KEY")
        secret = secret or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise AlpacaError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to your Alpaca PAPER account keys.")
        if "paper-api." not in base_url:
            raise AlpacaError(f"Refusing non-paper Alpaca URL {base_url!r} — live trading is not enabled.")
        self.base = base_url.rstrip("/")
        self.http = session or requests.Session()
        self.http.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.dry_run = dry_run
        self.min_order = min_order
        self.poll_seconds = poll_seconds
        self._pending, self.history = [], []
        self._next_id = 1
        self._account = self._positions = None

    # ── HTTP ──────────────────────────────────────────────────────────
    def _req(self, method, path, **kw):
        r = self.http.request(method, self.base + path, timeout=30, **kw)
        if r.status_code >= 400:
            raise AlpacaError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    # ── Broker interface ──────────────────────────────────────────────
    def account(self):
        if self._account is None:
            self._account = self._req("GET", "/v2/account")
        return self._account

    def cash(self):
        return float(self.account()["cash"])

    def equity(self):
        return float(self.account()["equity"])

    def holdings(self):
        if self._positions is None:
            self._positions = {p["symbol"]: float(p["qty"]) for p in self._req("GET", "/v2/positions")}
        return dict(self._positions)

    def pending(self):
        return list(self._pending)

    def submit(self, when, symbol, side, qty=None, notional=None, tag=""):
        """Queue an order; nothing reaches Alpaca until transmit()."""
        o = Order(self._next_id, when, symbol, side, qty, notional, tag)
        self._next_id += 1
        self._pending.append(o)
        return o

    def apply_dividend(self, symbol, per_share):
        pass                                   # Alpaca credits dividends itself

    def apply_split(self, symbol, ratio):
        pass

    # ── Sending and booking ───────────────────────────────────────────
    def transmit(self):
        """Send every queued order as a market day order: sells, wait for
        their fills, then buys against the refreshed cash balance."""
        unsent = [o for o in self._pending if o.broker_id is None and o.status == "pending"]
        held = self.holdings()
        sells = [o for o in unsent if o.side == "sell"]
        for o in sells:
            self._send(o, held)
        self._wait([o for o in sells if o.broker_id and o.broker_id != "dry-run"])
        self._account = self._positions = None
        for o in unsent:
            if o.side == "buy":
                self._send(o, held)
        for o in [o for o in self._pending if o.status == "rejected"]:
            self._pending.remove(o)
            self.history.append(o)

    def _send(self, o, held):
        body = {"symbol": o.symbol, "side": o.side, "type": "market", "time_in_force": "day",
                "client_order_id": f"pead-{o.when}-{o.id}"}
        if o.side == "sell" and o.qty is None and o.notional is None:
            qty = held.get(o.symbol, 0.0)
            if qty <= 0:
                o.status = "rejected"
                return
            body["qty"] = _decimal(qty)
        elif o.qty is not None:
            body["qty"] = _decimal(o.qty)
        else:
            if o.notional < self.min_order:
                o.status = "rejected"
                return
            body["notional"] = f"{o.notional:.2f}"
        if self.dry_run:
            o.broker_id = "dry-run"
            return
        try:
            o.broker_id = self._req("POST", "/v2/orders", json=body)["id"]
        except AlpacaError as e:
            o.status = "rejected"
            o.tag = f"{o.tag} [{str(e)[:80]}]"

    def _wait(self, orders, timeout=90):
        deadline = time.monotonic() + timeout
        waiting = list(orders)
        while waiting and time.monotonic() < deadline:
            waiting = [o for o in waiting
                       if self._req("GET", f"/v2/orders/{o.broker_id}")["status"] not in DONE_STATES]
            if waiting:
                time.sleep(self.poll_seconds)

    def settle(self, when, price=None):
        """Book the outcome of every sent order for session `when`. Returns
        filled orders; rejected, expired or never-sent ones move to history."""
        filled = []
        for o in [o for o in self._pending if o.when == when]:
            if o.broker_id in (None, "dry-run"):
                o.status = "rejected"
            else:
                info = self._req("GET", f"/v2/orders/{o.broker_id}")
                got = float(info.get("filled_qty") or 0)
                if got > 0 and info["status"] in DONE_STATES:
                    o.status, o.fill_qty = "filled", got
                    o.fill_price = float(info["filled_avg_price"])
                    filled.append(o)
                elif info["status"] in DONE_STATES:
                    o.status = "rejected"
                else:
                    continue                    # still working — look again next run
            self._pending.remove(o)
            self.history.append(o)
        self._account = self._positions = None
        return filled


def _decimal(x):
    return f"{x:.9f}".rstrip("0").rstrip(".")
