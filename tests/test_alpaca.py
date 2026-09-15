import unittest

from trader.alpaca import AlpacaBroker, AlpacaError


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.content = b"x"
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeAlpaca:
    """Just enough of the Alpaca v2 API: market orders fill instantly at a
    fixed price per symbol."""

    def __init__(self, cash=1000.0, positions=None, prices=None):
        self.headers = {}
        self.cash = cash
        self.positions = dict(positions or {})
        self.prices = prices or {}
        self.orders, self.sent = {}, []

    def request(self, method, url, timeout=None, json=None, params=None):
        path = url.split(".markets", 1)[1]
        if method == "GET" and path == "/v2/account":
            return FakeResponse(200, {"cash": str(self.cash), "equity": str(self.cash)})
        if method == "GET" and path == "/v2/positions":
            return FakeResponse(200, [{"symbol": s, "qty": str(q)} for s, q in self.positions.items()])
        if method == "POST" and path == "/v2/orders":
            self.sent.append(json)
            px = self.prices[json["symbol"]]
            qty = float(json["qty"]) if "qty" in json else float(json["notional"]) / px
            sign = 1 if json["side"] == "buy" else -1
            self.cash -= sign * qty * px
            self.positions[json["symbol"]] = self.positions.get(json["symbol"], 0.0) + sign * qty
            oid = f"o{len(self.orders) + 1}"
            self.orders[oid] = {"id": oid, "status": "filled", "filled_qty": str(qty),
                                "filled_avg_price": str(px)}
            return FakeResponse(200, self.orders[oid])
        if method == "GET" and path.startswith("/v2/orders/"):
            return FakeResponse(200, self.orders[path.rsplit("/", 1)[1]])
        return FakeResponse(404, {"message": "not found"})


class AlpacaBrokerTest(unittest.TestCase):
    def broker(self, fake, **kw):
        return AlpacaBroker("k", "s", session=fake, poll_seconds=0, **kw)

    def test_refuses_live_url_and_missing_keys(self):
        with self.assertRaises(AlpacaError):
            AlpacaBroker("k", "s", base_url="https://api.alpaca.markets", session=FakeAlpaca())
        with self.assertRaises(AlpacaError):
            AlpacaBroker(None, None, session=FakeAlpaca())

    def test_queue_transmit_settle(self):
        fake = FakeAlpaca(cash=100.0, positions={"SPY": 1.0}, prices={"SPY": 700.0, "AAA": 50.0})
        b = self.broker(fake)
        self.assertEqual(b.holdings(), {"SPY": 1.0})
        b.submit("2026-09-16", "AAA", "buy", notional=300.0, tag="entry:AAA|2026-09-16")
        b.submit("2026-09-16", "SPY", "sell", tag="exit:time")
        self.assertEqual(fake.sent, [])                          # queued, not sent
        b.transmit()
        self.assertEqual([o["side"] for o in fake.sent], ["sell", "buy"])
        sell, buy = fake.sent
        self.assertEqual((sell["qty"], sell["time_in_force"], sell["type"]), ("1", "day", "market"))
        self.assertEqual(buy["notional"], "300.00")
        self.assertTrue(buy["client_order_id"].startswith("pead-2026-09-16-"))
        fills = b.settle("2026-09-16")
        self.assertEqual({o.symbol: o.fill_price for o in fills}, {"SPY": 700.0, "AAA": 50.0})
        self.assertAlmostEqual(next(o for o in fills if o.symbol == "AAA").fill_qty, 6.0)
        self.assertEqual(b.pending(), [])

    def test_dry_run_sends_nothing_and_unsent_orders_reject(self):
        fake = FakeAlpaca(prices={"AAA": 50.0})
        b = self.broker(fake, dry_run=True)
        b.submit("2026-09-16", "AAA", "buy", notional=100.0)
        b.transmit()
        self.assertEqual(fake.sent, [])
        self.assertEqual(b.settle("2026-09-16"), [])
        self.assertEqual(b.history[0].status, "rejected")

    def test_selling_a_position_not_held_is_rejected_locally(self):
        fake = FakeAlpaca(prices={"AAA": 50.0})
        b = self.broker(fake)
        b.submit("2026-09-16", "AAA", "sell", tag="exit:stop")
        b.transmit()
        self.assertEqual(fake.sent, [])
        self.assertEqual(b.history[0].status, "rejected")


if __name__ == "__main__":
    unittest.main()
