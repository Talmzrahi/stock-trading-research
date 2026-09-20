# ═══════════════════════════════════════════════════════════════════════
#  The shadow account: the text signal, traded on paper, on its own books.
#
#  Same engine, same exits, same sizing as the live account — only the
#  signal differs, and the money is a separate simulated $1,000 in
#  data/shadow.db. It never touches data/trader.db, so the live account
#  stays a clean out-of-time test of the earnings-surprise signal while
#  this one accumulates the same kind of evidence for the text signal.
#
#  It exists because the text signal has not passed the gate in CLAUDE.md
#  (development evidence: borderline continuation, p=0.07). Shadow trading
#  is how a rule earns its way in: if it passes the gate later, the only
#  change is which account places the orders.
#
#  Fills use adjusted closes, as the backtest does, so dividends are in
#  the returns without booking each one; the live account uses raw closes
#  and books dividends itself. The two are therefore comparable in total
#  return, not tick for tick.
# ═══════════════════════════════════════════════════════════════════════

import pandas as pd

from . import state as st
from .backtest import cost_model
from .broker import Ledger
from .config import ROOT
from .engine import Engine
from .fusion import Fusion
from .signals.text import TextSignal

SHADOW_DB = ROOT / "data" / "shadow.db"


def shadow_events(base_events, scores):
    """The same events the live account sees, scored by the text signal."""
    fused = Fusion([TextSignal(scores)]).score(base_events)
    return base_events.drop(columns=[c for c in fused.columns if c in base_events.columns],
                            errors="ignore").join(fused)


def run_shadow(cfg, cal, closes, base_events, scores, today, db_path=SHADOW_DB, log=print):
    """Advance the shadow account one day. Returns a summary dict."""
    events = shadow_events(base_events, scores)
    eng = Engine(cfg, cal, closes, events)
    conn = st.open_state(db_path)
    ledger = st.load_ledger(conn, cfg)
    state = st.load_engine_state(conn, cal)
    today_idx = cal.get_loc(today) if today in cal else None

    if st.get(conn, "inception") is None:
        first = today if today_idx is not None else cal[cal.searchsorted(today)]
        st.put(conn, "inception", pd.Timestamp(first).strftime("%Y-%m-%d"))
        st.put(conn, "initial_capital", cfg.initial_capital)
        st.put(conn, "last_settled", cal[cal.searchsorted(first) - 1].strftime("%Y-%m-%d"))
        # Events already too late to enter when the account opened are not news.
        old = events.entry_idx < cal.searchsorted(first) - cfg.max_late_days
        state.processed.update(events.key[old])
        log(f"Shadow account opened with ${cfg.initial_capital:,.0f}, first session "
            f"{pd.Timestamp(first):%Y-%m-%d}.")

    last_settled = pd.Timestamp(st.get(conn, "last_settled"))
    ff, col = eng.ff, eng.col
    price = lambda row: (lambda s: ff[row][col[s]] if s in col else None)

    # Only sessions whose close is actually known can be settled. The daily
    # run's calendar runs ~200 days past today so exits can be scheduled;
    # settling into it would mark the account through dates that have not
    # happened and leave last_settled in the future.
    last_close = closes[cfg.benchmark].last_valid_index()
    horizon = min(pd.Timestamp(today), pd.Timestamp(last_close) + pd.Timedelta(days=1))
    known = cal[(cal > last_settled) & (cal < horizon)]
    marks = []
    for d in known:
        t = cal.get_loc(d)
        fills = ledger.settle(d.strftime("%Y-%m-%d"), price(t))
        eng.on_fills(t, fills, state)
        held = ledger.holdings()
        stocks = sum(q * (ff[t][col[s]] if s in col else 0.0)
                     for s, q in held.items() if s != cfg.benchmark)
        bench = held.get(cfg.benchmark, 0.0) * (ff[t][col[cfg.benchmark]])
        marks.append((d.strftime("%Y-%m-%d"), ledger.cash(), stocks, bench,
                      ledger.cash() + stocks + bench))
        last_settled = d

    decisions = []
    today_s = pd.Timestamp(today).strftime("%Y-%m-%d")
    if today_idx is not None and today_idx > 0 and st.run_status(conn, today_s) != "decided":
        decisions = eng.step(today_idx, state, ledger, price(today_idx - 1))

    st.save_ledger(conn, ledger)
    st.save_engine_state(conn, state, cal, today_s, decisions)
    st.save_marks(conn, marks)
    st.put(conn, "last_settled", last_settled.strftime("%Y-%m-%d"))
    st.record_run(conn, today_s, pd.Timestamp.now().isoformat(timespec="seconds"),
                  "decided" if decisions else "settled")
    conn.commit()

    equity = st.load_marks(conn)
    summary = {"decisions": decisions, "positions": len(state.positions),
               "cash": ledger.cash(), "pending": len(ledger.pending()),
               "equity": float(equity.equity.iloc[-1]) if len(equity) else cfg.initial_capital,
               "initial": float(st.get(conn, "initial_capital", cfg.initial_capital)),
               "trades": len(state.trades), "inception": st.get(conn, "inception")}
    conn.close()
    return summary
