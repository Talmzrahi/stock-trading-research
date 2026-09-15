# ═══════════════════════════════════════════════════════════════════════
#  The daily paper-trading run.
#
#    python -m trader.run_daily              normal run (Task Scheduler)
#    python -m trader.run_daily --dry-run    decide and report, save nothing
#
#  1. refresh data (universe weekly, prices, due earnings)
#  2. settle every order whose closing price is now known; credit
#     dividends and apply splits on the way
#  3. if today is a session and it's before the 15:50 ET market-on-close
#     cutoff: decide and submit orders for today's close
#  4. write reports/YYYY-MM-DD.md
#
#  Safe to run more than once a day: a second run settles and reports but
#  never submits a second set of orders. A missed day is caught up next
#  run — entries up to max_late_days late, due exits a day late.
# ═══════════════════════════════════════════════════════════════════════

import argparse
import json
import math
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from . import refresh
from . import state as st
from .backtest import market_from
from .config import ROOT, STRATEGY_FILE, load_config
from .data import RESEARCH_DB, TRADER_DB, connect, load_closes, load_earnings, load_universe
from .engine import Engine
from .market_calendar import ET, is_trading_day, moc_cutoff, next_trading_day, now_et, trading_days
from .monitor import tripwire_status
from .report import money, render

REPORTS   = ROOT / "reports"
BAND_FILE = ROOT / "config" / "tripwire_band.csv"


class Prices:
    """Raw closes for the broker (fills, marks); adjusted closes as fallback
    when a raw series is missing."""

    def __init__(self, eng, raw):
        self.eng = eng
        self.raw = raw.to_numpy() if len(raw.columns) else np.empty((len(eng.cal), 0))
        self.col = {s: i for i, s in enumerate(raw.columns)}

    def at(self, symbol, i):
        if symbol in self.col:
            v = self.raw[i, self.col[symbol]]
            if not math.isnan(v):
                return float(v)
        if symbol in self.eng.col:
            v = self.eng.ff[i, self.eng.col[symbol]]
            if not math.isnan(v):
                return float(v)
        return None


def parse_args(argv):
    p = argparse.ArgumentParser(description="Daily PEAD paper-trading run")
    p.add_argument("--dry-run", action="store_true", help="decide and report, but save nothing")
    p.add_argument("--skip-refresh", action="store_true", help="use data already in research.db")
    p.add_argument("--full-refresh", action="store_true",
                   help="re-download full price history and every member's earnings")
    p.add_argument("--now", help="override the clock (ET), 'YYYY-MM-DD HH:MM' — for testing")
    return p.parse_args(argv)


def refresh_data(rconn, cfg, now, held, full, notes):
    warnings = []
    try:
        ch = refresh.refresh_universe(rconn, now)
        if ch and (ch["added"] or ch["removed"]):
            notes.append(f"S&P 500 changes — added {ch['added'] or 'none'}, removed {ch['removed'] or 'none'}")
    except Exception as e:
        warnings.append(f"Universe refresh failed ({type(e).__name__}: {e}); using last snapshot.")
    members = refresh.latest_members(rconn)

    try:
        full_px = full or refresh.due(rconn, "prices_full", 30, now)
        symbols = sorted(set(members) | set(held) | {cfg.benchmark})
        print(f"Refreshing prices for {len(symbols)} symbols{' (full history)' if full_px else ''} …", flush=True)
        _, missing = refresh.refresh_prices(rconn, symbols, now, full=full_px)
        if missing:
            warnings.append(f"No price data for {len(missing)} symbol(s): {', '.join(missing[:12])}"
                            + (" …" if len(missing) > 12 else ""))
    except Exception as e:
        warnings.append(f"Price refresh failed ({type(e).__name__}: {e}).")

    try:
        full_eps = full or refresh.due(rconn, "earnings_full", 7, now)
        targets = members if full_eps else refresh.earnings_targets(rconn, members, now.date())
        print(f"Refreshing earnings for {len(targets)} symbols …", flush=True)
        ok, failed = refresh.refresh_earnings(rconn, targets)
        if full_eps:
            refresh.mark_done(rconn, "earnings_full", now)
        notes.append(f"Earnings refreshed for {ok} of {len(targets)} symbol(s)"
                     + (" — weekly full pass" if full_eps else ""))
        if failed:
            warnings.append(f"Earnings lookup failed for {len(failed)}: {', '.join(failed[:12])}")
    except Exception as e:
        warnings.append(f"Earnings refresh failed ({type(e).__name__}: {e}).")
    return members, warnings


def run(args):
    cfg = load_config()
    now = (datetime.strptime(args.now, "%Y-%m-%d %H:%M").replace(tzinfo=ET) if args.now else now_et())
    today = pd.Timestamp(now.date())
    today_s = today.strftime("%Y-%m-%d")
    warnings, notes = [], []
    bench = cfg.benchmark

    rconn = connect(RESEARCH_DB)
    refresh.init_tables(rconn)
    sconn = st.open_state(TRADER_DB)
    ledger = st.load_ledger(sconn, cfg)

    if args.skip_refresh:
        members = refresh.latest_members(rconn)
        notes.append("Data refresh skipped (--skip-refresh).")
    else:
        members, w = refresh_data(rconn, cfg, now, sorted(ledger.holdings()), args.full_refresh, notes)
        warnings += w

    print("Building signals …", flush=True)
    closes = load_closes(rconn)
    cal = trading_days(closes.index.min(), today + pd.Timedelta(days=200))
    market = market_from(cfg, cal, closes.reindex(cal), load_earnings(rconn), load_universe(rconn))
    eng = Engine(cfg, market.cal, market.closes, market.events)
    prices = Prices(eng, refresh.load_raw_closes(rconn).reindex(cal).ffill())
    state = st.load_engine_state(sconn, cal)

    # ── Account bookkeeping ───────────────────────────────────────────
    if st.get(sconn, "inception") is None:
        first = today if is_trading_day(today) else next_trading_day(today)
        st.put(sconn, "inception", first.strftime("%Y-%m-%d"))
        st.put(sconn, "last_settled", cal[cal.searchsorted(first) - 1].strftime("%Y-%m-%d"))
        st.put(sconn, "initial_capital", cfg.initial_capital)
        # Reports that were already too late before the account existed aren't news.
        old = market.events.entry_idx < cal.get_loc(first) - cfg.max_late_days
        state.processed.update(market.events.key[old])
        notes.append(f"Opened the simulated account with {money(cfg.initial_capital)}, first session {first:%Y-%m-%d}.")
    inception = pd.Timestamp(st.get(sconn, "inception"))
    initial = float(st.get(sconn, "initial_capital"))
    last_settled = pd.Timestamp(st.get(sconn, "last_settled"))
    last_close = market.closes[bench].last_valid_index()

    # ── Settle everything whose close is known ────────────────────────
    actions = refresh.load_actions(rconn)
    actions = actions[actions.date >= inception.strftime("%Y-%m-%d")]
    applied = st.applied_actions(sconn)
    new_applied, fills, marks = [], [], []

    def apply_action(a):
        if a.kind == "dividend":
            ledger.apply_dividend(a.symbol, a.value)
            notes.append(f"Dividend: {a.symbol} {money(a.value)}/share (ex {a.date})")
        elif a.kind == "split":
            ledger.apply_split(a.symbol, a.value)
            notes.append(f"Split: {a.symbol} x{a.value:g} (effective {a.date})")
        new_applied.append((a.symbol, a.date, a.kind))

    # Actions that arrived after their session was already settled.
    held = ledger.holdings()
    for a in actions[actions.date <= last_settled.strftime("%Y-%m-%d")].itertuples():
        if (a.symbol, a.date, a.kind) in applied or a.symbol not in held:
            continue
        pos = state.positions.get(a.symbol)
        if a.symbol == bench or (pos and cal[pos.entry_idx] < pd.Timestamp(a.date)):
            apply_action(a)

    sessions = cal[(cal > last_settled) & (cal <= last_close)] if last_close is not None else []
    for d in sessions:
        i, ds = cal.get_loc(d), d.strftime("%Y-%m-%d")
        held = ledger.holdings()
        for a in actions[actions.date == ds].itertuples():
            if (a.symbol, a.date, a.kind) not in applied and a.symbol in held:
                apply_action(a)
        day = ledger.settle(ds, lambda s, i=i: prices.at(s, i))
        eng.on_fills(i, day, state)
        fills += day
        h = ledger.holdings()
        stocks = sum(q * (prices.at(s, i) or 0) for s, q in h.items() if s != bench)
        bv = h.get(bench, 0.0) * (prices.at(bench, i) or 0)
        marks.append((ds, ledger.cash(), stocks, bv, ledger.cash() + stocks + bv))
    if len(sessions):
        last_settled = sessions[-1]
    rejected = [o for o in fills if o.status == "rejected"]
    fills += [o for o in ledger.history if o.status == "rejected" and o not in fills]
    if rejected:
        warnings.append(f"{len(rejected)} order(s) rejected at settlement: "
                        + ", ".join(f"{o.side} {o.symbol}" for o in rejected))

    # ── Decide today's orders ─────────────────────────────────────────
    decisions, submitted, run_status = [], [], "settled"
    t = cal.get_loc(today) if is_trading_day(today) else None
    unsettled = [o for o in ledger.pending() if pd.Timestamp(o.when) < today]
    if t is None:
        status = "Market closed today — no orders."
    elif st.run_status(sconn, today_s) == "decided":
        status = "Today's orders were already decided by an earlier run — this run only settled and reported."
    elif now > moc_cutoff(today):
        status = (f"Ran after the {moc_cutoff(today):%H:%M} ET market-on-close cutoff — no orders today. "
                  f"The next run takes entries up to {cfg.max_late_days} sessions late and sends any due exits.")
    elif unsettled:
        status = "Earlier orders are still unsettled (their closing prices aren't in the data yet) — not trading on a stale book."
        warnings.append(status)
    elif pd.isna(market.closes[bench].iloc[t - 1]):
        status = f"No {bench} close for {cal[t - 1]:%Y-%m-%d} — price data is stale, no orders."
        warnings.append(status)
    else:
        n0 = len(ledger.pending())
        decisions = eng.step(t, state, ledger, lambda s: prices.at(s, t - 1))
        submitted = ledger.pending()[n0:]
        run_status = "decided"
        status = (f"{len(submitted)} market-on-close order(s) submitted for the {today:%Y-%m-%d} close."
                  if submitted else "No trades needed today.")

    missing_px = [s for s in ledger.holdings() if prices.at(s, cal.get_loc(last_close)) is None]
    if missing_px:
        warnings.append(f"Held symbols with no price: {', '.join(missing_px)} — equity understated.")

    # ── Persist ───────────────────────────────────────────────────────
    new_trades = list(state.trades)
    if args.dry_run:
        sconn.rollback()
    else:
        st.save_ledger(sconn, ledger)
        st.save_engine_state(sconn, state, cal, today_s, decisions)
        st.put(sconn, "last_settled", last_settled.strftime("%Y-%m-%d"))
        st.save_marks(sconn, marks)
        st.save_applied(sconn, new_applied)
        st.record_run(sconn, today_s, now.isoformat(), run_status)
        sconn.commit()

    # ── Report ────────────────────────────────────────────────────────
    li = cal.get_loc(last_close)
    h = ledger.holdings()
    stocks = sum(q * (prices.at(s, li) or 0) for s, q in h.items() if s != bench)
    bv = h.get(bench, 0.0) * (prices.at(bench, li) or 0)
    equity = ledger.cash() + stocks + bv
    base = cal.searchsorted(inception) - 1
    spy = market.closes[bench]
    spy_since = spy.iloc[li] / spy.iloc[base] - 1 if li > base else float("nan")

    positions = []
    stop_rule = next((r for r in eng.exits if r.name == "stop"), None)
    bc = eng.col[bench]
    for sym, p in sorted(state.positions.items()):
        c = eng.col[sym]
        ret = eng.ff[li, c] / eng.ff[p.entry_idx, c] - 1
        bret = eng.ff[li, bc] / eng.ff[p.entry_idx, bc] - 1
        gap = float("nan")
        if stop_rule is not None and li > p.entry_idx:
            level = np.nanmax(eng.px[p.entry_idx:li + 1, c]) * (1 - stop_rule.k * stop_rule.vol[li, c])
            gap = eng.ff[li, c] / level - 1
        positions.append(dict(
            symbol=sym, entry=f"{cal[p.entry_idx]:%Y-%m-%d}", held=max(li - p.entry_idx, 0),
            due=f"{cal[p.exit_due_idx]:%Y-%m-%d}" if p.exit_due_idx < len(cal) else "—",
            value=h.get(sym, 0.0) * (prices.at(sym, li) or 0), ret=ret, alpha=ret - bret, stop_gap=gap))

    trades = st.load_trades(sconn)
    if args.dry_run and new_trades:
        trades = pd.concat([trades, pd.DataFrame([vars(x) for x in new_trades])], ignore_index=True)
    band = pd.read_csv(BAND_FILE) if BAND_FILE.exists() else None
    trip = tripwire_status(trades.alpha.to_numpy(), band) if band is not None else {"state": "no_band"}
    expected = float("nan")
    if STRATEGY_FILE.exists():
        expected = json.loads(STRATEGY_FILE.read_text(encoding="utf-8"))["evidence"]["chosen_summary"]["mean_alpha"]

    ctx = dict(
        dry_run=args.dry_run, now=now, today=today_s, status=status, warnings=warnings, notes=notes,
        cfg=cfg, decisions=decisions, submitted=submitted, fills=fills, positions=positions,
        trades=trades, tripwire=trip, expected_alpha=expected,
        upcoming=refresh.upcoming_reports(rconn, members, today.date()),
        account=dict(equity=equity, cash=ledger.cash(), stocks=stocks, bench=bv, initial=initial,
                     inception=f"{inception:%Y-%m-%d}", ret_since=equity / initial - 1,
                     spy_since=spy_since, exposure=stocks / equity if equity else float("nan"),
                     prices_as_of=f"{last_close:%Y-%m-%d} close"))
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{today_s}{'-dryrun' if args.dry_run else ''}.md"
    path.write_text(render(ctx), encoding="utf-8")

    rconn.close()
    sconn.close()
    print(status)
    for w in warnings:
        print(f"WARNING: {w}")
    if trip.get("state") == "FIRED":
        print("TRIPWIRE FIRED — see the report.")
    print(f"Report: {path}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    try:
        run(args)
    except Exception:
        REPORTS.mkdir(exist_ok=True)
        path = REPORTS / f"{datetime.now(ET):%Y-%m-%d}-ERROR.md"
        path.write_text("# Daily run crashed\n\nNo orders were saved.\n\n```\n"
                        + traceback.format_exc() + "```\n", encoding="utf-8")
        traceback.print_exc()
        print(f"Crash report: {path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
