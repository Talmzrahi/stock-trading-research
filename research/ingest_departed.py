# ═══════════════════════════════════════════════════════════════════════
#  Survivorship repair: fetch data for firms that LEFT the S&P 500.
#
#  research/ingest.py pulled prices and earnings for TODAY's constituents
#  only. universe_history has 880 symbols ever in the index; the 377 that
#  have since left had no data at all, so every point-in-time backtest saw
#  only survivors — 52% of the index's member-quarters in 2010, 78% in
#  2020. Firms leave the index disproportionately after doing badly, so
#  this flatters every result so far.
#
#  yfinance still serves many departed firms (demoted, still listed) but
#  not acquired or delisted ones, and some symbols now belong to a
#  DIFFERENT company. A symbol is kept only if its price history covers at
#  least half the sessions it was actually an index member AND it has an
#  earnings report inside that period. A reused ticker whose new owner
#  also has history over the old span would slip through — rare, but
#  possible; renamed firms (ANTM → ELV) stay missing.
#
#  Writes to prices / earnings like ingest.py; verdicts per symbol go to
#  departed_coverage. Resumable.
#
#    python research/ingest_departed.py           check every unchecked symbol
#    python research/ingest_departed.py --retry   re-try 'no_prices' one at a time
#                                                 (batch downloads get throttled)
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.market_calendar import trading_days  # noqa: E402
from trader.refresh import _ticker_frame  # noqa: E402

DB_FILE      = ROOT / "data" / "research.db"
START        = "2002-01-01"
BATCH        = 50
MIN_COVERAGE = 0.5
PAUSE        = 0.4
RETRY_PAUSE  = 1.5


def init(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS departed_coverage (
        symbol TEXT PRIMARY KEY, member_from TEXT, member_until TEXT,
        px_first TEXT, px_last TEXT, coverage REAL, n_earnings INTEGER, status TEXT)""")
    conn.commit()


def departed_spans(conn):
    u = pd.read_sql_query("SELECT as_of, symbol FROM universe_history", conn)
    snaps = sorted(u.as_of.unique())
    nxt = {s: snaps[i + 1] if i + 1 < len(snaps) else s for i, s in enumerate(snaps)}
    current = set(u.symbol[u.as_of == snaps[-1]])
    g = u.groupby("symbol").as_of.agg(["min", "max"])
    g = g[~g.index.isin(current)].copy()
    g["until"] = g["max"].map(nxt)        # still a member until the next snapshot at most
    return g


def record(conn, sym, span, closes, coverage, n_earn, status):
    conn.execute("INSERT OR REPLACE INTO departed_coverage VALUES (?,?,?,?,?,?,?,?)", (
        sym, span["min"], span["until"],
        None if closes is None or closes.empty else str(closes.index.min().date()),
        None if closes is None or closes.empty else str(closes.index.max().date()),
        coverage, n_earn, status))


def evaluate(conn, sym, span, closes):
    """Coverage check, earnings fetch, and write-or-reject for one symbol."""
    if closes is None or closes.empty:
        record(conn, sym, span, None, 0.0, 0, "no_prices")
        return "no_prices"
    idx = closes.index.tz_localize(None) if closes.index.tz is not None else closes.index
    closes = closes.set_axis(idx.normalize())
    sessions = trading_days(span["min"], span["until"])
    coverage = float(closes.index.isin(sessions).sum() / max(len(sessions), 1))
    if coverage < MIN_COVERAGE:
        record(conn, sym, span, closes, coverage, 0, "reused_or_partial")
        return "reused_or_partial"

    try:
        ed = yf.Ticker(sym).get_earnings_dates(limit=100)
    except Exception:
        ed = None
    time.sleep(PAUSE)
    rows = []
    if ed is not None and len(ed):
        for ts, r in ed.iterrows():
            est, act = r.get("EPS Estimate"), r.get("Reported EPS")
            rows.append((sym, ts.isoformat(),
                         None if pd.isna(est) else float(est),
                         None if pd.isna(act) else float(act),
                         None if pd.isna(r.get("Surprise(%)")) else float(r["Surprise(%)"])))
    lo = pd.Timestamp(span["min"]) - pd.Timedelta(days=120)
    hi = pd.Timestamp(span["until"])
    n_in = sum(1 for _, ts, est, act, _ in rows
               if est is not None and act is not None
               and lo <= pd.Timestamp(ts).tz_convert(None) <= hi)
    if n_in == 0:
        record(conn, sym, span, closes, coverage, 0, "no_earnings")
        return "no_earnings"

    conn.executemany("INSERT OR IGNORE INTO prices (symbol, date, close) VALUES (?,?,?)",
                     [(sym, d.strftime("%Y-%m-%d"), float(c)) for d, c in closes.items()])
    conn.executemany("""INSERT OR IGNORE INTO earnings
                        (symbol, announced_at, eps_estimate, eps_actual, surprise_pct)
                        VALUES (?,?,?,?,?)""", rows)
    record(conn, sym, span, closes, coverage, n_in, "kept")
    return "kept"


def check_batches(conn, spans, todo):
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            data = yf.download(batch, start=START, auto_adjust=True, progress=False,
                               threads=True, group_by="ticker")
        except Exception as e:
            print(f"  batch {i // BATCH}: download failed — {type(e).__name__}: {str(e)[:100]}")
            continue
        for sym in batch:
            df = _ticker_frame(data, sym, len(batch) == 1)
            closes = None if df is None or "Close" not in df.columns else df["Close"].dropna()
            evaluate(conn, sym, spans.loc[sym], closes)
        conn.commit()
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)} checked", flush=True)


def retry_one_by_one(conn, spans, todo):
    for n, sym in enumerate(todo, 1):
        try:
            h = yf.Ticker(sym).history(start=START, auto_adjust=True)
            closes = h["Close"].dropna() if len(h) else None
        except Exception:
            closes = None
        verdict = evaluate(conn, sym, spans.loc[sym], closes)
        conn.commit()
        if verdict != "no_prices":
            print(f"  {sym}: {verdict}", flush=True)
        if n % 25 == 0:
            print(f"  {n}/{len(todo)} retried", flush=True)
        time.sleep(RETRY_PAUSE)


def main():
    retry = "--retry" in sys.argv
    conn = sqlite3.connect(DB_FILE, timeout=60)
    init(conn)
    spans = departed_spans(conn)
    if retry:
        # Batch downloads under throttling come back empty or truncated —
        # several unrelated tickers "starting" on the same recent day is the
        # signature — so recent-start rejections get a second look too.
        todo = [r[0] for r in conn.execute(
            """SELECT symbol FROM departed_coverage
               WHERE status='no_prices'
                  OR (status='reused_or_partial' AND px_first >= date('now', '-18 months'))""")
                if r[0] in spans.index]
        print(f"Retrying {len(todo)} empty or truncated symbols one at a time")
        retry_one_by_one(conn, spans, todo)
    else:
        done = {r[0] for r in conn.execute("SELECT symbol FROM departed_coverage")}
        todo = [s for s in spans.index if s not in done]
        print(f"{len(spans)} departed symbols, {len(todo)} still to check")
        check_batches(conn, spans, todo)

    print("\nVerdicts:", dict(conn.execute(
        "SELECT status, COUNT(*) FROM departed_coverage GROUP BY status").fetchall()))
    u = pd.read_sql_query("SELECT as_of, symbol FROM universe_history", conn)
    have = {r[0] for r in conn.execute("SELECT DISTINCT symbol FROM earnings")}
    u["has"] = u.symbol.isin(have)
    print("Share of index member-quarters with earnings data, by year:")
    print("  " + "  ".join(f"{y}: {v:.0%}" for y, v in u.groupby(u.as_of.str[:4]).has.mean().items()))
    conn.close()


if __name__ == "__main__":
    main()
