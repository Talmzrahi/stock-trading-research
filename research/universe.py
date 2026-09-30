# ═══════════════════════════════════════════════════════════════════════
#  Point-in-time S&P 500 membership, from Wikipedia revision history.
#
#  Snapshots the constituent list as it stood at each quarter end by
#  fetching the revision of the page live at that date. Coverage starts
#  ~2007, when the page was created.
#
#  WHAT THIS FIXES: look-ahead inclusion. Today's list contains firms that
#  joined recently, often because they had already done well, and their
#  pre-inclusion earnings were being counted in backtests of an S&P 500
#  strategy that could not have held them at the time.
#
#  WHAT THIS DOES NOT FIX: firms removed after failing are still missing,
#  because yfinance does not serve delisted tickers (most return nothing,
#  and some, like SHLD, return a DIFFERENT company that later reused the
#  symbol). That bias needs a paid delisted-securities dataset such as
#  CRSP. Results on the losing tail stay contaminated; say so rather than
#  implying the universe is clean.
#
#  ALWAYS run research/universe_ids.py afterwards. point_in_time matches by
#  FIRM (CIK), not ticker, so a snapshot here without a matching universe_ids
#  row falls back to "SYM:<ticker>" and matches almost nothing. Adding the
#  missing 2012-2014 quarters without it cut recognised 2012 events from
#  ~1,360 to 16 — a repair that looked like a finding until it was traced.
#
#    python research/universe.py
#    python research/universe_ids.py      # not optional
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

DB_FILE = Path(__file__).resolve().parent.parent / "data" / "research.db"
API     = "https://en.wikipedia.org/w/api.php"
TITLE   = "List_of_S&P_500_companies"
START, END = 2007, 2027

SESSION = requests.Session()
# Wikipedia's User-Agent policy asks tools to identify themselves and give a
# means of contact; a generic browser string is what gets throttled first.
SESSION.headers.update({"User-Agent": "PEAD-research/1.0 "
                                      "(https://github.com/Talmzrahi/stock-trading-research)"})
PAUSE   = 3.0          # between quarters; two requests each
RETRIES = 4


def get(url, **params):
    """Wikipedia throttles bursts, and a throttled reply is an HTML error page
    where JSON or a table is expected. Retrying with backoff is the difference
    between a filled quarter and a silent hole — 2012-2014 went missing that way."""
    for attempt in range(RETRIES):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 200 and r.text.lstrip()[:1] in "{<":
                return r
        except requests.RequestException:
            pass
        time.sleep(2.0 * 2 ** attempt)
    return None


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS universe_history (
            as_of  TEXT NOT NULL,
            symbol TEXT NOT NULL,
            PRIMARY KEY (as_of, symbol)
        );
        CREATE INDEX IF NOT EXISTS ix_uh_sym ON universe_history(symbol);
    """)
    conn.commit()


def revision_at(iso_ts):
    resp = get(API, action="query", prop="revisions", titles=TITLE,
               rvlimit=1, rvdir="older", rvstart=iso_ts,
               rvprop="ids|timestamp", format="json")
    if resp is None:
        return None, None
    r = resp.json()
    page = next(iter(r["query"]["pages"].values()))
    revs = page.get("revisions")
    return (revs[0]["revid"], revs[0]["timestamp"]) if revs else (None, None)


def symbols_in_revision(revid):
    resp = get("https://en.wikipedia.org/w/index.php", oldid=revid)
    if resp is None:
        return []
    html = resp.text
    best = None
    for t in pd.read_html(StringIO(html)):
        cols = {str(c).strip().lower(): c for c in t.columns}
        key = next((cols[c] for c in cols if "symbol" in c or "ticker" in c), None)
        # The constituent table is the one with a symbol column and ~500 rows;
        # navboxes and footnote tables also parse, so size is the tiebreak.
        if key is not None and 350 <= len(t) <= 600 and (best is None or len(t) > len(best[0])):
            best = (t, key)
    if best is None:
        return []
    t, key = best
    syms = (t[key].astype(str).str.strip().str.upper()
            .str.replace(".", "-", regex=False)
            .str.replace(r"\[.*\]", "", regex=True))
    # A trailing footnote row parses as NaN, and under pandas' str dtype NaN
    # survives .astype(str) as a float — which is what silently cost 2007-2010.
    return sorted({s for s in syms if isinstance(s, str) and s
                   and 1 <= len(s) <= 6 and s.replace("-", "").isalpha()})


def main():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    init_db(conn)

    quarters = [f"{y}-{m:02d}-01T00:00:00Z"
                for y in range(START, END) for m in (1, 4, 7, 10)]
    quarters = [q for q in quarters if q <= pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")]

    have = {r[0] for r in conn.execute("SELECT DISTINCT as_of FROM universe_history")}
    for q in quarters:
        as_of = q[:10]
        if as_of in have:
            continue
        try:
            revid, ts = revision_at(q)
            if not revid:
                print(f"  {as_of}: no revision yet")
                continue
            syms = symbols_in_revision(revid)
            if len(syms) < 350:
                print(f"  {as_of}: parsed only {len(syms)} symbols, skipping")
                continue
            conn.executemany("INSERT OR IGNORE INTO universe_history (as_of, symbol) VALUES (?,?)",
                             [(as_of, s) for s in syms])
            conn.commit()
            print(f"  {as_of}: {len(syms)} symbols (rev {revid} @ {ts[:10]})")
        except Exception as e:
            print(f"  {as_of}: {type(e).__name__} {str(e)[:80]}")
        time.sleep(PAUSE)

    n_snap = conn.execute("SELECT COUNT(DISTINCT as_of) FROM universe_history").fetchone()[0]
    n_sym = conn.execute("SELECT COUNT(DISTINCT symbol) FROM universe_history").fetchone()[0]
    print(f"\n{n_snap} snapshots, {n_sym} distinct symbols ever seen")

    # A missing quarter is not harmless: point_in_time falls back to the last
    # snapshot before it, so a hole means events matched against a stale index.
    have = {r[0] for r in conn.execute("SELECT DISTINCT as_of FROM universe_history")}
    gaps = [q[:10] for q in quarters if q[:10] not in have]
    print(f"still missing: {len(gaps)} quarters" + (f" — {gaps}" if gaps else ""))

    cur = {r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM universe_history WHERE as_of=(SELECT MAX(as_of) FROM universe_history)")}
    ever = {r[0] for r in conn.execute("SELECT DISTINCT symbol FROM universe_history")}
    print(f"in the index today: {len(cur)}   left at some point: {len(ever - cur)}")
    print("(those departures are the firms free data cannot give us back)")
    conn.close()


if __name__ == "__main__":
    main()
