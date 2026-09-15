# ═══════════════════════════════════════════════════════════════════════
#  Data for the pre-registered out-of-sample test (research/prereg_midsmall.md):
#  S&P MidCap 400 + SmallCap 600, point-in-time, survivorship-aware.
#
#  1. membership  quarterly snapshots of both Wikipedia pages' revision
#                 history → universe_history(as_of, symbol, idx)
#  2. identity    firm = majority CIK per ticker across the 400, 600 and
#                 (read-only) S&P 500 snapshots, else company name, else
#                 SYM:<ticker> → universe_ids
#  3. prices      yfinance adjusted closes for every symbol EVER a member,
#                 plus IJH / IJR / SPY. A ticker no longer in either index is
#                 kept only if its history covers ≥50% of its member sessions
#  4. earnings    EPS estimate/actual for kept symbols; a departed ticker also
#                 needs ≥1 report inside its member span
#
#  Everything goes to data/oos_midsmall.db — the live S&P 500 system never
#  reads it. Resumable at every stage.
#
#    python research/oos_midsmall_data.py
# ═══════════════════════════════════════════════════════════════════════

import re
import sqlite3
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.market_calendar import trading_days  # noqa: E402
from trader.refresh import _ticker_frame  # noqa: E402

DB_FILE      = ROOT / "data" / "oos_midsmall.db"
SP500_DB     = ROOT / "data" / "research.db"
PAGES        = {"400": ("List_of_S&P_400_companies", 2011), "600": ("List_of_S&P_600_companies", 2018)}
ETFS         = ["IJH", "IJR", "SPY"]
PRICE_START  = "2009-01-01"
MIN_COVERAGE = 0.5
API          = "https://en.wikipedia.org/w/api.php"

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0 (index membership research script)"})

SCHEMA = """
    CREATE TABLE IF NOT EXISTS universe_history (
        as_of TEXT NOT NULL, symbol TEXT NOT NULL, idx TEXT NOT NULL,
        PRIMARY KEY (as_of, symbol, idx));
    CREATE TABLE IF NOT EXISTS universe_ids (
        as_of TEXT NOT NULL, symbol TEXT NOT NULL, idx TEXT NOT NULL, name TEXT, cik TEXT, firm TEXT,
        PRIMARY KEY (as_of, symbol, idx));
    CREATE TABLE IF NOT EXISTS snapshot_log (idx TEXT, as_of TEXT, status TEXT, n INTEGER,
        PRIMARY KEY (idx, as_of));
    CREATE TABLE IF NOT EXISTS prices (
        symbol TEXT NOT NULL, date TEXT NOT NULL, close REAL NOT NULL, PRIMARY KEY (symbol, date));
    CREATE TABLE IF NOT EXISTS earnings (
        symbol TEXT NOT NULL, announced_at TEXT NOT NULL,
        eps_estimate REAL, eps_actual REAL, surprise_pct REAL, PRIMARY KEY (symbol, announced_at));
    CREATE TABLE IF NOT EXISTS coverage (
        symbol TEXT PRIMARY KEY, member_from TEXT, member_until TEXT, current INTEGER,
        px_first TEXT, px_last TEXT, coverage REAL, n_earnings INTEGER, status TEXT);
"""


# ── 1. Membership ─────────────────────────────────────────────────────
def api_json(params, tries=6):
    for i in range(tries):
        try:
            r = HTTP.get(API, params=params, timeout=30)
            return r.json()
        except Exception:
            time.sleep(5 * 2 ** i)
    raise RuntimeError("Wikipedia API kept failing")


def page_html(revid, tries=6):
    for i in range(tries):
        try:
            r = HTTP.get("https://en.wikipedia.org/w/index.php", params={"oldid": revid}, timeout=30)
            if r.status_code == 200 and "<table" in r.text:
                return r.text
        except Exception:
            pass
        time.sleep(5 * 2 ** i)
    raise RuntimeError(f"revision {revid} kept failing")


def revision_at(title, ts):
    r = api_json({"action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
                  "rvdir": "older", "rvstart": ts, "rvprop": "ids", "format": "json"})
    revs = next(iter(r["query"]["pages"].values())).get("revisions")
    return revs[0]["revid"] if revs else None


def parse_constituents(html):
    """The first table with a ticker column and ≥300 rows — the constituent
    list comes before the 'changes' table, whose columns are tuples."""
    for t in pd.read_html(StringIO(html)):
        cols = {str(c).strip().lower(): c for c in t.columns}
        if len(t) < 300 or any(k.startswith("(") for k in cols):
            continue
        sym = next((cols[k] for k in cols if "symbol" in k or "ticker" in k), None)
        if sym is None:
            continue
        name = next((cols[k] for k in cols if "security" in k or "company" in k), None)
        cik = cols.get("cik")
        df = pd.DataFrame({
            "symbol": (t[sym].astype(str).str.strip().str.upper()
                       .str.replace(".", "-", regex=False).str.replace(r"\[.*\]", "", regex=True)),
            "name": t[name].astype(str) if name is not None else "",
            "cik": (pd.to_numeric(t[cik], errors="coerce").map(lambda x: f"{int(x):010d}" if pd.notna(x) else None)
                    if cik is not None else None),
        })
        df = df[df.symbol.str.len().between(1, 6) & df.symbol.str.replace("-", "").str.isalpha()]
        return df.drop_duplicates("symbol")
    return None


def snapshots(conn):
    now = pd.Timestamp.now(tz="UTC")
    for idx, (title, first_year) in PAGES.items():
        done = {r[0] for r in conn.execute("SELECT as_of FROM snapshot_log WHERE idx=?", (idx,))}
        quarters = [f"{y}-{m:02d}-01" for y in range(first_year, now.year + 1) for m in (1, 4, 7, 10)
                    if pd.Timestamp(f"{y}-{m:02d}-01", tz="UTC") <= now]
        for as_of in quarters:
            if as_of in done:
                continue
            revid = revision_at(title, f"{as_of}T00:00:00Z")
            df = parse_constituents(page_html(revid)) if revid else None
            if df is None or len(df) < 300:
                status, n = ("no_revision" if not revid else "unparseable"), 0 if df is None else len(df)
            else:
                status, n = "ok", len(df)
                conn.executemany("INSERT OR IGNORE INTO universe_history VALUES (?,?,?)",
                                 [(as_of, s, idx) for s in df.symbol])
                conn.executemany("INSERT OR REPLACE INTO universe_ids (as_of, symbol, idx, name, cik) VALUES (?,?,?,?,?)",
                                 [(as_of, r.symbol, idx, r.name, r.cik) for r in df.itertuples()])
            conn.execute("INSERT OR REPLACE INTO snapshot_log VALUES (?,?,?,?)", (idx, as_of, status, n))
            conn.commit()
            print(f"  S&P {idx} {as_of}: {status} ({n})", flush=True)
            time.sleep(2.0)


# ── 2. Identity ───────────────────────────────────────────────────────
SUFFIXES = (r"\b(the|inc|incorporated|corp|corporation|co|company|companies|ltd|plc|group|"
            r"holding|holdings|nv|sa|lp|llc|class [a-z])\b")


def norm_name(s):
    s = re.sub(r"\[.*?\]", "", str(s).lower()).replace("&", " and ")
    return " ".join(re.sub(SUFFIXES, " ", re.sub(r"[^a-z0-9 ]", " ", s)).split())


def assign_firms(conn):
    ids = pd.read_sql_query("SELECT as_of, symbol, idx, name, cik FROM universe_ids", conn)
    pool = ids[["symbol", "name", "cik"]]
    if SP500_DB.exists():
        big = sqlite3.connect(f"file:{SP500_DB}?mode=ro", uri=True)
        try:
            pool = pd.concat([pool, pd.read_sql_query("SELECT symbol, name, cik FROM universe_ids", big)])
        except Exception:
            pass
        big.close()
    known = pool[pool.cik.notna()]
    by_symbol = known.groupby("symbol").cik.agg(lambda s: s.value_counts().index[0]).to_dict()
    good = known[known.cik == known.symbol.map(by_symbol)]
    by_name = good.assign(n=good.name.map(norm_name)).drop_duplicates("n").set_index("n").cik.to_dict()
    ids["firm"] = [by_symbol.get(r.symbol) or by_name.get(norm_name(r.name)) or f"SYM:{r.symbol}"
                   for r in ids.itertuples()]
    conn.executemany("UPDATE universe_ids SET firm=? WHERE as_of=? AND symbol=? AND idx=?",
                     zip(ids.firm, ids.as_of, ids.symbol, ids.idx))
    conn.commit()
    print(f"  {ids.symbol.nunique()} tickers → {ids.firm.nunique()} firms; "
          f"{ids.firm.str.startswith('SYM:').mean():.1%} of rows without a CIK identity")


# ── 3/4. Prices and earnings ──────────────────────────────────────────
def member_spans(conn):
    u = pd.read_sql_query("SELECT DISTINCT as_of, symbol FROM universe_history", conn)
    snaps = sorted(u.as_of.unique())
    nxt = {s: snaps[i + 1] if i + 1 < len(snaps) else pd.Timestamp.now().strftime("%Y-%m-%d")
           for i, s in enumerate(snaps)}
    latest = {r[0] for r in conn.execute(
        """SELECT symbol FROM universe_history h WHERE as_of =
           (SELECT MAX(as_of) FROM universe_history WHERE idx = h.idx)""")}
    g = u.groupby("symbol").as_of.agg(["min", "max"])
    g["until"] = g["max"].map(nxt)
    g["current"] = g.index.isin(latest)
    return g


def fetch_prices(conn, spans):
    done = {r[0] for r in conn.execute("SELECT symbol FROM coverage")}
    todo = [s for s in list(spans.index) if s not in done]
    for etf in ETFS:
        if not conn.execute("SELECT 1 FROM prices WHERE symbol=? LIMIT 1", (etf,)).fetchone():
            todo.append(etf)
    print(f"  prices: {len(todo)} symbols to fetch")
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        try:
            data = yf.download(batch, start=PRICE_START, auto_adjust=True, progress=False,
                               threads=True, group_by="ticker")
        except Exception as e:
            print(f"    batch failed: {type(e).__name__}")
            continue
        for sym in batch:
            df = _ticker_frame(data, sym, len(batch) == 1)
            closes = None if df is None or "Close" not in df.columns else df["Close"].dropna()
            if closes is not None and len(closes):
                idx = closes.index.tz_localize(None) if closes.index.tz is not None else closes.index
                closes = closes.set_axis(idx.normalize())
            if sym in ETFS:
                if closes is not None and len(closes):
                    conn.executemany("INSERT OR IGNORE INTO prices VALUES (?,?,?)",
                                     [(sym, d.strftime("%Y-%m-%d"), float(c)) for d, c in closes.items()])
                continue
            span = spans.loc[sym]
            if closes is None or closes.empty:
                status, cov = "no_prices", 0.0
            else:
                sessions = trading_days(span["min"], span["until"])
                cov = float(closes.index.isin(sessions).sum() / max(len(sessions), 1))
                status = "prices_ok" if (span["current"] or cov >= MIN_COVERAGE) else "reused_or_partial"
            if status == "prices_ok":
                conn.executemany("INSERT OR IGNORE INTO prices VALUES (?,?,?)",
                                 [(sym, d.strftime("%Y-%m-%d"), float(c)) for d, c in closes.items()])
            conn.execute("INSERT OR REPLACE INTO coverage VALUES (?,?,?,?,?,?,?,?,?)", (
                sym, span["min"], span["until"], int(span["current"]),
                None if closes is None or closes.empty else str(closes.index.min().date()),
                None if closes is None or closes.empty else str(closes.index.max().date()),
                cov, None, status))
        conn.commit()
        print(f"    {min(i + 100, len(todo))}/{len(todo)}", flush=True)
        time.sleep(1.0)


def fetch_earnings(conn):
    todo = conn.execute("SELECT symbol, member_from, member_until, current FROM coverage "
                        "WHERE status='prices_ok'").fetchall()
    print(f"  earnings: {len(todo)} symbols to fetch")
    for n, (sym, lo, hi, current) in enumerate(todo, 1):
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=100)
        except Exception:
            ed = None
        rows = []
        if ed is not None and len(ed):
            for ts, r in ed.iterrows():
                est, act = r.get("EPS Estimate"), r.get("Reported EPS")
                rows.append((sym, ts.isoformat(), None if pd.isna(est) else float(est),
                             None if pd.isna(act) else float(act),
                             None if pd.isna(r.get("Surprise(%)")) else float(r["Surprise(%)"])))
        a, b = pd.Timestamp(lo) - pd.Timedelta(days=120), pd.Timestamp(hi)
        n_in = sum(1 for _, ts, est, act, _ in rows if est is not None and act is not None
                   and a <= pd.Timestamp(ts).tz_convert(None) <= b)
        if n_in or current:
            conn.executemany("INSERT OR IGNORE INTO earnings VALUES (?,?,?,?,?)", rows)
            status = "kept" if n_in else "current_no_earnings"
        else:
            conn.execute("DELETE FROM prices WHERE symbol=?", (sym,))
            status = "no_earnings"
        conn.execute("UPDATE coverage SET n_earnings=?, status=? WHERE symbol=?", (n_in, status, sym))
        conn.commit()
        if n % 100 == 0:
            print(f"    {n}/{len(todo)}", flush=True)
        time.sleep(0.4)


def main():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=60)
    conn.executescript(SCHEMA)
    print("1. Membership snapshots")
    snapshots(conn)
    print("2. Firm identity")
    assign_firms(conn)
    spans = member_spans(conn)
    print(f"3. Prices ({len(spans)} symbols ever a member)")
    fetch_prices(conn, spans)
    print("4. Earnings")
    fetch_earnings(conn)
    print("\nCoverage:", dict(conn.execute("SELECT status, COUNT(*) FROM coverage GROUP BY status").fetchall()))
    print("Snapshots:", dict(conn.execute("SELECT idx || ' ' || status, COUNT(*) FROM snapshot_log GROUP BY idx, status").fetchall()))
    conn.close()


if __name__ == "__main__":
    main()
