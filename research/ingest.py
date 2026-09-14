# ═══════════════════════════════════════════════════════════════════════
#  Phase 1 data ingestion — pulls the inputs needed to test the core
#  hypothesis (VIX-elevated regime + earnings beat -> excess drift).
#
#  Pulls three things into data/research.db:
#    1. Daily closes for the test universe + SPY benchmark
#    2. Daily VIX closes
#    3. Quarterly EPS estimate vs. actual per ticker (~20y via yfinance)
#
#  Resumable: re-running skips tickers already fetched. Safe to interrupt.
#    python research/ingest.py
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")

DB_FILE     = Path(__file__).resolve().parent.parent / "data" / "research.db"
START_DATE  = "2002-01-01"
BENCHMARK   = "SPY"
VIX_SYMBOL  = "^VIX"
PRICE_BATCH = 40          # tickers per bulk download call
EARNINGS_PAUSE = 0.4      # seconds between per-ticker earnings calls


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            date   TEXT NOT NULL,
            close  REAL NOT NULL,
            PRIMARY KEY (symbol, date)
        );
        CREATE TABLE IF NOT EXISTS vix (
            date  TEXT PRIMARY KEY,
            close REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS earnings (
            symbol        TEXT NOT NULL,
            announced_at  TEXT NOT NULL,   -- ISO8601 with tz, exact announcement timestamp
            eps_estimate  REAL,
            eps_actual    REAL,
            surprise_pct  REAL,
            PRIMARY KEY (symbol, announced_at)
        );
        CREATE TABLE IF NOT EXISTS ingest_progress (
            symbol     TEXT NOT NULL,
            kind       TEXT NOT NULL,      -- 'prices' | 'earnings'
            status     TEXT NOT NULL,      -- 'done' | 'empty' | 'error'
            updated_at TEXT,
            PRIMARY KEY (symbol, kind)
        );
    """)
    conn.commit()


def done_symbols(conn, kind):
    rows = conn.execute(
        "SELECT symbol FROM ingest_progress WHERE kind=? AND status IN ('done','empty')",
        (kind,),
    ).fetchall()
    return {r[0] for r in rows}


def mark(conn, symbol, kind, status):
    conn.execute(
        "INSERT OR REPLACE INTO ingest_progress (symbol, kind, status, updated_at) VALUES (?,?,?,?)",
        (symbol, kind, status, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def sp500_universe():
    """Current S&P 500 constituents from Wikipedia.

    NOTE: this is the *surviving* membership as of today, which introduces
    survivorship bias. The backtest is designed around a within-universe
    comparison (high-VIX beats vs. normal-VIX beats) so the bias largely
    cancels rather than inflating the headline result.
    """
    # Wikipedia rejects urllib's default user-agent with a 403, so fetch via requests.
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=30).text
    tables = pd.read_html(StringIO(html))
    syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return sorted(set(syms))


def save_prices(conn, df, symbol):
    rows = [(symbol, d.strftime("%Y-%m-%d"), float(c))
            for d, c in df.items() if pd.notna(c)]
    if rows:
        conn.executemany("INSERT OR IGNORE INTO prices (symbol, date, close) VALUES (?,?,?)", rows)
    return len(rows)


def fetch_prices(conn, symbols):
    pending = [s for s in symbols if s not in done_symbols(conn, "prices")]
    print(f"\nPrices: {len(pending)} symbols pending ({len(symbols) - len(pending)} already done)")
    if not pending:
        return

    for i in range(0, len(pending), PRICE_BATCH):
        batch = pending[i:i + PRICE_BATCH]
        try:
            data = yf.download(batch, start=START_DATE, auto_adjust=True,
                               progress=False, threads=True, group_by="ticker")
        except Exception as e:
            print(f"  batch {i // PRICE_BATCH}: download failed — {type(e).__name__}: {str(e)[:120]}")
            continue

        for sym in batch:
            try:
                if len(batch) == 1:
                    closes = data["Close"]
                else:
                    closes = data[sym]["Close"]
                n = save_prices(conn, closes.dropna(), sym)
                mark(conn, sym, "prices", "done" if n else "empty")
            except Exception:
                mark(conn, sym, "prices", "error")
        conn.commit()
        print(f"  {min(i + PRICE_BATCH, len(pending))}/{len(pending)} symbols")


def fetch_vix(conn):
    existing = conn.execute("SELECT COUNT(*) FROM vix").fetchone()[0]
    if existing:
        print(f"\nVIX: already have {existing} rows, skipping")
        return
    print("\nVIX: downloading …")
    h = yf.Ticker(VIX_SYMBOL).history(start=START_DATE)
    rows = [(d.strftime("%Y-%m-%d"), float(c)) for d, c in h["Close"].items() if pd.notna(c)]
    conn.executemany("INSERT OR IGNORE INTO vix (date, close) VALUES (?,?)", rows)
    conn.commit()
    print(f"  {len(rows)} VIX days stored")


def fetch_earnings(conn, symbols):
    pending = [s for s in symbols if s not in done_symbols(conn, "earnings")]
    print(f"\nEarnings: {len(pending)} symbols pending ({len(symbols) - len(pending)} already done)")

    for idx, sym in enumerate(pending, 1):
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=100)
            if ed is None or len(ed) == 0:
                mark(conn, sym, "earnings", "empty")
            else:
                rows = []
                for ts, r in ed.iterrows():
                    rows.append((
                        sym,
                        ts.isoformat(),
                        None if pd.isna(r.get("EPS Estimate")) else float(r["EPS Estimate"]),
                        None if pd.isna(r.get("Reported EPS")) else float(r["Reported EPS"]),
                        None if pd.isna(r.get("Surprise(%)")) else float(r["Surprise(%)"]),
                    ))
                conn.executemany(
                    """INSERT OR IGNORE INTO earnings
                       (symbol, announced_at, eps_estimate, eps_actual, surprise_pct)
                       VALUES (?,?,?,?,?)""",
                    rows,
                )
                conn.commit()
                mark(conn, sym, "earnings", "done")
        except Exception as e:
            mark(conn, sym, "earnings", "error")
            print(f"  {sym}: {type(e).__name__} {str(e)[:90]}")

        if idx % 25 == 0:
            print(f"  {idx}/{len(pending)} symbols")
        time.sleep(EARNINGS_PAUSE)


def main():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=30)
    init_db(conn)

    universe = sp500_universe()
    print(f"Universe: {len(universe)} S&P 500 symbols")
    symbols = sorted(set(universe + [BENCHMARK]))

    fetch_vix(conn)
    fetch_prices(conn, symbols)
    fetch_earnings(conn, universe)

    for label, q in [
        ("price rows",      "SELECT COUNT(*) FROM prices"),
        ("symbols w/ price", "SELECT COUNT(DISTINCT symbol) FROM prices"),
        ("vix rows",        "SELECT COUNT(*) FROM vix"),
        ("earnings rows",   "SELECT COUNT(*) FROM earnings"),
        ("earnings w/ actuals",
         "SELECT COUNT(*) FROM earnings WHERE eps_actual IS NOT NULL AND eps_estimate IS NOT NULL"),
    ]:
        print(f"{label:22s}: {conn.execute(q).fetchone()[0]:,}")

    conn.close()
    print(f"\nStored in {DB_FILE}")


if __name__ == "__main__":
    main()
