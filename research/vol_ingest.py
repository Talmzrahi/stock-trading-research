# ═══════════════════════════════════════════════════════════════════════
#  Phase 0 for the volatility-targeting work: prices and volatility
#  indices, as far back as the free sources go.
#
#  Written to data/vol.db, NOT research.db. inference_check.py carries a
#  T0 replication guard tied to research.db, and the universe repair on
#  2026-09-22 already showed how a change there silently moves recorded
#  results. This work keeps its own store.
#
#  Why this direction needs no survivorship handling: every instrument
#  here is an index ETF. There is no "delisted constituent" problem to
#  reconstruct, which is the single hardest criticism to answer elsewhere
#  in this project. The one look-ahead risk is INCEPTION — an ETF cannot
#  be held before it existed — so first/last dates are recorded per
#  symbol and must be respected by anything downstream.
#
#  VIX matters here and is already justified: it forecasts next-month
#  realised volatility better than trailing realised volatility does
#  (R² 0.517 vs 0.420 on 2002-2026). It was collected for the Phase 1
#  VIX x earnings-beat hypothesis, which was rejected — as a RETURN
#  predictor. As a volatility predictor it is the best input available.
#
#    .venv\Scripts\python.exe research\vol_ingest.py
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")

ROOT   = Path(__file__).resolve().parent.parent
DB     = ROOT / "data" / "vol.db"
START  = "1990-01-01"

# Equities, then diversifiers. Kept small on purpose: every extra asset is
# another parameter to fit, and the gate punishes that.
ETFS = {
    "SPY": "US large cap",      "QQQ": "US large-cap growth",
    "IJH": "US mid cap",        "IJR": "US small cap",
    "EFA": "developed ex-US",   "EEM": "emerging markets",
    "TLT": "long Treasuries",   "IEF": "7-10y Treasuries",
    "GLD": "gold",              "SHY": "1-3y Treasuries (cash proxy)",
}
INDICES = {"^VIX": "VIX", "^VXN": "VXN (Nasdaq)", "^GSPC": "S&P 500 price index"}


def init(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL, date TEXT NOT NULL, close REAL NOT NULL,
            PRIMARY KEY (symbol, date));
        CREATE TABLE IF NOT EXISTS coverage (
            symbol TEXT PRIMARY KEY, kind TEXT, note TEXT,
            first_date TEXT, last_date TEXT, n_days INTEGER);
    """)
    conn.commit()


def store(conn, symbol, series, kind, note):
    rows = [(symbol, d.strftime("%Y-%m-%d"), float(c))
            for d, c in series.items() if pd.notna(c)]
    if not rows:
        print(f"   {symbol:<7} nothing returned")
        return
    conn.executemany("INSERT OR REPLACE INTO prices (symbol,date,close) VALUES (?,?,?)", rows)
    conn.execute("""INSERT OR REPLACE INTO coverage
                    (symbol,kind,note,first_date,last_date,n_days) VALUES (?,?,?,?,?,?)""",
                 (symbol, kind, note, rows[0][1], rows[-1][1], len(rows)))
    conn.commit()
    span = (pd.Timestamp(rows[-1][1]) - pd.Timestamp(rows[0][1])).days / 365.25
    print(f"   {symbol:<7} {rows[0][1]} → {rows[-1][1]}  {len(rows):>6,} days ({span:.1f}y)  {note}")


def fetch(symbol, adjust):
    """adjust=True gives total return (dividends reinvested) for ETFs;
    an index level must NOT be adjusted."""
    h = yf.Ticker(symbol).history(start=START, auto_adjust=adjust)
    return h["Close"] if "Close" in h else pd.Series(dtype=float)


def main():
    conn = sqlite3.connect(DB, timeout=30)
    init(conn)
    print(f"→ {DB}\n\nETFs (total return, dividends reinvested):")
    for sym, note in ETFS.items():
        try:
            store(conn, sym, fetch(sym, True), "etf", note)
        except Exception as e:
            print(f"   {sym:<7} {type(e).__name__}: {str(e)[:70]}")
    print("\nIndices (levels, never adjusted):")
    for sym, note in INDICES.items():
        try:
            store(conn, sym.lstrip("^"), fetch(sym, False), "index", note)
        except Exception as e:
            print(f"   {sym:<7} {type(e).__name__}: {str(e)[:70]}")

    cov = pd.read_sql_query("SELECT * FROM coverage ORDER BY first_date", conn)
    conn.close()
    print(f"\n{len(cov)} instruments stored.")
    common = cov[cov.kind == "etf"].first_date.max()
    print(f"All ETFs available from: {common}  (the binding inception)")
    print(f"SPY alone reaches back to: {cov[cov.symbol=='SPY'].first_date.iloc[0]}")
    print("\nInception is the only look-ahead risk here; respect first_date downstream.")


if __name__ == "__main__":
    main()
