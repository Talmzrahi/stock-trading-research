# ═══════════════════════════════════════════════════════════════════════
#  Layer 1, live: keep research.db current for the daily run.
#
#  universe  current S&P 500 from Wikipedia, weekly; a new snapshot row is
#            written only when membership actually changed
#  prices    last ~400 days re-downloaded every run (a dividend rewrites
#            adjusted history), full history monthly. Adjusted closes go
#            to `prices` (what research/ and the signal use), raw closes to
#            `raw_prices` (what the simulated broker fills at), dividends
#            and splits to `corporate_actions`
#  earnings  daily for symbols whose report is due or just happened and
#            still lacks an actual; every member weekly to catch reschedules
# ═══════════════════════════════════════════════════════════════════════

import time
from datetime import datetime, timedelta
from datetime import time as dtime
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

from .market_calendar import ET

WIKI_URL             = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT           = {"User-Agent": "Mozilla/5.0 (sp500 research script)"}
HISTORY_START        = "2002-01-01"
PRICE_LOOKBACK_DAYS  = 400
PRICE_BATCH          = 100
BAR_FINAL_AT         = dtime(16, 30)   # ET; before this, today's daily bar is still moving
EARNINGS_WINDOW_DAYS = 4
EARNINGS_PAUSE       = 0.4

SCHEMA = """
    CREATE TABLE IF NOT EXISTS prices (
        symbol TEXT NOT NULL, date TEXT NOT NULL, close REAL NOT NULL,
        PRIMARY KEY (symbol, date));
    CREATE TABLE IF NOT EXISTS earnings (
        symbol TEXT NOT NULL, announced_at TEXT NOT NULL,
        eps_estimate REAL, eps_actual REAL, surprise_pct REAL,
        PRIMARY KEY (symbol, announced_at));
    CREATE TABLE IF NOT EXISTS universe_history (
        as_of TEXT NOT NULL, symbol TEXT NOT NULL,
        PRIMARY KEY (as_of, symbol));
    CREATE TABLE IF NOT EXISTS raw_prices (
        symbol TEXT NOT NULL, date TEXT NOT NULL, close REAL NOT NULL,
        PRIMARY KEY (symbol, date));
    CREATE TABLE IF NOT EXISTS corporate_actions (
        symbol TEXT NOT NULL, date TEXT NOT NULL, kind TEXT NOT NULL, value REAL NOT NULL,
        PRIMARY KEY (symbol, date, kind));
    CREATE TABLE IF NOT EXISTS universe_ids (
        as_of TEXT NOT NULL, symbol TEXT NOT NULL, name TEXT, cik TEXT, firm TEXT,
        PRIMARY KEY (as_of, symbol));
    CREATE TABLE IF NOT EXISTS refresh_log (kind TEXT PRIMARY KEY, at TEXT NOT NULL);
"""


def init_tables(conn):
    conn.executescript(SCHEMA)


def due(conn, kind, days, now):
    row = conn.execute("SELECT at FROM refresh_log WHERE kind=?", (kind,)).fetchone()
    return row is None or now - datetime.fromisoformat(row[0]) >= timedelta(days=days)


def mark_done(conn, kind, now):
    conn.execute("INSERT OR REPLACE INTO refresh_log (kind, at) VALUES (?, ?)", (kind, now.isoformat()))
    conn.commit()


# ── Universe ──────────────────────────────────────────────────────────
def current_sp500():
    """Current constituents: symbol, name, and SEC CIK (the firm identity
    that survives ticker renames)."""
    html = requests.get(WIKI_URL, headers=USER_AGENT, timeout=30).text
    table = pd.read_html(StringIO(html))[0]
    df = pd.DataFrame({
        "symbol": table["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False),
        "name": table["Security"].astype(str),
        "cik": pd.to_numeric(table["CIK"], errors="coerce").map(
            lambda x: f"{int(x):010d}" if pd.notna(x) else None),
    })
    return df.drop_duplicates("symbol").sort_values("symbol")


def latest_members(conn):
    return sorted(r[0] for r in conn.execute(
        "SELECT symbol FROM universe_history WHERE as_of=(SELECT MAX(as_of) FROM universe_history)"))


def refresh_universe(conn, now, every_days=7):
    if not due(conn, "universe", every_days, now):
        return None
    table = current_sp500()
    if len(table) < 450:
        raise RuntimeError(f"parsed only {len(table)} symbols from Wikipedia")
    prev, new = set(latest_members(conn)), set(table.symbol)
    if new != prev:
        as_of = now.date().isoformat()
        conn.executemany("INSERT OR IGNORE INTO universe_history (as_of, symbol) VALUES (?,?)",
                         [(as_of, s) for s in table.symbol])
        conn.executemany(
            "INSERT OR REPLACE INTO universe_ids (as_of, symbol, name, cik, firm) VALUES (?,?,?,?,?)",
            [(as_of, r.symbol, r.name, r.cik, r.cik or f"SYM:{r.symbol}") for r in table.itertuples()])
    mark_done(conn, "universe", now)
    return {"added": sorted(new - prev), "removed": sorted(prev - new)}


# ── Prices ────────────────────────────────────────────────────────────
def _ticker_frame(data, sym, single):
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        for level in range(cols.nlevels):
            if sym in cols.get_level_values(level):
                return data.xs(sym, axis=1, level=level)
        return None
    return data if single else None


def last_final_bar(now):
    return now.date() if now.time() >= BAR_FINAL_AT else now.date() - timedelta(days=1)


def refresh_prices(conn, symbols, now, full=False):
    recent = (now.date() - timedelta(days=PRICE_LOOKBACK_DAYS)).isoformat()
    start = HISTORY_START if full else recent
    final = pd.Timestamp(last_final_bar(now))
    written, missing = 0, []
    for i in range(0, len(symbols), PRICE_BATCH):
        batch = symbols[i:i + PRICE_BATCH]
        try:
            data = yf.download(batch, start=start, auto_adjust=False, actions=True,
                               progress=False, threads=True, group_by="ticker")
        except Exception:
            missing.extend(batch)
            continue
        for sym in batch:
            df = _ticker_frame(data, sym, len(batch) == 1)
            if df is None or "Close" not in df.columns:
                missing.append(sym)
                continue
            idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
            df = df.set_axis(idx.normalize())
            df = df[(df.index <= final) & df["Close"].notna()]
            if df.empty:
                missing.append(sym)
                continue
            dates = df.index.strftime("%Y-%m-%d")
            adj = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
            conn.executemany("INSERT OR REPLACE INTO prices (symbol, date, close) VALUES (?,?,?)",
                             [(sym, d, float(c)) for d, c in zip(dates, adj) if pd.notna(c)])
            conn.executemany("INSERT OR REPLACE INTO raw_prices (symbol, date, close) VALUES (?,?,?)",
                             [(sym, d, float(c)) for d, c in zip(dates, df["Close"]) if d >= recent])
            for kind, name in (("dividend", "Dividends"), ("split", "Stock Splits")):
                if name in df.columns:
                    s = df[name].fillna(0)
                    s = s[s != 0]
                    conn.executemany(
                        "INSERT OR REPLACE INTO corporate_actions (symbol, date, kind, value) VALUES (?,?,?,?)",
                        [(sym, d.strftime("%Y-%m-%d"), kind, float(v)) for d, v in s.items()])
            written += len(df)
        conn.commit()
    if full:
        mark_done(conn, "prices_full", now)
    return written, missing


def load_raw_closes(conn):
    p = pd.read_sql_query("SELECT symbol, date, close FROM raw_prices", conn)
    if p.empty:
        return pd.DataFrame(dtype=float)
    p["date"] = pd.to_datetime(p["date"])
    return p.pivot(index="date", columns="symbol", values="close").sort_index()


def load_actions(conn):
    return pd.read_sql_query("SELECT symbol, date, kind, value FROM corporate_actions", conn)


# ── Earnings ──────────────────────────────────────────────────────────
def _num(x):
    return None if x is None or pd.isna(x) else float(x)


def _et_dates(df):
    return pd.to_datetime(df.announced_at, utc=True, format="mixed").dt.tz_convert(ET).dt.date


def earnings_targets(conn, members, today, window=EARNINGS_WINDOW_DAYS):
    """Members whose report is due or just happened without an actual yet,
    plus members with no earnings history at all (new index additions)."""
    df = pd.read_sql_query("SELECT symbol, announced_at, eps_actual FROM earnings", conn)
    if df.empty:
        return list(members)
    d = _et_dates(df)
    pending = set(df.symbol[(d >= today - timedelta(days=window)) & (d <= today) & df.eps_actual.isna()])
    return sorted((pending & set(members)) | (set(members) - set(df.symbol)))


def refresh_earnings(conn, symbols, pause=EARNINGS_PAUSE):
    ok, failed = 0, []
    for sym in symbols:
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=12)
        except Exception:
            failed.append(sym)
            time.sleep(pause)
            continue
        if ed is not None and len(ed):
            rows = [(sym, ts.isoformat(), _num(r.get("EPS Estimate")), _num(r.get("Reported EPS")),
                     _num(r.get("Surprise(%)"))) for ts, r in ed.iterrows()]
            # Replace the window yfinance just returned: announcement times
            # shift when a date is confirmed, which would otherwise leave a
            # stale duplicate row for the same quarter.
            lo = ed.index.min() - pd.Timedelta(days=3)
            stale = [(sym, a) for (a,) in conn.execute(
                "SELECT announced_at FROM earnings WHERE symbol=?", (sym,)) if pd.Timestamp(a) >= lo]
            conn.executemany("DELETE FROM earnings WHERE symbol=? AND announced_at=?", stale)
            conn.executemany("""INSERT OR REPLACE INTO earnings
                                (symbol, announced_at, eps_estimate, eps_actual, surprise_pct)
                                VALUES (?,?,?,?,?)""", rows)
            conn.commit()
            ok += 1
        time.sleep(pause)
    return ok, failed


def upcoming_reports(conn, members, today, days=7):
    df = pd.read_sql_query("SELECT symbol, announced_at FROM earnings WHERE eps_actual IS NULL", conn)
    if df.empty:
        return df
    ts = pd.to_datetime(df.announced_at, utc=True, format="mixed").dt.tz_convert(ET)
    df["date"], df["time"] = ts.dt.date, ts.dt.strftime("%H:%M")
    df = df[(df.date > today) & (df.date <= today + timedelta(days=days)) & df.symbol.isin(members)]
    return df.sort_values(["date", "symbol"])[["date", "time", "symbol"]]
