# ═══════════════════════════════════════════════════════════════════════
#  Read access to data/research.db — the same tables research/ uses.
#  Live refreshing of those tables lives in trader/refresh.py.
# ═══════════════════════════════════════════════════════════════════════

import sqlite3

import pandas as pd

from .config import ROOT

RESEARCH_DB = ROOT / "data" / "research.db"
TRADER_DB   = ROOT / "data" / "trader.db"
EDGAR_DB    = ROOT / "data" / "edgar.db"      # cached earnings releases (trader/text)


def connect(path=RESEARCH_DB):
    return sqlite3.connect(path, timeout=60)


def load_closes(conn):
    """Split/dividend-adjusted closes, dates x symbols."""
    p = pd.read_sql_query("SELECT symbol, date, close FROM prices", conn)
    p["date"] = pd.to_datetime(p["date"])
    return p.pivot(index="date", columns="symbol", values="close").sort_index()


def load_earnings(conn):
    e = pd.read_sql_query(
        """SELECT symbol, announced_at, eps_estimate, eps_actual FROM earnings
           WHERE eps_actual IS NOT NULL AND eps_estimate IS NOT NULL""", conn)
    e["announced_at"] = pd.to_datetime(e["announced_at"], utc=True, format="mixed")
    return e


def load_universe(conn):
    """Snapshots of index membership, with a `firm` identity (SEC CIK) when
    research/universe_ids.py has been run — see trader/events.py."""
    u = pd.read_sql_query("SELECT as_of, symbol FROM universe_history", conn)
    has_ids = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='universe_ids'").fetchone()
    if has_ids:
        ids = pd.read_sql_query("SELECT as_of, symbol, firm FROM universe_ids WHERE firm IS NOT NULL", conn)
        if len(ids):
            u = u.merge(ids, on=["as_of", "symbol"], how="left")
            u["firm"] = u.firm.fillna("SYM:" + u.symbol)
    u["as_of"] = pd.to_datetime(u["as_of"])
    return u
