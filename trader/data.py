# ═══════════════════════════════════════════════════════════════════════
#  Read access to data/research.db — the same tables research/ uses.
#  Live refreshing of those tables lives in trader/refresh.py.
# ═══════════════════════════════════════════════════════════════════════

import sqlite3

import pandas as pd

from .config import ROOT

RESEARCH_DB = ROOT / "data" / "research.db"
TRADER_DB   = ROOT / "data" / "trader.db"


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
    u = pd.read_sql_query("SELECT as_of, symbol FROM universe_history", conn)
    u["as_of"] = pd.to_datetime(u["as_of"])
    return u
