# ═══════════════════════════════════════════════════════════════════════
#  Persistence for the live paper account — data/trader.db.
#
#  Positions are stored by session DATE, not calendar index, so the book
#  survives the calendar being rebuilt with a different span.
# ═══════════════════════════════════════════════════════════════════════

import sqlite3

import pandas as pd

from .backtest import cost_model
from .broker import Ledger, Order
from .engine import EngineState
from .exits import Position

SCHEMA = """
    CREATE TABLE IF NOT EXISTS account (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS holdings (symbol TEXT PRIMARY KEY, qty REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY, when_date TEXT, symbol TEXT, side TEXT, qty REAL,
        notional REAL, tag TEXT, status TEXT, fill_price REAL, fill_qty REAL, broker_id TEXT);
    CREATE TABLE IF NOT EXISTS positions (
        symbol TEXT NOT NULL, kind TEXT NOT NULL,        -- open | pending
        event_key TEXT, entry_date TEXT, due_offset INTEGER, conviction REAL,
        late_days INTEGER, qty REAL, entry_fill REAL, resets INTEGER,
        PRIMARY KEY (symbol, kind));
    CREATE TABLE IF NOT EXISTS processed (key TEXT PRIMARY KEY, date TEXT);
    CREATE TABLE IF NOT EXISTS trades (
        symbol TEXT, key TEXT, entry_date TEXT, exit_date TEXT, hold_days INTEGER,
        conviction REAL, reason TEXT, late_days INTEGER, resets INTEGER,
        entry_fill REAL, exit_fill REAL, ret REAL, bench_ret REAL, alpha REAL);
    CREATE TABLE IF NOT EXISTS decisions (
        date TEXT, symbol TEXT, action TEXT, key TEXT, conviction REAL, detail TEXT);
    CREATE TABLE IF NOT EXISTS equity (
        date TEXT PRIMARY KEY, cash REAL, stocks REAL, bench REAL, equity REAL);
    CREATE TABLE IF NOT EXISTS applied_actions (
        symbol TEXT, date TEXT, kind TEXT, PRIMARY KEY (symbol, date, kind));
    CREATE TABLE IF NOT EXISTS runs (date TEXT PRIMARY KEY, at TEXT, status TEXT);
"""


ORDER_COLS = "id, when_date, symbol, side, qty, notional, tag, status, fill_price, fill_qty, broker_id"


def open_state(path):
    conn = sqlite3.connect(path, timeout=60)
    conn.executescript(SCHEMA)
    if "broker_id" not in {r[1] for r in conn.execute("PRAGMA table_info(orders)")}:
        conn.execute("ALTER TABLE orders ADD COLUMN broker_id TEXT")     # pre-Alpaca databases
        conn.commit()
    return conn


def get(conn, key, default=None):
    row = conn.execute("SELECT value FROM account WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def put(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO account (key, value) VALUES (?, ?)", (key, str(value)))


# ── Broker ────────────────────────────────────────────────────────────
def load_ledger(conn, cfg):
    cash = get(conn, "cash")
    led = Ledger(float(cash) if cash is not None else cfg.initial_capital,
                 cost_model(cfg), cfg.min_order)
    led._pos = {s: q for s, q in conn.execute("SELECT symbol, qty FROM holdings")}
    load_orders_into(conn, led)
    return led


def load_orders_into(conn, broker):
    """Pending orders and the next order id, for any broker."""
    broker._pending = [Order(*r) for r in conn.execute(
        f"SELECT {ORDER_COLS} FROM orders WHERE status='pending' ORDER BY id")]
    broker._next_id = (conn.execute("SELECT MAX(id) FROM orders").fetchone()[0] or 0) + 1


def save_orders(conn, broker):
    conn.executemany(f"INSERT OR REPLACE INTO orders ({ORDER_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     [(o.id, o.when, o.symbol, o.side, o.qty, o.notional, o.tag, o.status,
                       o.fill_price, o.fill_qty, o.broker_id) for o in broker.history + broker.pending()])


def save_ledger(conn, led):
    put(conn, "cash", repr(led.cash()))
    conn.execute("DELETE FROM holdings")
    conn.executemany("INSERT INTO holdings (symbol, qty) VALUES (?, ?)", led.holdings().items())
    save_orders(conn, led)


# ── Engine ────────────────────────────────────────────────────────────
def load_engine_state(conn, cal):
    state = EngineState()
    for kind, sym, key, entry, off, conv, late, qty, fill, resets in conn.execute(
            """SELECT kind, symbol, event_key, entry_date, due_offset, conviction, late_days,
                      qty, entry_fill, resets FROM positions"""):
        e = cal.get_loc(pd.Timestamp(entry))
        pos = Position(sym, key, e, e + off, conv, late, qty or 0.0,
                       float("nan") if fill is None else fill, resets)
        (state.positions if kind == "open" else state.pending_entries)[sym] = pos
    state.processed = {k for (k,) in conn.execute("SELECT key FROM processed")}
    return state


def save_engine_state(conn, state, cal, today, decisions):
    conn.execute("DELETE FROM positions")
    rows = []
    for kind, book in (("open", state.positions), ("pending", state.pending_entries)):
        for sym, p in book.items():
            rows.append((sym, kind, p.event_key, cal[p.entry_idx].strftime("%Y-%m-%d"),
                         p.exit_due_idx - p.entry_idx, p.conviction, p.late_days, p.qty,
                         p.entry_fill, p.resets))
    conn.executemany("INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.executemany("INSERT OR IGNORE INTO processed (key, date) VALUES (?, ?)",
                     [(k, today) for k in state.processed])
    conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     [(t.symbol, t.key, t.entry_date.strftime("%Y-%m-%d"),
                       t.exit_date.strftime("%Y-%m-%d"), t.hold_days, t.conviction, t.reason,
                       t.late_days, t.resets, t.entry_fill, t.exit_fill, t.ret, t.bench_ret,
                       t.alpha) for t in state.trades])
    conn.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?)",
                     [(d.date, d.symbol, d.action, d.key, d.conviction, d.detail) for d in decisions])


# ── Records ───────────────────────────────────────────────────────────
def save_marks(conn, rows):
    conn.executemany("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?)", rows)


def load_marks(conn):
    return pd.read_sql_query("SELECT * FROM equity ORDER BY date", conn)


def load_trades(conn):
    t = pd.read_sql_query("SELECT * FROM trades ORDER BY exit_date", conn)
    for c in ("entry_date", "exit_date"):
        t[c] = pd.to_datetime(t[c])
    return t


def applied_actions(conn):
    return {tuple(r) for r in conn.execute("SELECT symbol, date, kind FROM applied_actions")}


def save_applied(conn, keys):
    conn.executemany("INSERT OR IGNORE INTO applied_actions VALUES (?,?,?)", keys)


def run_status(conn, date):
    row = conn.execute("SELECT status FROM runs WHERE date=?", (date,)).fetchone()
    return row[0] if row else None


def record_run(conn, date, at, status):
    if run_status(conn, date) == "decided":
        status = "decided"
    conn.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?)", (date, at, status))
