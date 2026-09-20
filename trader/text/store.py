# ═══════════════════════════════════════════════════════════════════════
#  Where the text pipeline keeps things.
#
#    data/edgar.db      the filings themselves, the same store the bulk
#                       download filled. Today's release is cached here,
#                       so next quarter it is this company's history.
#    data/trader.db     text_scores: one row per event, written once, the
#                       day it was scored. Never overwritten — a score is
#                       evidence about what was knowable that afternoon,
#                       and re-reading it later with a better model would
#                       destroy exactly the record worth keeping.
#
#  Everything is stored, including releases the model could not score and
#  why, so the gaps are visible rather than silently missing.
# ═══════════════════════════════════════════════════════════════════════

import json
import sqlite3
import zlib

from .novelty import PRIOR_K, fingerprint
from .parse import blocks

SCORES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS text_scores (
        event_key   TEXT PRIMARY KEY,
        scored_on   TEXT,
        symbol      TEXT,
        cik         TEXT,
        accession   TEXT,
        ann_date    TEXT,
        entry_date  TEXT,
        status      TEXT,
        n_sentences INTEGER,
        prediction  REAL,
        percentile  REAL,
        features    TEXT);
    CREATE INDEX IF NOT EXISTS ix_text_scores_date ON text_scores(entry_date);
"""


def init_scores(conn):
    conn.executescript(SCORES_SCHEMA)
    conn.commit()


def scored_keys(conn):
    return {k for (k,) in conn.execute("SELECT event_key FROM text_scores")}


def record(conn, row):
    """Write one event's score. The first write wins."""
    conn.execute(
        """INSERT OR IGNORE INTO text_scores
           (event_key, scored_on, symbol, cik, accession, ann_date, entry_date,
            status, n_sentences, prediction, percentile, features)
           VALUES (:event_key, :scored_on, :symbol, :cik, :accession, :ann_date, :entry_date,
                   :status, :n_sentences, :prediction, :percentile, :features)""",
        {**row, "features": json.dumps(row.get("features")) if row.get("features") else None})
    conn.commit()


def percentiles(conn):
    """{event_key: percentile} for every event scored so far."""
    return {k: p for k, p in conn.execute(
        "SELECT event_key, percentile FROM text_scores WHERE percentile IS NOT NULL")}


# ── the filing cache ───────────────────────────────────────────────────

def html_of(edgar, accession):
    row = edgar.execute("SELECT html FROM release_html WHERE accession = ?", (accession,)).fetchone()
    return zlib.decompress(row[0]).decode("utf-8") if row and row[0] else None


def cache_release(edgar, cik, symbol, meta, doc_type, filename, html):
    """Keep today's filing where the bulk download put every other one."""
    edgar.execute("INSERT OR IGNORE INTO company_8ks VALUES (?,?,?,?,?)",
                  (meta["accession"], cik, meta["filed_date"], meta["accepted_at"], meta["items"]))
    edgar.execute("INSERT OR REPLACE INTO filings VALUES (?,?,?,?,?,?,?,?,?)",
                  (meta["accession"], cik, symbol, meta["filed_date"], meta["accepted_at"],
                   meta["items"], filename, len(html), None))
    edgar.execute("INSERT OR REPLACE INTO release_html VALUES (?,?,?,?,?)",
                  (meta["accession"], doc_type, filename, len(html),
                   zlib.compress(html.encode("utf-8"), 9)))
    edgar.commit()


def history(edgar, cik, before_date, k=PRIOR_K):
    """Fingerprints of the company's k releases before `before_date`,
    oldest first — what layer 0 compares today's release against."""
    rows = edgar.execute(
        """SELECT f.accession FROM filings f JOIN release_html h USING (accession)
           WHERE f.cik = ? AND f.filed_date < ? AND h.html IS NOT NULL
           ORDER BY f.filed_date DESC LIMIT ?""", (cik, before_date, k)).fetchall()
    out = []
    for (accession,) in reversed(rows):
        html = html_of(edgar, accession)
        if html:
            out.append(fingerprint(blocks(html)))
    return out


def open_edgar(path):
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
