# ═══════════════════════════════════════════════════════════════════════
#  Archive the parts of the research databases that cannot simply be
#  re-downloaded.
#
#  Finnhub's free news history is a rolling ~12-month window, so the
#  scored articles here stop being retrievable once they age out — unlike
#  prices and VIX, which yfinance will serve back to 1990 any time. The
#  article text is archived alongside the scores so the corpus can be
#  re-scored with a different model later without refetching.
#
#  Index membership is archived for the same reason: rebuilding it means
#  re-scraping ~150 Wikipedia revisions (about 90 minutes), and old
#  revisions can be edited or deleted. The S&P 400/600 membership in
#  data/oos_midsmall.db is archived alongside it, prefixed oos_.
#
#  prices/vix are deliberately excluded: 2.7M rows, and re-downloadable.
#
#    python research/archive.py export
#    python research/archive.py restore
# ═══════════════════════════════════════════════════════════════════════

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT       = Path(__file__).resolve().parent.parent
DB_FILE    = ROOT / "data" / "research.db"
OOS_DB     = ROOT / "data" / "oos_midsmall.db"
ARCHIVE    = ROOT / "data_archive"
TABLES     = ["event_articles", "earnings", "event_fetch_progress",
              "universe_history", "universe_ids", "departed_coverage"]
OOS_TABLES = ["universe_history", "universe_ids", "snapshot_log", "coverage"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_schema(conn):
    here = Path(__file__).resolve().parent
    _load("ingest", here / "ingest.py").init_db(conn)
    _load("sentiment_events", here / "sentiment_events.py").init_db(conn)
    _load("universe", here / "universe.py").init_db(conn)
    _load("universe_ids", here / "universe_ids.py").init(conn)
    _load("ingest_departed", here / "ingest_departed.py").init(conn)


def _dump(conn, tables, prefix=""):
    for t in tables:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {t}", conn)
        except Exception:
            print(f"  {prefix + t:24s} not in this database, skipped")
            continue
        out = ARCHIVE / f"{prefix}{t}.csv.gz"
        df.to_csv(out, index=False, compression="gzip")
        print(f"  {prefix + t:24s} {len(df):>8,} rows -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")


def _restore_tables(conn, tables, prefix=""):
    for t in tables:
        src = ARCHIVE / f"{prefix}{t}.csv.gz"
        if not src.exists():
            print(f"  {prefix + t:24s} no archive file, skipped")
            continue
        df = pd.read_csv(src)
        cols = ",".join(df.columns)
        placeholders = ",".join("?" * len(df.columns))
        conn.executemany(
            f"INSERT OR IGNORE INTO {t} ({cols}) VALUES ({placeholders})",
            df.astype(object).where(pd.notna(df), None).itertuples(index=False, name=None),
        )
        conn.commit()
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {prefix + t:24s} restored, table now holds {n:,} rows")


def export():
    if not DB_FILE.exists():
        raise SystemExit(f"{DB_FILE} not found")
    ARCHIVE.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    _dump(conn, TABLES)
    conn.close()
    if OOS_DB.exists():
        conn = sqlite3.connect(OOS_DB)
        _dump(conn, OOS_TABLES, prefix="oos_")
        conn.close()
    print(f"\nArchived to {ARCHIVE}")


def restore():
    conn = sqlite3.connect(DB_FILE)
    ensure_schema(conn)
    _restore_tables(conn, TABLES)
    conn.close()
    if any((ARCHIVE / f"oos_{t}.csv.gz").exists() for t in OOS_TABLES):
        OOS_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(OOS_DB)
        conn.executescript(_load("oos", Path(__file__).resolve().parent / "oos_midsmall_data.py").SCHEMA)
        _restore_tables(conn, OOS_TABLES, prefix="oos_")
        conn.close()
    print("\nRestore complete. prices/vix/earnings are not archived — rerun "
          "research/ingest.py, research/ingest_departed.py and "
          "research/oos_midsmall_data.py for those.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "export"
    if mode == "export":
        export()
    elif mode == "restore":
        restore()
    else:
        raise SystemExit("usage: archive.py [export|restore]")
