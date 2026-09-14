# ═══════════════════════════════════════════════════════════════════════
#  Archive the parts of research.db that cannot simply be re-downloaded.
#
#  Finnhub's free news history is a rolling ~12-month window, so the
#  scored articles here stop being retrievable once they age out — unlike
#  prices and VIX, which yfinance will serve back to 1990 any time. The
#  article text is archived alongside the scores so the corpus can be
#  re-scored with a different model later without refetching.
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

ROOT    = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "data" / "research.db"
ARCHIVE = ROOT / "data_archive"
TABLES  = ["event_articles", "earnings", "event_fetch_progress"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_schema(conn):
    here = Path(__file__).resolve().parent
    _load("ingest", here / "ingest.py").init_db(conn)
    _load("sentiment_events", here / "sentiment_events.py").init_db(conn)


def export():
    if not DB_FILE.exists():
        raise SystemExit(f"{DB_FILE} not found")
    ARCHIVE.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    for t in TABLES:
        df = pd.read_sql_query(f"SELECT * FROM {t}", conn)
        out = ARCHIVE / f"{t}.csv.gz"
        df.to_csv(out, index=False, compression="gzip")
        print(f"  {t:22s} {len(df):>8,} rows -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    conn.close()
    print(f"\nArchived to {ARCHIVE}")


def restore():
    conn = sqlite3.connect(DB_FILE)
    ensure_schema(conn)
    for t in TABLES:
        src = ARCHIVE / f"{t}.csv.gz"
        if not src.exists():
            print(f"  {t:22s} no archive file, skipped")
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
        print(f"  {t:22s} restored, table now holds {n:,} rows")
    conn.close()
    print("\nRestore complete. prices/vix are not archived — rerun research/ingest.py for those.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "export"
    if mode == "export":
        export()
    elif mode == "restore":
        restore()
    else:
        raise SystemExit("usage: archive.py [export|restore]")
