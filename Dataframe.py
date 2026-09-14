# ═══════════════════════════════════════════════════════════════════════
#  EXPORT — Build clean sentiment DataFrame and save as CSV.
#  Run AFTER Main.py has populated the local sentiment database.
# ═══════════════════════════════════════════════════════════════════════

import sys
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd

# Windows consoles default to cp1252, which can't encode the emoji in the
# print statements below (Colab's terminal is UTF-8, so this didn't show
# up there). Force UTF-8 so output doesn't crash.
sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────────────────────────────
DB_FILE      = "data/sentiment_history.db"
EXPORT_FILE  = "data/sentiment_export.csv"

# Optional filters — set to None to include everything
FILTER_FROM  = None   # e.g. "2026-01-01"
FILTER_TO    = None   # e.g. "2026-06-30"
FILTER_STOCK = None   # e.g. "NVDA"  — None means all stocks

# ─────────────────────────────────────────────────────────────────────
# Load from DB
# ─────────────────────────────────────────────────────────────────────
if not Path(DB_FILE).exists():
    raise FileNotFoundError(f"Database not found: {DB_FILE}\nRun the backfill first.")

conn = sqlite3.connect(DB_FILE)

query  = "SELECT stock, week_start, avg_score, total_articles_fetched FROM weekly_summary WHERE 1=1"
params = []

if FILTER_FROM:
    query += " AND week_start >= ?";  params.append(FILTER_FROM)
if FILTER_TO:
    query += " AND week_start <= ?";  params.append(FILTER_TO)
if FILTER_STOCK:
    query += " AND stock = ?";        params.append(FILTER_STOCK)

query += " ORDER BY week_start DESC, stock ASC"

df = pd.read_sql_query(query, conn, params=params)
conn.close()

# ─────────────────────────────────────────────────────────────────────
# Format the week column as "19 - 25/06/2026"
# ─────────────────────────────────────────────────────────────────────
def format_week(week_start_str):
    from datetime import timedelta
    start = datetime.strptime(week_start_str, "%Y-%m-%d")
    end   = start + timedelta(days=6)
    return f"{start.day} - {end.day}/{end.strftime('%m/%Y')}"

df["week"] = df["week_start"].apply(format_week)

# ─────────────────────────────────────────────────────────────────────
# Rename and reorder columns
# ─────────────────────────────────────────────────────────────────────
df = df.rename(columns={
    "stock":                  "stock",
    "avg_score":              "sentiment_score",
    "total_articles_fetched": "article_count",
})

df = df[["week", "stock", "sentiment_score", "article_count"]]

# ─────────────────────────────────────────────────────────────────────
# Preview
# ─────────────────────────────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
print(f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
print(df.head(20).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────
df.to_csv(EXPORT_FILE, index=False)
print(f"\n💾 Saved: {EXPORT_FILE}")