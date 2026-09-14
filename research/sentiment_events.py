# ═══════════════════════════════════════════════════════════════════════
#  Pre-earnings news sentiment, for the fusion test.
#
#  For every earnings event in the Finnhub news window (~12 months), pulls
#  the news published BEFORE the announcement and scores it with the same
#  3-model ensemble Main.py uses. Strictly pre-announcement: no look-ahead.
#
#  Two resumable phases so a crash in scoring never loses fetched news:
#    python research/sentiment_events.py fetch
#    python research/sentiment_events.py score
#  (or `python research/sentiment_events.py all` to run both)
# ═══════════════════════════════════════════════════════════════════════

import os
import random
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

DB_FILE       = Path(__file__).resolve().parent.parent / "data" / "research.db"
WINDOW_DAYS   = 14       # look this many days before the announcement
MAX_ARTICLES  = 25       # per event; sampled if the window has more
BATCH_SIZE    = 32       # articles per model forward pass
CHUNK         = 1024     # rows pulled per pass, length-sorted before batching
MAX_TOKENS    = 192      # news summaries are short; 512 just pads dead space
CALLS_PER_MIN = 50       # Finnhub free tier allows 60/min

# Same ensemble configuration as Main.py, so scores are comparable.
NEUTRAL_BIAS  = {"finbert": 0.0244, "fintone": 0.0204, "twitter": 0.1337}
MODEL_WEIGHTS = {"finbert": 0.34,   "fintone": 0.33,   "twitter": 0.33}
MODEL_TEMPS   = {"finbert": 1.5,    "fintone": 4.0,    "twitter": 2.0}

_rate_lock, _next_at = threading.Lock(), [0.0]


def wait_rate():
    with _rate_lock:
        now = time.monotonic()
        if now < _next_at[0]:
            time.sleep(_next_at[0] - now)
            now = time.monotonic()
        _next_at[0] = now + 60.0 / CALLS_PER_MIN


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS event_articles (
            symbol       TEXT NOT NULL,
            event_ts     TEXT NOT NULL,
            published_ts INTEGER,
            headline     TEXT,
            summary      TEXT,
            finbert      REAL,
            fintone      REAL,
            twitter      REAL,
            final_score  REAL,
            scored       INTEGER DEFAULT 0,
            UNIQUE(symbol, event_ts, published_ts, headline)
        );
        CREATE INDEX IF NOT EXISTS ix_unscored ON event_articles(scored);
        CREATE TABLE IF NOT EXISTS event_fetch_progress (
            symbol   TEXT NOT NULL,
            event_ts TEXT NOT NULL,
            status   TEXT NOT NULL,
            n        INTEGER,
            PRIMARY KEY (symbol, event_ts)
        );
    """)
    conn.commit()


def events_in_window(conn):
    df = pd.read_sql_query(
        """SELECT symbol, announced_at, surprise_pct FROM earnings
           WHERE eps_actual IS NOT NULL AND eps_estimate IS NOT NULL""",
        conn,
    )
    df["ts"] = pd.to_datetime(df.announced_at, utc=True, format="mixed")
    now = pd.Timestamp.now(tz="UTC")
    # Finnhub free news reaches back ~12 months; leave a margin so the
    # pre-announcement window is inside coverage rather than half-empty.
    lo = now - pd.Timedelta(days=365 - WINDOW_DAYS)
    return df[(df.ts >= lo) & (df.ts <= now)].reset_index(drop=True)


# ── Phase 1: fetch ────────────────────────────────────────────────────
def phase_fetch(conn):
    import finnhub

    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        raise SystemExit("FINNHUB_API_KEY is not set")
    client = finnhub.Client(api_key=key)

    ev = events_in_window(conn)
    done = {(r[0], r[1]) for r in conn.execute(
        "SELECT symbol, event_ts FROM event_fetch_progress WHERE status IN ('done','empty')")}
    todo = [r for r in ev.itertuples() if (r.symbol, r.ts.isoformat()) not in done]
    print(f"Fetch: {len(todo)} events pending ({len(ev) - len(todo)} already done)")

    for i, r in enumerate(todo, 1):
        ev_ts = r.ts.isoformat()
        end = (r.ts.date() - timedelta(days=1))
        start = end - timedelta(days=WINDOW_DAYS - 1)
        try:
            wait_rate()
            news = client.company_news(r.symbol, _from=str(start), to=str(end))
            cutoff = int(r.ts.timestamp())
            news = [a for a in (news or []) if a.get("datetime", 0) < cutoff]
            if len(news) > MAX_ARTICLES:
                news = random.sample(news, MAX_ARTICLES)
            rows = [(r.symbol, ev_ts, a.get("datetime"),
                     (a.get("headline") or "")[:300], (a.get("summary") or "")[:600])
                    for a in news
                    if (a.get("summary") or a.get("headline"))]
            if rows:
                conn.executemany(
                    """INSERT OR IGNORE INTO event_articles
                       (symbol, event_ts, published_ts, headline, summary)
                       VALUES (?,?,?,?,?)""", rows)
            conn.execute(
                "INSERT OR REPLACE INTO event_fetch_progress (symbol,event_ts,status,n) VALUES (?,?,?,?)",
                (r.symbol, ev_ts, "done" if rows else "empty", len(rows)))
            conn.commit()
        except Exception as e:
            conn.execute(
                "INSERT OR REPLACE INTO event_fetch_progress (symbol,event_ts,status,n) VALUES (?,?,?,?)",
                (r.symbol, ev_ts, "error", 0))
            conn.commit()
            print(f"  {r.symbol} {ev_ts[:10]}: {type(e).__name__} {str(e)[:80]}")

        if i % 100 == 0:
            n = conn.execute("SELECT COUNT(*) FROM event_articles").fetchone()[0]
            print(f"  {i}/{len(todo)} events — {n:,} articles stored")

    n = conn.execute("SELECT COUNT(*) FROM event_articles").fetchone()[0]
    print(f"Fetch complete — {n:,} articles stored")


# ── Phase 2: score ────────────────────────────────────────────────────
def phase_score(conn):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    todo = conn.execute("SELECT COUNT(*) FROM event_articles WHERE scored=0").fetchone()[0]
    if not todo:
        print("Score: nothing pending")
        return
    print(f"Score: {todo:,} articles pending")

    print("loading models …")
    specs = {}
    for name, repo in [("finbert", "ProsusAI/finbert"),
                       ("fintone", "ldeb/solved-finbert-tone"),
                       ("twitter", "cardiffnlp/twitter-roberta-base-sentiment-latest")]:
        tok = AutoTokenizer.from_pretrained(repo)
        mod = AutoModelForSequenceClassification.from_pretrained(repo)
        mod.eval()
        lab = mod.config.id2label
        if name == "finbert":
            pos, neg = 0, 1
        else:
            pos = next(k for k, v in lab.items() if "pos" in v.lower())
            neg = next(k for k, v in lab.items() if "neg" in v.lower())
        specs[name] = (tok, mod, pos, neg)
    print("models ready")

    def score_batch(texts):
        out = {}
        for name, (tok, mod, pos, neg) in specs.items():
            src = [re.sub(r"@\w+", "@user", re.sub(r"http\S+", "http", t))
                   for t in texts] if name == "twitter" else texts
            enc = tok(src, return_tensors="pt", truncation=True, max_length=MAX_TOKENS, padding=True)
            with torch.no_grad():
                logits = mod(**enc).logits
            probs = F.softmax(logits / MODEL_TEMPS[name], dim=-1)
            net = (probs[:, pos] - probs[:, neg] - NEUTRAL_BIAS[name]).clamp(-1, 1)
            out[name] = net.tolist()
        return out

    t0, seen = time.time(), 0
    while True:
        rows = conn.execute(
            """SELECT rowid, headline, summary FROM event_articles
               WHERE scored=0 LIMIT ?""", (CHUNK,)).fetchall()
        if not rows:
            break
        prepped = [(r[0], ((r[2] if r[2] and len(r[2].strip()) >= 10 else (r[1] or ""))[:600] or "n/a"))
                   for r in rows]
        # Length-sort so each batch pads to a similar length instead of to the
        # longest outlier in an arbitrary mix.
        prepped.sort(key=lambda x: len(x[1]))

        for i in range(0, len(prepped), BATCH_SIZE):
            chunk = prepped[i:i + BATCH_SIZE]
            s = score_batch([t for _, t in chunk])
            upd = []
            for j, (rid, _) in enumerate(chunk):
                fb, ft, tw = s["finbert"][j], s["fintone"][j], s["twitter"][j]
                final = (fb * MODEL_WEIGHTS["finbert"] + ft * MODEL_WEIGHTS["fintone"]
                         + tw * MODEL_WEIGHTS["twitter"])
                upd.append((fb, ft, tw, final, rid))
            conn.executemany(
                """UPDATE event_articles SET finbert=?, fintone=?, twitter=?, final_score=?, scored=1
                   WHERE rowid=?""", upd)
            conn.commit()
            seen += len(chunk)
            if seen % (BATCH_SIZE * 20) < BATCH_SIZE:
                rate = seen / (time.time() - t0)
                print(f"  {seen:,}/{todo:,} scored — {rate:.1f}/s — eta {(todo-seen)/rate/60:.0f}m")

    print(f"Score complete — {seen:,} articles in {(time.time()-t0)/60:.1f}m")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    init_db(conn)
    if mode in ("fetch", "all"):
        phase_fetch(conn)
    if mode in ("score", "all"):
        phase_score(conn)
    conn.close()


if __name__ == "__main__":
    main()
