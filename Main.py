# ═══════════════════════════════════════════════════════════════════════
#  Sentiment backfill — fetches news for a stock list, scores it with a
#  3-model NLP ensemble, and stores decay-weighted weekly sentiment in a
#  local SQLite database.
#
#  Setup (run once):
#    pip install -r requirements.txt
#  Then set FINNHUB_API_KEY (see CONFIG below) and run:
#    python Main.py
# ═══════════════════════════════════════════════════════════════════════

import os
import sys

# Windows consoles default to cp1252, which can't encode the emoji used in
# the print statements below (Colab's terminal is UTF-8, so this didn't
# show up there). Force UTF-8 so status output doesn't crash mid-run.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────
# ⚙️  CONFIG — change these if you want
# ─────────────────────────────────────────────────────────────────────
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
MONTHS_BACK     = 1       # how many months of history to fetch (change to e.g. 24 for 2 years)
DB_FILE         = "data/sentiment_history.db"  # saved locally

STOCKS = [
    "SPY", "DIA", "VTI", "NVDA", "GOOGL",
    "AVGO", "AAPL", "TSLA", "MSFT", "AMZN",
    "WMT", "META", "MU", "RGTI", "IONQ",
    "QBTS", "CORZ", "NBIS", "BE",
    "VRT", "CEG", "ETN", "LUNR", "ASTS",
    "RKLB", "EOSE", "LLY", "KOFOL.PR"
]

# ─────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────
import math
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import finnhub
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
LAMBDA                = 0.1
MAX_WORKERS           = 3
TARGET_CALLS_PER_MIN  = 50   # global cap across ALL worker threads combined — Finnhub free tier allows 60/min
CLT_SAMPLE_SIZE       = 50   # if a week has >50 articles, randomly sample 50 (CLT threshold)

NEUTRAL_BIAS  = {"finbert": 0.0244, "fintone": 0.0204, "twitter": 0.1337}
MODEL_WEIGHTS = {"finbert": 0.34,   "fintone": 0.33,   "twitter": 0.33}
MODEL_TEMPS   = {"finbert": 1.5,    "fintone": 4.0,    "twitter": 2.0}

# ─────────────────────────────────────────────────────────────────────
# Global rate limiter
# Previously each worker thread slept independently after its own call,
# which meant 3 workers achieved ~3x the intended rate (the cause of the
# 429 "Too many requests" errors). This version uses one shared lock so
# the COMBINED rate across all threads never exceeds the target.
# ─────────────────────────────────────────────────────────────────────
_rate_lock      = threading.Lock()
_next_call_at   = [0.0]
_MIN_INTERVAL   = 60.0 / TARGET_CALLS_PER_MIN

def wait_for_rate_limit():
    with _rate_lock:
        now = time.monotonic()
        if now < _next_call_at[0]:
            time.sleep(_next_call_at[0] - now)
            now = time.monotonic()
        _next_call_at[0] = now + _MIN_INTERVAL

# ─────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────
print("Loading sentiment models …")

_finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
_finbert_model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
_finbert_model.eval()

_fintone_tokenizer = AutoTokenizer.from_pretrained("ldeb/solved-finbert-tone")
_fintone_model     = AutoModelForSequenceClassification.from_pretrained("ldeb/solved-finbert-tone")
_fintone_model.eval()
_fintone_id2label  = _fintone_model.config.id2label
_fintone_pos_idx   = next(k for k, v in _fintone_id2label.items() if "pos" in v.lower())
_fintone_neg_idx   = next(k for k, v in _fintone_id2label.items() if "neg" in v.lower())

_twitter_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
_twitter_model     = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
_twitter_model.eval()
_twitter_id2label  = _twitter_model.config.id2label
_twitter_pos_idx   = next(k for k, v in _twitter_id2label.items() if "pos" in v.lower())
_twitter_neg_idx   = next(k for k, v in _twitter_id2label.items() if "neg" in v.lower())

print("Models ready ✅\n")

# ─────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────
def preprocess_for_twitter(text):
    text = re.sub(r'http\S+', 'http', text)
    text = re.sub(r'@\w+', '@user', text)
    return text

def run_model(text, model, tokenizer, pos_idx, neg_idx, temperature=1.0, bias=0.0):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs     = F.softmax(outputs.logits / temperature, dim=-1)[0]
    raw_net   = probs[pos_idx].item() - probs[neg_idx].item()
    net_score = round(max(-1.0, min(1.0, raw_net - bias)), 3)
    label     = "Positive" if net_score >= 0.1 else "Negative" if net_score <= -0.1 else "Neutral"
    return label, round(probs.max().item(), 3), net_score

def run_finbert(text):
    return run_model(text, _finbert_model, _finbert_tokenizer,
                     0, 1, MODEL_TEMPS["finbert"], NEUTRAL_BIAS["finbert"])

def run_fintone(text):
    return run_model(text, _fintone_model, _fintone_tokenizer,
                     _fintone_pos_idx, _fintone_neg_idx,
                     MODEL_TEMPS["fintone"], NEUTRAL_BIAS["fintone"])

def run_twitter(text):
    return run_model(preprocess_for_twitter(text),
                     _twitter_model, _twitter_tokenizer,
                     _twitter_pos_idx, _twitter_neg_idx,
                     MODEL_TEMPS["twitter"], NEUTRAL_BIAS["twitter"])

def ensemble_finance(summary):
    if not summary or len(summary.strip()) < 10:
        return None
    runners = {"finbert": run_finbert, "fintone": run_fintone, "twitter": run_twitter}
    results = {}
    for name, fn in runners.items():
        try:
            label, conf, score = fn(summary)
            results[name] = {"label": label, "conf": conf, "net_score": score}
        except Exception as e:
            print(f"  ⚠️  {name} error: {e}")
            results[name] = {"label": "Neutral", "conf": 0.0, "net_score": 0.0}

    final_score = round(
        max(-1.0, min(1.0,
            sum(results[n]["net_score"] * MODEL_WEIGHTS[n] for n in results))),
        3,
    )
    final_label = "Positive" if final_score >= 0.1 else "Negative" if final_score <= -0.1 else "Neutral"
    label_agree = len(set(v["label"] for v in results.values())) == 1

    return {
        "finbert_label": results["finbert"]["label"], "finbert_conf": results["finbert"]["conf"],
        "finbert_score": results["finbert"]["net_score"],
        "fintone_label": results["fintone"]["label"], "fintone_conf": results["fintone"]["conf"],
        "fintone_score": results["fintone"]["net_score"],
        "twitter_label": results["twitter"]["label"], "twitter_conf": results["twitter"]["conf"],
        "twitter_score": results["twitter"]["net_score"],
        "final_score":   final_score,
        "final_label":   final_label,
        "agreement":     label_agree,
    }

# ─────────────────────────────────────────────────────────────────────
# Temporal decay
# ─────────────────────────────────────────────────────────────────────
def trading_days_elapsed(article_timestamp, reference_ts=None):
    article_dt = pd.Timestamp(article_timestamp, unit='s')
    reference  = pd.Timestamp(reference_ts) if reference_ts else pd.Timestamp.now()
    if article_dt >= reference:
        return 0.0
    trading_days = np.busday_count(article_dt.date(), reference.date())
    intraday     = (reference - reference.normalize()).seconds / 86400
    if trading_days == 0:
        hours_elapsed = (reference - article_dt).total_seconds() / 3600
        return min(1.0, hours_elapsed / 8)
    return float(trading_days) + intraday

# ─────────────────────────────────────────────────────────────────────
# SQLite setup
# ─────────────────────────────────────────────────────────────────────
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            stock          TEXT    NOT NULL,
            week_start     TEXT    NOT NULL,
            article_ts     INTEGER,
            summary        TEXT,
            age_in_tdays   REAL,
            decay_weight   REAL,
            finbert_label  TEXT,  finbert_conf REAL,  finbert_score REAL,
            fintone_label  TEXT,  fintone_conf REAL,  fintone_score REAL,
            twitter_label  TEXT,  twitter_conf REAL,  twitter_score REAL,
            final_score    REAL,
            final_label    TEXT,
            agreement      INTEGER,
            UNIQUE(stock, week_start, summary)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weekly_summary (
            stock                   TEXT  NOT NULL,
            week_start              TEXT  NOT NULL,
            avg_score               REAL,
            min_score               REAL,
            max_score               REAL,
            total_articles_scored   INTEGER,
            total_articles_fetched  INTEGER,
            agreements              INTEGER,
            overall                 TEXT,
            PRIMARY KEY (stock, week_start)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            stock       TEXT NOT NULL,
            week_start  TEXT NOT NULL,
            status      TEXT NOT NULL,
            fetched_at  TEXT,
            PRIMARY KEY (stock, week_start)
        )
    """)
    conn.commit()
    return conn

def already_done(conn, stock, week_start):
    return conn.execute(
        "SELECT 1 FROM progress WHERE stock=? AND week_start=? AND status='done'",
        (stock, week_start),
    ).fetchone() is not None

def save_articles(conn, rows):
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO articles
          (stock, week_start, article_ts, summary, age_in_tdays, decay_weight,
           finbert_label, finbert_conf, finbert_score,
           fintone_label, fintone_conf, fintone_score,
           twitter_label, twitter_conf, twitter_score,
           final_score, final_label, agreement)
        VALUES
          (:stock, :week_start, :article_ts, :summary, :age_in_tdays, :decay_weight,
           :finbert_label, :finbert_conf, :finbert_score,
           :fintone_label, :fintone_conf, :fintone_score,
           :twitter_label, :twitter_conf, :twitter_score,
           :final_score, :final_label, :agreement)
        """,
        rows,
    )
    conn.commit()

def save_weekly_summary(conn, stock, week_start, rows, total_fetched=0):
    if not rows:
        return
    scores  = [r["final_score"] for r in rows]
    weights = [r["decay_weight"] for r in rows]
    avg     = round(sum(s * w for s, w in zip(scores, weights)) / sum(weights), 3)
    overall = "Bullish" if avg >= 0.1 else "Bearish" if avg <= -0.1 else "Neutral"
    conn.execute(
        """
        INSERT OR REPLACE INTO weekly_summary
          (stock, week_start, avg_score, min_score, max_score,
           total_articles_scored, total_articles_fetched, agreements, overall)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (stock, week_start, avg, round(min(scores), 3), round(max(scores), 3),
         len(rows), total_fetched, sum(1 for r in rows if r["agreement"]), overall),
    )
    conn.commit()

def mark_progress(conn, stock, week_start, status):
    conn.execute(
        "INSERT OR REPLACE INTO progress (stock, week_start, status, fetched_at) VALUES (?, ?, ?, ?)",
        (stock, week_start, status, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()

# ─────────────────────────────────────────────────────────────────────
# Date windows
# ─────────────────────────────────────────────────────────────────────
def week_windows(months_back=1):
    """
    Yields (week_start_str, week_end_str) starting with the MOST RECENT
    week (today) and walking backward `months_back` months.
    """
    today      = datetime.today().date()
    cutoff     = today - timedelta(days=30 * months_back)
    window_end = today
    while window_end >= cutoff:
        window_start = max(window_end - timedelta(days=6), cutoff)
        yield window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
        window_end = window_start - timedelta(days=1)

# ─────────────────────────────────────────────────────────────────────
# Per-stock-per-week fetcher
# ─────────────────────────────────────────────────────────────────────
def process_week(client, stock, week_start, week_end, db_path):
    # Each call opens its own connection — SQLite connections cannot be
    # shared across threads, and process_week runs inside worker threads.
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        if already_done(conn, stock, week_start):
            return stock, week_start, -1, 0

        wait_for_rate_limit()
        news = client.company_news(stock, _from=week_start, to=week_end)

        if not news:
            mark_progress(conn, stock, week_start, "empty")
            return stock, week_start, 0, 0

        # CLT sampling: if the week has more than CLT_SAMPLE_SIZE articles,
        # draw a random sample of exactly CLT_SAMPLE_SIZE. This keeps model
        # inference fast while the sample mean still converges to the true
        # population mean (Central Limit Theorem, n=50 threshold).
        total_articles = len(news)
        if total_articles > CLT_SAMPLE_SIZE:
            news = random.sample(news, CLT_SAMPLE_SIZE)

        ref_ts = datetime.strptime(week_end, "%Y-%m-%d")
        rows   = []

        for article in news:
            text = article.get("summary") or article.get("headline") or ""
            if not text:
                continue
            result = ensemble_finance(text)
            if result is None:
                continue
            article_ts   = article.get("datetime", ref_ts.timestamp())
            age_in_tdays = trading_days_elapsed(article_ts, reference_ts=ref_ts)
            decay_weight = round(math.exp(-LAMBDA * age_in_tdays), 4)
            rows.append({
                "stock":        stock,
                "week_start":   week_start,
                "article_ts":   article_ts,
                "summary":      text[:120],
                "age_in_tdays": round(age_in_tdays, 2),
                "decay_weight": decay_weight,
                **result,
            })

        save_articles(conn, rows)
        save_weekly_summary(conn, stock, week_start, rows, total_fetched=total_articles)
        mark_progress(conn, stock, week_start, "done")
        # Return (scored, total) so the print line can show e.g. "18/47 articles"
        return stock, week_start, len(rows), total_articles

    except Exception as e:
        mark_progress(conn, stock, week_start, "error")
        print(f"  ❌ {stock} {week_start}: {e}")
        return stock, week_start, 0, 0

    finally:
        conn.close()

# ─────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────
if not FINNHUB_API_KEY:
    raise SystemExit(
        "FINNHUB_API_KEY is not set. Set it as an environment variable, e.g.\n"
        "  PowerShell:  $env:FINNHUB_API_KEY = \"your-key-here\"\n"
        "  bash:        export FINNHUB_API_KEY=your-key-here"
    )

windows    = list(week_windows(months_back=MONTHS_BACK))
total_jobs = len(STOCKS) * len(windows)
skipped = done = errors = 0

print(f"Backfill plan: {len(STOCKS)} stocks × {len(windows)} weeks = {total_jobs} jobs")
print(f"Database: {DB_FILE}\n")

os.makedirs(os.path.dirname(DB_FILE) or ".", exist_ok=True)
conn = init_db(DB_FILE)   # creates tables once, in the main thread
conn.close()               # each worker thread opens its own connection
client = finnhub.Client(api_key=FINNHUB_API_KEY)

for week_start, week_end in windows:
    print(f"\n── Week {week_start} → {week_end} ──")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_week, client, stock, week_start, week_end, DB_FILE): stock
            for stock in STOCKS
        }
        for future in as_completed(futures):
            stock_out, ws_out, count, total = future.result()
            if count == -1:
                skipped += 1
            elif count == 0:
                errors += 1
                print(f"  ⚠️  {stock_out} ({ws_out}): no articles")
            else:
                done += 1
                print(f"  ✅ {stock_out} ({ws_out}): scored {count}/{total} articles")

print(f"\n{'═'*50}")
print(f"Backfill complete.")
print(f"  Done:    {done}")
print(f"  Skipped: {skipped}  (already in DB)")
print(f"  Empty/Error: {errors}")
print(f"  Database saved to: {DB_FILE}")