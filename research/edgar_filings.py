# ═══════════════════════════════════════════════════════════════════════
#  Download the 8-K earnings releases behind every point-in-time earnings
#  event — the text source for the v2 sentiment signal.
#
#  Design and pre-registration: research/design_sentiment_v2.md
#
#  Every US company files an 8-K carrying Item 2.02 ("Results of
#  Operations") when it reports, with the press release attached as an
#  EX-99 exhibit. Unlike news this is complete, timestamped to the second,
#  attributed to exactly one company, and free.
#
#  Three stages, each resumable:
#    1. per company, list its 8-Ks carrying Item 2.02 (1-2 requests each)
#    2. per earnings event, fetch the matching press release as flat text
#       (the v2 representation, kept so v2 stays reproducible)
#    3. the same release as compact HTML, structure intact (v3; see
#       research/release_text.py). Stage 2's flat text had lost the
#       paragraphs and tables, which cost a full re-download to recover —
#       so the structure is what gets cached now.
#
#  The SEC asks automated clients to identify themselves. The contact is
#  read from `git config user.email` at run time (override with
#  SEC_CONTACT) and is never written into the repo.
#
#    python research/edgar_filings.py              full run, resumable
#    python research/edgar_filings.py --limit 5    a few companies, to try it
#    python research/edgar_filings.py --html-only  stage 3 alone
#    python research/edgar_filings.py --set midsmall   the S&P 400/600 exam set
# ═══════════════════════════════════════════════════════════════════════

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.data import RESEARCH_DB, load_earnings, load_universe  # noqa: E402
from trader.events import point_in_time  # noqa: E402
from trader.market_calendar import ET  # noqa: E402
from release_text import compact_html, exhibit  # noqa: E402

# The SEC's acceptanceDateTime really is UTC despite the ambiguity around
# its 'Z' suffix: across a sample the hours cluster at 11-13 and 20-21,
# which as UTC are the 06:00-08:00 pre-market and 16:00-17:00 post-close
# release windows. Read as Eastern they would mean midday and 9pm
# announcements, which is not how companies report. Note the 8-K is filed
# shortly AFTER the press release goes out, so this timestamp is a
# conservative marker of when the text was publicly available.
DB_FILE     = ROOT / "data" / "edgar.db"
# Which earnings events to cover: source database -> filings database. The
# S&P 400/600 set is the v3 exam set and lives in its own files, so no
# script aimed at the development data can read it by accident.
SETS        = {"sp500":    (RESEARCH_DB, DB_FILE),
               "midsmall": (ROOT / "data" / "oos_midsmall.db", ROOT / "data" / "edgar_midsmall.db")}
START_DATE  = "2010-01-01"
RATE        = 8.0        # requests/second; the SEC's published ceiling is 10
WORKERS     = 5          # each filing costs two round trips of ~400ms, so one
                         # thread idles on latency at ~1 filing/second while the
                         # rate limit would allow four. Workers share the single
                         # global limiter, so the SEC's ceiling still holds.
MATCH_DAYS  = 2          # an 8-K this many days either side of the announcement
MIN_CHARS   = 400        # shorter than this is a stub, not a press release

SCHEMA = """
    CREATE TABLE IF NOT EXISTS filings (
        accession TEXT PRIMARY KEY, cik TEXT NOT NULL, symbol TEXT,
        filed_date TEXT, accepted_at TEXT, items TEXT,
        doc_name TEXT, n_chars INTEGER, body BLOB);
    CREATE TABLE IF NOT EXISTS company_8ks (
        accession TEXT PRIMARY KEY, cik TEXT NOT NULL,
        filed_date TEXT, accepted_at TEXT, items TEXT);
    CREATE TABLE IF NOT EXISTS company_progress (
        cik TEXT PRIMARY KEY, status TEXT, n_8ks INTEGER, updated_at TEXT);
    CREATE TABLE IF NOT EXISTS event_filings (
        event_key TEXT PRIMARY KEY, cik TEXT, symbol TEXT, ann_date TEXT,
        accession TEXT, status TEXT);
    CREATE INDEX IF NOT EXISTS ix_8k_cik ON company_8ks(cik);
    CREATE TABLE IF NOT EXISTS release_html (
        accession TEXT PRIMARY KEY, doc_type TEXT, filename TEXT,
        n_bytes INTEGER, html BLOB);
"""


# ── HTTP with the SEC's rate limit ────────────────────────────────────
class Limiter:
    def __init__(self, per_second):
        self.gap = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                time.sleep(self.next_at - now)
                now = time.monotonic()
            self.next_at = max(now, self.next_at) + self.gap


LIMIT = Limiter(RATE)
SESSION = requests.Session()


def contact():
    who = os.environ.get("SEC_CONTACT")
    if not who:
        who = subprocess.run(["git", "config", "user.email"], capture_output=True,
                             text=True).stdout.strip()
    if not who:
        raise SystemExit("Set SEC_CONTACT to a contact email — the SEC requires one.")
    return f"PEAD research {who}"


def get(url, tries=4):
    for attempt in range(tries):
        LIMIT.wait()
        try:
            r = SESSION.get(url, headers={"User-Agent": SESSION.headers["User-Agent"]}, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return None


# ── Stage 1: which 8-Ks carry earnings ────────────────────────────────
def list_earnings_8ks(cik):
    """Every 8-K with Item 2.02 since START_DATE. The API's `recent` block
    caps at 1,000 filings, so older ones come from its shard files."""
    r = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if r is None:
        return None
    payload = r.json()
    blocks = [payload["filings"]["recent"]]
    for shard in payload["filings"].get("files", []):
        s = get(f"https://data.sec.gov/submissions/{shard['name']}")
        if s is not None:
            blocks.append(s.json())

    out = []
    for block in blocks:
        forms = block.get("form", [])
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            filed = block["filingDate"][i]
            if filed < START_DATE:
                continue
            items = block.get("items", [""] * len(forms))[i] or ""
            if "2.02" not in items:
                continue
            out.append((block["accessionNumber"][i], filed,
                        block.get("acceptanceDateTime", [""] * len(forms))[i], items))
    return out


# ── Stage 2: pull the press release out of a filing ───────────────────
EXHIBIT = re.compile(r"^EX-99", re.I)


SUBMISSION_DOC = re.compile(r"<TYPE>(EX-99[.\d]*)\s*\n(.*?)</DOCUMENT>", re.S | re.I)


def press_release(cik, accession):
    """The EX-99 exhibit, as plain text.

    Fetched as the single combined submission file rather than index page
    plus exhibit. Throughput here is capped by the SEC's request-per-second
    limit, not by bandwidth, so one larger request beats two smaller ones:
    it doubles the filings-per-second ceiling for ~36% more bytes. Falls
    back to the two-request path when the combined file has no EX-99
    marker."""
    nodash = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}"

    full = get(f"{base}/{accession}.txt")
    if full is not None:
        m = SUBMISSION_DOC.search(full.text)
        if m:
            return m.group(1), to_text(m.group(2))

    page = get(f"{base}/{accession}-index.htm")
    if page is None:
        return None, None
    try:
        tables = pd.read_html(StringIO(page.text))
    except ValueError:
        return None, None

    doc = None
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        if "type" not in cols or "document" not in cols:
            continue
        hit = table[table[cols["type"]].astype(str).str.match(EXHIBIT, na=False)]
        if len(hit):
            doc = str(hit.iloc[0][cols["document"]]).split()[0]
            break
    if not doc:
        return None, None

    body = get(f"{base}/{doc}")
    if body is None:
        return doc, None
    return doc, to_text(body.text)


def release_html(cik, accession):
    """The press release as compact HTML: (type, filename, html), or None.

    The first EX-99 exhibit, as stage 2 took, so v2 and v3 read one text.
    Index page first here, unlike stage 2: the combined submission bundles
    every image and XBRL file and is built on demand for older filings —
    18s for one where index + exhibit took 1.3s — so two small requests
    beat one large one. The combined file is the fallback."""
    nodash = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}"
    found = None
    page = get(f"{base}/{accession}-index.htm")
    try:
        tables = pd.read_html(StringIO(page.text)) if page is not None else []
    except ValueError:
        tables = []
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        if "type" not in cols or "document" not in cols:
            continue
        hit = table[table[cols["type"]].astype(str).str.match(EXHIBIT, na=False)]
        if len(hit):
            name = str(hit.iloc[0][cols["document"]]).split()[0]
            doc = get(f"{base}/{name}")
            if doc is not None:
                found = (str(hit.iloc[0][cols["type"]]).upper(), name, doc.text)
            break
    if found is None:
        full = get(f"{base}/{accession}.txt")
        found = exhibit(full.text) if full is not None else None
    if found is None:
        return None
    kind, name, raw = found
    return kind, name, compact_html(raw)


def to_text(html):
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
            .replace("&#8217;", "'").replace("&#8212;", "-").replace("&#39;", "'"))
    text = re.sub(r"&#\d+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Work queue ────────────────────────────────────────────────────────
def build_queue(conn, source=RESEARCH_DB):
    """One row per point-in-time earnings event that still needs a filing."""
    research = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    earn = load_earnings(research)
    universe = load_universe(research)
    ids = pd.read_sql_query(
        "SELECT symbol, firm FROM universe_ids WHERE firm IS NOT NULL", research)
    research.close()

    ann = earn.announced_at.dt.tz_convert(ET)
    earn = earn.assign(ann_date=ann.dt.date.astype(str))
    earn = earn[earn.ann_date >= START_DATE]
    _, pit = point_in_time(earn.symbol, pd.to_datetime(earn.ann_date), universe)
    earn = earn[pd.Series(pit, index=earn.index)].copy()

    cik_of = (ids[~ids.firm.str.startswith("SYM:")]
              .drop_duplicates("symbol").set_index("symbol").firm.to_dict())
    earn["cik"] = earn.symbol.map(cik_of)
    earn = earn.dropna(subset=["cik"])
    earn["event_key"] = earn.symbol + "|" + earn.ann_date

    rows = [(r.event_key, r.cik, r.symbol, r.ann_date, None, "pending")
            for r in earn.drop_duplicates("event_key").itertuples()]
    conn.executemany("INSERT OR IGNORE INTO event_filings VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def match_events(conn):
    """Pair each event with the 8-K filed nearest its announcement, within
    MATCH_DAYS. A nearest-key join expresses this far more clearly than a
    correlated subquery."""
    events = pd.read_sql_query(
        "SELECT event_key, cik, ann_date FROM event_filings WHERE accession IS NULL", conn)
    filings = pd.read_sql_query("SELECT accession, cik, filed_date FROM company_8ks", conn)
    if events.empty or filings.empty:
        return
    events["on"] = pd.to_datetime(events.ann_date)
    filings["on"] = pd.to_datetime(filings.filed_date)
    paired = pd.merge_asof(
        events.sort_values("on"), filings.sort_values("on"),
        on="on", by="cik", direction="nearest", tolerance=pd.Timedelta(days=MATCH_DAYS))

    hits = paired.dropna(subset=["accession"])
    conn.executemany("UPDATE event_filings SET accession=?, status='matched' WHERE event_key=?",
                     zip(hits.accession, hits.event_key))
    misses = paired[paired.accession.isna()]
    conn.executemany("UPDATE event_filings SET status='no_filing' WHERE event_key=?",
                     ((k,) for k in misses.event_key))
    conn.commit()


def fetch_structured(conn):
    """Stage 3: compact HTML for every release stage 2 cached as text."""
    todo = conn.execute(
        """SELECT accession, cik FROM filings WHERE body IS NOT NULL
             AND accession NOT IN (SELECT accession FROM release_html)""").fetchall()
    print(f"\nStage 3 — structured HTML for {len(todo):,} press releases")

    def fetch_html(row):
        try:
            return row[0], release_html(row[1], row[0])
        except Exception as e:                   # one malformed document must not end the run
            print(f"   {row[0]}: {type(e).__name__} {str(e)[:80]}", flush=True)
            return row[0], None

    # Latency-bound at two requests per release, so more threads than stage 2;
    # the shared limiter still holds the SEC's rate ceiling.
    got = 0
    with ThreadPoolExecutor(max_workers=2 * WORKERS) as pool:
        for i, (accession, res) in enumerate(pool.map(fetch_html, todo), 1):
            kind, name, html = res if res else (None, None, None)
            conn.execute("INSERT OR REPLACE INTO release_html VALUES (?,?,?,?,?)",
                         (accession, kind, name, len(html) if html else 0,
                          zlib.compress(html.encode("utf-8"), 9) if html else None))
            got += html is not None
            if i % 250 == 0 or i == len(todo):
                conn.commit()
                print(f"   {i}/{len(todo)}  ({got} stored)", flush=True)
    conn.commit()


def fetch_structured_events(conn):
    """Stage 2 for sets downloaded after v2: the matched releases straight
    to compact HTML, with no flattened text. `filings` keeps the metadata
    (dates, acceptance time) and body stays NULL."""
    todo = conn.execute(
        """SELECT DISTINCT e.accession, e.cik, e.symbol FROM event_filings e
           WHERE e.accession IS NOT NULL
             AND e.accession NOT IN (SELECT accession FROM release_html)""").fetchall()
    print(f"\nStage 2 — structured HTML for {len(todo):,} press releases")

    def fetch_html(row):
        try:
            return row, release_html(row[1], row[0])
        except Exception as e:                   # one malformed document must not end the run
            print(f"   {row[0]}: {type(e).__name__} {str(e)[:80]}", flush=True)
            return row, None

    got = 0
    with ThreadPoolExecutor(max_workers=2 * WORKERS) as pool:
        for i, ((accession, cik, symbol), res) in enumerate(pool.map(fetch_html, todo), 1):
            kind, name, html = res if res else (None, None, None)
            meta = conn.execute(
                "SELECT filed_date, accepted_at, items FROM company_8ks WHERE accession=?",
                (accession,)).fetchone() or ("", "", "")
            conn.execute("INSERT OR REPLACE INTO filings VALUES (?,?,?,?,?,?,?,?,?)",
                         (accession, cik, symbol, meta[0], meta[1], meta[2], name,
                          len(html) if html else 0, None))
            conn.execute("INSERT OR REPLACE INTO release_html VALUES (?,?,?,?,?)",
                         (accession, kind, name, len(html) if html else 0,
                          zlib.compress(html.encode("utf-8"), 9) if html else None))
            got += html is not None
            if i % 250 == 0 or i == len(todo):
                conn.commit()
                print(f"   {i}/{len(todo)}  ({got} stored)", flush=True)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only this many companies — for a trial run")
    ap.add_argument("--set", choices=sorted(SETS), default="sp500",
                    help="whose earnings events: sp500 (v2/v3 development) or midsmall (v3 exam)")
    ap.add_argument("--html-only", action="store_true",
                    help="only stage 3; leave the v2 tables exactly as they are")
    args = ap.parse_args()

    SESSION.headers.update({"User-Agent": contact(), "Accept-Encoding": "gzip, deflate"})
    print(f"Identifying to the SEC as: {SESSION.headers['User-Agent'].split('@')[0]}@…")

    source, target = SETS[args.set]
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    if args.html_only:
        fetch_structured(conn)
        conn.close()
        return

    n = build_queue(conn, source)
    total = conn.execute("SELECT COUNT(*) FROM event_filings").fetchone()[0]
    print(f"Work queue: {total:,} earnings events ({n:,} inserted this run)")

    ciks = [r[0] for r in conn.execute(
        """SELECT DISTINCT cik FROM event_filings WHERE cik NOT IN
           (SELECT cik FROM company_progress WHERE status IN ('done','empty')) ORDER BY cik""")]
    if args.limit:
        ciks = ciks[:args.limit]
    print(f"\nStage 1 — listing 8-Ks for {len(ciks):,} companies")
    for i, cik in enumerate(ciks, 1):
        found = list_earnings_8ks(cik)
        if found is None:
            status, count = "error", 0
        else:
            conn.executemany("INSERT OR IGNORE INTO company_8ks VALUES (?,?,?,?,?)",
                             [(a, cik, d, acc, it) for a, d, acc, it in found])
            status, count = ("done" if found else "empty"), len(found)
        conn.execute("INSERT OR REPLACE INTO company_progress VALUES (?,?,?,?)",
                     (cik, status, count, datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        if i % 25 == 0 or i == len(ciks):
            print(f"   {i}/{len(ciks)} companies", flush=True)

    # Match each event to the 8-K filed nearest its announcement date.
    print("\nMatching events to filings …")
    match_events(conn)
    matched = conn.execute("SELECT COUNT(*) FROM event_filings WHERE accession IS NOT NULL").fetchone()[0]
    print(f"   {matched:,} of {total:,} events matched to an 8-K "
          f"({matched / max(total, 1) * 100:.1f}%)")

    if args.set != "sp500":
        fetch_structured_events(conn)
        n_ok = conn.execute("SELECT COUNT(*) FROM release_html WHERE html IS NOT NULL").fetchone()[0]
        print(f"\n{n_ok:,} press releases stored as structured HTML in {target.name}")
        conn.close()
        return

    todo = conn.execute(
        """SELECT DISTINCT e.accession, e.cik, e.symbol FROM event_filings e
           WHERE e.accession IS NOT NULL
             AND e.accession NOT IN (SELECT accession FROM filings)""").fetchall()
    print(f"\nStage 2 — fetching {len(todo):,} press releases")
    ok = 0

    def fetch(row):
        accession, cik, symbol = row
        doc, text = press_release(cik, accession)
        return accession, cik, symbol, doc, text

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (accession, cik, symbol, doc, text) in enumerate(pool.map(fetch, todo), 1):
            meta = conn.execute(
                "SELECT filed_date, accepted_at, items FROM company_8ks WHERE accession=?",
                (accession,)).fetchone() or ("", "", "")
            good = text is not None and len(text) >= MIN_CHARS
            conn.execute("INSERT OR REPLACE INTO filings VALUES (?,?,?,?,?,?,?,?,?)",
                         (accession, cik, symbol, meta[0], meta[1], meta[2], doc,
                          len(text) if text else 0,
                          zlib.compress(text.encode("utf-8"), 6) if good else None))
            ok += good
            if i % 250 == 0 or i == len(todo):
                conn.commit()
                print(f"   {i}/{len(todo)}  ({ok} with usable text)", flush=True)
    conn.commit()

    fetch_structured(conn)

    have = conn.execute("SELECT COUNT(*) FROM filings WHERE body IS NOT NULL").fetchone()[0]
    size = conn.execute("SELECT SUM(LENGTH(body)) FROM filings").fetchone()[0] or 0
    chars = conn.execute("SELECT AVG(n_chars) FROM filings WHERE body IS NOT NULL").fetchone()[0] or 0
    print(f"\n{have:,} press releases cached, {size / 1e6:.0f} MB compressed, "
          f"{chars:,.0f} characters each on average")
    conn.close()


def read_text(blob):
    """Decompress a cached filing body, or a release_html.html blob."""
    return zlib.decompress(blob).decode("utf-8") if blob else None


if __name__ == "__main__":
    main()
