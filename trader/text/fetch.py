# ═══════════════════════════════════════════════════════════════════════
#  Getting a press release out of EDGAR.
#
#  The same client the bulk download used (research/edgar_filings.py
#  imports it from here), plus `find_release` for the live case: a company
#  reported an hour ago, find that 8-K and read it.
#
#  The SEC asks automated clients to identify themselves and publishes a
#  10 requests/second ceiling; this stays under it and shares one limiter
#  across threads. The contact comes from SEC_CONTACT or `git config
#  user.email` at run time and is never written into the repo.
# ═══════════════════════════════════════════════════════════════════════

import os
import re
import subprocess
import threading
import time
from io import StringIO

import pandas as pd
import requests

from .parse import compact_html, exhibit

RATE       = 8.0              # requests/second; the SEC's published ceiling is 10
MATCH_DAYS = 2                # an 8-K this many days either side of the announcement
EXHIBIT    = re.compile(r"^EX-99", re.I)
START_DATE = "2010-01-01"


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


def identify():
    """Set the User-Agent once per process."""
    if "User-Agent" not in SESSION.headers or "PEAD" not in SESSION.headers.get("User-Agent", ""):
        SESSION.headers.update({"User-Agent": contact(), "Accept-Encoding": "gzip, deflate"})
    return SESSION.headers["User-Agent"]


def get(url, tries=4):
    identify()
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


def submissions(cik):
    """The company's filing index, most recent first. None if unreachable."""
    r = get(f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json")
    return r.json() if r is not None else None


def list_earnings_8ks(cik, start_date=START_DATE):
    """Every 8-K with Item 2.02 since start_date. The API's `recent` block
    caps at 1,000 filings, so older ones come from its shard files."""
    payload = submissions(cik)
    if payload is None:
        return None
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
            if filed < start_date:
                continue
            items = block.get("items", [""] * len(forms))[i] or ""
            if "2.02" not in items:
                continue
            out.append((block["accessionNumber"][i], filed,
                        block.get("acceptanceDateTime", [""] * len(forms))[i], items))
    return out


def find_release(cik, ann_date, days=MATCH_DAYS):
    """The earnings 8-K nearest an announcement date, from the company's
    recent filings only — one request, which is what a daily run wants.
    Returns {accession, filed_date, accepted_at, items} or None."""
    payload = submissions(cik)
    if payload is None:
        return None
    recent = payload["filings"]["recent"]
    target = pd.Timestamp(ann_date)
    best = None
    for i, form in enumerate(recent.get("form", [])):
        if form != "8-K" or "2.02" not in (recent.get("items", [""] * (i + 1))[i] or ""):
            continue
        filed = pd.Timestamp(recent["filingDate"][i])
        gap = abs((filed - target).days)
        if gap <= days and (best is None or gap < best[0]):
            best = (gap, {"accession": recent["accessionNumber"][i],
                          "filed_date": recent["filingDate"][i],
                          "accepted_at": recent.get("acceptanceDateTime", [""] * (i + 1))[i],
                          "items": recent.get("items", [""] * (i + 1))[i]})
    return best[1] if best else None


def release_html(cik, accession):
    """The press release as compact HTML: (type, filename, html), or None.

    The first EX-99 exhibit. Index page first: the combined submission
    file bundles every image and XBRL file and is built on demand for
    older filings — 18s for one where index + exhibit took 1.3s — so two
    small requests beat one large one. The combined file is the fallback."""
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
