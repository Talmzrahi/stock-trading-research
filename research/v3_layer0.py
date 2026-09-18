# ═══════════════════════════════════════════════════════════════════════
#  v3 layer 0 — what is actually new in each earnings release.
#
#  Design: research/design_sentiment_v3.md. Companies repeat most of each
#  release every quarter: contacts, company description, safe harbor,
#  table labels, the sentence that says "revenue was $X, up Y%". Only
#  what changed can carry news the market has not seen in the numbers.
#
#  Each release is cut into units — sentences of paragraphs, and table
#  rows — and every unit is compared with the same company's previous
#  PRIOR_K releases:
#
#    boilerplate  the identical text appeared before
#    template     the same words; only numbers, dates or quarter names
#                 changed ("revenue was $6.76 billion, up 13%") — the
#                 news here is in the numbers, not the language
#    edited       mostly the same wording (word-pair overlap >= EDITED)
#    new          nothing close in the company's recent releases
#
#  No returns and no labels are touched. Layer 0 is text bookkeeping, so
#  it can run on any release — including the S&P 400/600 exam set — at
#  no cost to any future test.
#
#    python research/v3_layer0.py              every release with HTML
#    python research/v3_layer0.py --sample 40  a few companies, printed
#    python research/v3_layer0.py --set midsmall   the S&P 400/600 exam set
# ═══════════════════════════════════════════════════════════════════════

import argparse
import json
import re
import sqlite3
import sys
import zlib
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_text import blocks  # noqa: E402

EDGAR_DB = ROOT / "data" / "edgar.db"
V3_DB    = ROOT / "data" / "v3.db"
# releases in -> layer 0 out. The S&P 400/600 exam set has its own files.
SETS     = {"sp500":    (EDGAR_DB, V3_DB),
            "midsmall": (ROOT / "data" / "edgar_midsmall.db", ROOT / "data" / "v3_midsmall.db")}
PRIOR_K  = 4        # compare with the previous year of releases
EDITED   = 0.45     # word-pair Jaccard for "edited": the low point of the sentence
                    # similarity histogram (design doc, "Layer 0")
HASH_DIM = 1 << 20

CLASSES = ("boilerplate", "template", "edited", "new")

# ── cutting text into units ────────────────────────────────────────────

ABBREV = re.compile(r"(?:\b(?:inc|corp|co|ltd|llc|no|vs|approx|jan|feb|mar|apr|jun|jul|aug|"
                    r"sep|sept|oct|nov|dec|mr|mrs|ms|dr|st|calif|mass|fla|ill|wash|ariz|colo|"
                    r"conn|minn|mich|penn|tenn|va|wis|e\.g|i\.e|etc|u\.s|u\.k|n\.a)|\b[a-z])\.$",
                    re.I)
BOUNDARY = re.compile(r"(?<=[.!?])[\"”’)]*\s+(?=[\"“(]?[A-Z0-9$])")


def sentences(text):
    """Split a paragraph at sentence ends, but not after "Inc." or "U.S."
    or a middle initial, which would cut one sentence into fragments."""
    parts, start = [], 0
    for m in BOUNDARY.finditer(text):
        head = text[start:m.start()].rstrip("\"”’)")
        if ABBREV.search(head[-12:]):
            continue
        parts.append(text[start:m.start()].strip())
        start = m.end()
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def units(release_blocks):
    """(block index, kind, text) for every sentence and table row.
    kind: "s" sentence, "h" heading-like fragment, "row" table row."""
    out = []
    for i, b in enumerate(release_blocks):
        if b["k"] == "row":
            out.append((i, "row", " | ".join(b["c"])))
            continue
        for s in sentences(b["t"]):
            short = len(s.split()) < 6 and not s.endswith((".", "!", "?"))
            out.append((i, "h" if short else "s", s))
    return out


# ── what counts as "the same" ──────────────────────────────────────────

MONTH = re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
                   r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?")
PERIOD = re.compile(r"\b(?:first|second|third|fourth|q[1-4]|[1-4](?:st|nd|rd|th))\b")
NUMBER = re.compile(r"[$€£]?\(?[-–—]?\d[\d,]*(?:\.\d+)?\)?%?")
WORD = re.compile(r"[a-z#<>]+")


def template_key(text):
    """The words with every number, month and quarter name masked, so a
    sentence that only updated its figures matches last quarter's."""
    t = text.lower()
    t = MONTH.sub(" <m> ", t)
    t = PERIOD.sub(" <q> ", t)
    t = NUMBER.sub(" # ", t)
    return " ".join(WORD.findall(t))


def exact_key(text):
    return re.sub(r"\s+", " ", text.strip().lower())


PLACEHOLDER = {"^", "$", "#", "<m>", "<q>"}


def word_pairs(key):
    """Hashed word bigrams (with edge markers) — the unit of similarity.
    Pairs of placeholders only ("# #", "^ #") are skipped: after masking,
    every table row shares them, which says nothing and turned each
    release-vs-history comparison into all rows against all rows."""
    w = ["^"] + key.split() + ["$"]
    return {zlib.crc32(f"{a} {b}".encode()) % HASH_DIM for a, b in zip(w, w[1:])
            if not (a in PLACEHOLDER and b in PLACEHOLDER)}


def _matrix(pair_sets):
    rows = np.repeat(np.arange(len(pair_sets)), [len(s) for s in pair_sets])
    cols = np.fromiter((c for s in pair_sets for c in s), dtype=np.int64, count=len(rows))
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                             shape=(len(pair_sets), HASH_DIM))


def max_similarity(current, prior):
    """For each current unit, its highest word-pair Jaccard with any prior
    unit. Sparse matrix product, so a release against a year of history is
    one multiplication rather than ~10^5 pairwise comparisons."""
    if not current or not prior:
        return np.zeros(len(current))
    A, B = _matrix(current), _matrix(prior)
    inter = (A @ B.T).tocsr()
    na = np.asarray(A.sum(axis=1)).ravel()
    nb = np.asarray(B.sum(axis=1)).ravel()
    best = np.zeros(len(current))
    for i in range(inter.shape[0]):
        lo, hi = inter.indptr[i], inter.indptr[i + 1]
        if lo == hi:
            continue
        j, x = inter.indices[lo:hi], inter.data[lo:hi]
        best[i] = float(np.max(x / (na[i] + nb[j] - x)))
    return best


# ── one company's releases, in order ───────────────────────────────────

def classify_company(releases, edited=EDITED, prior_k=PRIOR_K):
    """releases: [(accession, filed_date, blocks)] for one company, any
    order. Returns one record per release. Only strictly earlier releases
    count as history."""
    history = deque(maxlen=prior_k)
    out = []
    for n_prior, (accession, filed, rb) in enumerate(sorted(releases, key=lambda r: r[1])):
        us = units(rb)
        exact = [exact_key(t) for _, _, t in us]
        keys = [template_key(t) for _, _, t in us]
        pairs = [word_pairs(k) for k in keys]

        seen_exact = set().union(*(h["exact"] for h in history)) if history else set()
        seen_keys = set().union(*(h["keys"] for h in history)) if history else set()
        prior_pairs = [p for h in history for p in h["pairs"]]
        sim = max_similarity(pairs, prior_pairs)

        rows, in_doc, counts = [], set(), dict.fromkeys(CLASSES, 0)
        for (blk, kind, text), e, k, s in zip(us, exact, keys, sim):
            if e in seen_exact:
                cls, s = "boilerplate", 1.0
            elif k in seen_keys:
                cls, s = "template", 1.0
            elif s >= edited:
                cls = "edited"
            else:
                cls = "new"
            counts[cls] += 1
            rows.append([blk, kind, cls, round(float(s), 3), k in in_doc, text])
            in_doc.add(k)

        out.append({"accession": accession, "filed_date": filed, "n_prior": n_prior,
                    "n_units": len(us), **{f"n_{c}": counts[c] for c in CLASSES},
                    "units": rows})
        history.append({"exact": set(exact), "keys": set(keys), "pairs": pairs})
    return out


# ── running it ─────────────────────────────────────────────────────────

SCHEMA = """
    CREATE TABLE IF NOT EXISTS layer0 (
        accession TEXT PRIMARY KEY, cik TEXT, filed_date TEXT, n_prior INTEGER,
        n_units INTEGER, n_boilerplate INTEGER, n_template INTEGER,
        n_edited INTEGER, n_new INTEGER, units BLOB);
"""


def companies(edgar_db=EDGAR_DB):
    """{cik: [(accession, filed_date)]} for every release with HTML. One
    pass over the small columns; filings has no index on cik, and filtering
    it per company re-read the whole blob-heavy table 500 times."""
    conn = sqlite3.connect(f"file:{edgar_db}?mode=ro", uri=True, timeout=60)
    have = {a for (a,) in conn.execute("SELECT accession FROM release_html WHERE html IS NOT NULL")}
    out = {}
    for accession, cik, filed in conn.execute("SELECT accession, cik, filed_date FROM filings"):
        if accession in have:
            out.setdefault(cik, []).append((accession, filed))
    conn.close()
    return out


def load_releases(listing, edgar_db=EDGAR_DB):
    """[(accession, filed_date, blocks)] for one company's listing."""
    conn = sqlite3.connect(f"file:{edgar_db}?mode=ro", uri=True, timeout=60)
    out = []
    for accession, filed in listing:
        (html,) = conn.execute("SELECT html FROM release_html WHERE accession = ?",
                               (accession,)).fetchone()
        out.append((accession, filed, blocks(zlib.decompress(html).decode("utf-8"))))
    conn.close()
    return out


def run_company(item):
    cik, listing, edgar_db = item          # paths passed in: Windows workers
    try:                                    # re-import the module fresh
        return cik, classify_company(load_releases(listing, edgar_db)), None
    except Exception as e:                       # report, never lose the whole run
        return cik, [], f"{type(e).__name__}: {str(e)[:100]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="only this many companies, printed, not saved")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--set", choices=sorted(SETS), default="sp500")
    args = ap.parse_args()

    edgar_db, v3_db = SETS[args.set]
    listing = companies(edgar_db)
    ciks = sorted(listing)
    if args.sample:
        ciks = sorted(np.random.default_rng(0).choice(ciks, size=min(args.sample, len(ciks)),
                                                      replace=False))
    print(f"Layer 0 over {len(ciks):,} companies …", flush=True)

    out = None
    if not args.sample:
        out = sqlite3.connect(v3_db, timeout=60)
        out.execute("PRAGMA journal_mode=WAL")
        out.executescript(SCHEMA)

    totals, done, errors = dict.fromkeys(CLASSES, 0), 0, 0
    with Pool(args.workers) as pool:
        work = [(cik, listing[cik], edgar_db) for cik in ciks]
        for i, (cik, recs, err) in enumerate(pool.imap_unordered(run_company, work), 1):
            if err:
                errors += 1
                print(f"   {cik}: {err}", flush=True)
            for r in recs:
                for c in CLASSES:
                    totals[c] += r[f"n_{c}"]
                if out is not None:
                    out.execute("INSERT OR REPLACE INTO layer0 VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (r["accession"], cik, r["filed_date"], r["n_prior"], r["n_units"],
                                 r["n_boilerplate"], r["n_template"], r["n_edited"], r["n_new"],
                                 zlib.compress(json.dumps(r["units"]).encode("utf-8"), 6)))
            done += len(recs)
            if out is not None and (i % 25 == 0 or i == len(ciks)):
                out.commit()
            if i % 50 == 0 or i == len(ciks):
                print(f"   {i}/{len(ciks)} companies, {done:,} releases", flush=True)

    n = sum(totals.values())
    print(f"\n{done:,} releases, {n:,} units, {errors} companies failed")
    for c in CLASSES:
        print(f"   {c:<12}{totals[c]:>10,}  {totals[c] / max(n, 1):6.1%}")
    if out is not None:
        out.close()


if __name__ == "__main__":
    main()
