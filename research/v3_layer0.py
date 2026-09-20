# ═══════════════════════════════════════════════════════════════════════
#  v3 layer 0 — what is actually new in each earnings release.
#
#  Design: research/design_sentiment_v3.md. Companies repeat most of each
#  release every quarter: contacts, company description, safe harbor,
#  table labels, the sentence that says "revenue was $X, up Y%". Only
#  what changed can carry news the market has not seen in the numbers.
#
#  The classification itself lives in trader/text/novelty.py, so the daily
#  run labels a filing exactly as the model was fitted. This script is the
#  bulk pass over every stored release. Each release is cut into units —
#  sentences of paragraphs, and table rows — and every unit is compared
#  with the same company's previous PRIOR_K releases:
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
import sqlite3
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.text.novelty import (CLASSES, EDITED, PRIOR_K, classify_company,  # noqa: E402,F401
                                 max_similarity, sentences, template_key, units, word_pairs)
from trader.text.parse import blocks  # noqa: E402

EDGAR_DB = ROOT / "data" / "edgar.db"
V3_DB    = ROOT / "data" / "v3.db"
# releases in -> layer 0 out. The S&P 400/600 exam set has its own files.
SETS     = {"sp500":    (EDGAR_DB, V3_DB),
            "midsmall": (ROOT / "data" / "edgar_midsmall.db", ROOT / "data" / "v3_midsmall.db")}

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
