# ═══════════════════════════════════════════════════════════════════════
#  Turn cached 8-K press releases into features.
#
#  Design: research/design_sentiment_v2.md. Raw text comes from
#  research/edgar_filings.py and is never re-downloaded — this step can be
#  re-run as often as the feature set changes.
#
#  Two features go to the model, as registered:
#    tone_z        Loughran-McDonald tone, against the company's own norm
#    guidance      direction and concreteness of forward-looking language
#
#  Other raw counts (uncertainty, litigious, modality, non-GAAP emphasis,
#  length) are STORED but not used: parsing is the expensive part, and the
#  design defers the decision on those. Using them would need the design
#  updated first.
#
#  POINT-IN-TIME DISCIPLINE. Every normalisation looks strictly backwards:
#    - against the company's own 12 most recent PRIOR filings (minimum 6),
#      using median and MAD so one odd quarter cannot manufacture an outlier
#    - against all companies' filings in the TRAILING 365 days
#  Normalising within a calendar quarter would use filings that had not
#  happened yet — the same look-ahead that broke the first decile cutoffs.
#
#    python research/edgar_features.py
# ═══════════════════════════════════════════════════════════════════════

import gzip
import json
import re
import sqlite3
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT      = Path(__file__).resolve().parent.parent
DB_FILE   = ROOT / "data" / "edgar.db"
WORDS     = ROOT / "data_archive" / "loughran_mcdonald.json.gz"
MIN_PRIOR = 6      # a company needs this many earlier filings to have a "norm"
PRIOR_N   = 12     # ... and only the most recent this many count, so a firm's
                   #     norm tracks what it has become, not what it was in 2010
WINSOR    = 5.0    # cap |z|; short histories otherwise produce absurd outliers
WINDOW    = 365    # days of trailing cross-section

TOKEN = re.compile(r"[a-z']+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Every release ends with a safe-harbor disclaimer, and measured across 300 of
# them it holds 80% of all Loughran-McDonald negative words. Scoring the whole
# document therefore measures how long a company's lawyers are, not how the
# quarter went: tone averages -0.32 over the full text against +0.26 over the
# narrative, and the two correlate only 0.45. Tone is computed on the narrative.
BOILERPLATE = re.compile(r"forward[- ]looking statements|safe harbor|"
                         r"cautionary (?:note|statement)|this (?:press )?release contains|"
                         r"risks and uncertainties", re.I)

GUIDANCE = re.compile(r"guidance|outlook|forecast|expects?|anticipates?|full[- ]year|"
                      r"fiscal\s+20\d\d|for the (?:full |fiscal )?year", re.I)
RAISES = re.compile(r"\b(rais(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|boost(?:s|ed)?|"
                    r"improv(?:e|es|ed)|upgrad(?:e|es|ed)|higher than|above (?:our )?(?:prior|previous))\b", re.I)
CUTS = re.compile(r"\b(lower(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|cut(?:s|ting)?|"
                  r"decreas(?:e|es|ed|ing)|weaken(?:s|ed|ing)?|below (?:our )?(?:prior|previous)|"
                  r"withdraw(?:s|n|ing)?|suspend(?:s|ed|ing)?)\b", re.I)
HOLDS = re.compile(r"\b(reaffirm(?:s|ed|ing)?|reiterat(?:e|es|ed|ing)|maintain(?:s|ed|ing)?|"
                   r"unchanged|confirms?)\b", re.I)

NUMBER  = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
DOLLAR  = re.compile(r"\$\s?\d")
PERCENT = re.compile(r"\d\s?(?:%|percent)")
RANGE   = re.compile(r"\$?\s?\d[\d,.]*\s*(?:to|-|–)\s*\$?\s?\d[\d,.]*")
NONGAAP = re.compile(r"non-?gaap|adjusted|excluding (?:certain|special)|core (?:earnings|eps)", re.I)


HASH_DIM = 1024


def hashed_vector(tokens):
    """A fixed-width bag of words, L2-normalised, so cosine similarity is a
    dot product. Hashing keeps 26,000 documents in ~100MB instead of holding
    a full vocabulary per document, and crc32 keeps it reproducible across
    runs where Python's own hash() would not."""
    v = np.zeros(HASH_DIM, dtype=np.float32)
    for t in tokens:
        v[zlib.crc32(t.encode("utf-8")) % HASH_DIM] += 1.0
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def language_change(df, vectors):
    """How much a company rewrote its release since last time.

    Cohen, Malloy & Nguyen (Lazy Prices, JF 2020) found that companies which
    quietly rewrite their filings go on to underperform the copy-pasters.
    Shown there on 10-K/10-Q filings, so applying it to 8-K press releases
    is an extension rather than a replication. Compared both with the prior
    release and with the same quarter a year earlier, since Q4 releases do
    not read like Q1 ones.
    """
    sim_prev = np.full(len(df), np.nan)
    sim_year = np.full(len(df), np.nan)
    for _, idx in df.groupby("cik").groups.items():
        rows = df.index.get_indexer(idx)
        order = rows[np.argsort(df.filed_date.values[rows])]
        for i, pos in enumerate(order):
            if i >= 1:
                sim_prev[pos] = float(vectors[pos] @ vectors[order[i - 1]])
            if i >= 4:
                sim_year[pos] = float(vectors[pos] @ vectors[order[i - 4]])
    return sim_prev, sim_year


def load_words():
    with gzip.open(WORDS, "rt", encoding="utf-8") as f:
        return {k: set(v) for k, v in json.load(f).items()}


def split_narrative(text):
    """The news, and the legal boilerplate that follows it."""
    m = BOILERPLATE.search(text)
    return (text[:m.start()], text[m.start():]) if m else (text, "")


def raw_features(text, lex):
    """Everything measurable from one press release, before any normalising."""
    narrative, boiler = split_narrative(text)
    tokens = TOKEN.findall(narrative.lower())
    n = max(len(tokens), 1)
    counts = {c: sum(t in words for t in tokens) for c, words in lex.items()}

    pos, neg = counts.get("positive", 0), counts.get("negative", 0)
    guide_sentences = [s for s in SENTENCE.split(narrative) if GUIDANCE.search(s)]
    guide_text = " ".join(guide_sentences)
    ups, downs = len(RAISES.findall(guide_text)), len(CUTS.findall(guide_text))

    return {
        "n_words": len(tokens),
        "boiler_share": len(TOKEN.findall(boiler.lower())) / max(n + len(TOKEN.findall(boiler.lower())), 1),
        # tone
        "tone": (pos - neg) / (pos + neg + 1),
        "neg_share": neg / n,
        "pos_share": pos / n,
        # guidance
        "guide_share": len(guide_sentences) / max(len(SENTENCE.split(text)), 1),
        "guide_dir": (ups - downs) / (ups + downs + 1),
        "guide_holds": len(HOLDS.findall(guide_text)),
        "guide_numeric": int(bool(NUMBER.search(guide_text))),
        # specificity
        "number_density": len(NUMBER.findall(narrative)) / n,
        "dollar_density": len(DOLLAR.findall(narrative)) / n,
        "percent_density": len(PERCENT.findall(narrative)) / n,
        "range_count": len(RANGE.findall(narrative)),
        # stored, not used by the registered model
        "uncertainty_share": counts.get("uncertainty", 0) / n,
        "litigious_share": counts.get("litigious", 0) / n,
        "weak_modal_share": counts.get("weak_modal", 0) / n,
        "strong_modal_share": counts.get("strong_modal", 0) / n,
        "nongaap_density": len(NONGAAP.findall(narrative)) / n,
    }


RAW_COLS = ["n_words", "tone", "neg_share", "pos_share", "guide_share", "guide_dir",
            "guide_holds", "guide_numeric", "number_density", "dollar_density",
            "percent_density", "range_count", "uncertainty_share", "litigious_share",
            "weak_modal_share", "strong_modal_share", "nongaap_density", "boiler_share"]
NORMALISE = ["tone", "neg_share", "guide_dir", "guide_share", "number_density",
             "nongaap_density", "n_words", "sim_prev", "sim_year", "days_since_prev"]


def own_history_z(df, cols, min_prior=MIN_PRIOR, window=PRIOR_N):
    """Robust z-score against the company's own EARLIER filings only.

    Median and MAD rather than mean and standard deviation: with a dozen
    prior filings one unusual quarter otherwise shrinks the denominator and
    manufactures a huge score (AMD scored +7.8 sd this way before the fix).
    """
    df = df.sort_values(["cik", "filed_date"])
    out = {}
    for col in cols:
        g = df.groupby("cik")[col]
        roll = lambda f: g.transform(lambda s: s.shift().rolling(window, min_periods=min_prior).apply(f, raw=True))
        centre = roll(np.median)
        spread = roll(lambda a: np.median(np.abs(a - np.median(a))) * 1.4826)
        z = (df[col] - centre) / spread.replace(0, np.nan)
        out[f"{col}_z"] = z.clip(-WINSOR, WINSOR)
    return pd.DataFrame(out, index=df.index)


def trailing_percentile(dates, values, window_days=WINDOW, min_history=200):
    """Rank against all companies' filings in the trailing window, strictly
    earlier — the same rule trader/signals/sue.py uses."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    v = np.asarray(values, dtype=float)
    order = np.argsort(d, kind="stable")
    d_s, v_s = d[order], v[order]
    starts = np.searchsorted(d_s, d_s - np.timedelta64(window_days, "D"), side="left")
    _, first = np.unique(d_s, return_index=True)
    bounds = np.append(first, len(d_s))
    out_s = np.full(len(v_s), np.nan)
    for f, last in zip(bounds[:-1], bounds[1:]):
        hist = np.sort(v_s[starts[f]:f][~np.isnan(v_s[starts[f]:f])])
        if len(hist) < min_history:
            continue
        x = v_s[f:last]
        out_s[f:last] = np.where(np.isnan(x), np.nan,
                                 np.searchsorted(hist, x, side="left") / len(hist))
    out = np.empty_like(out_s)
    out[order] = out_s
    return out


def main():
    lex = load_words()
    print(f"Word lists: " + ", ".join(f"{k} {len(v):,}" for k, v in lex.items()))

    conn = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True, timeout=60)
    rows = conn.execute("""SELECT f.accession, f.cik, f.symbol, f.filed_date, f.accepted_at, f.body
                           FROM filings f WHERE f.body IS NOT NULL""").fetchall()
    conn.close()
    print(f"Parsing {len(rows):,} press releases …", flush=True)

    records, vectors = [], []
    for i, (accession, cik, symbol, filed, accepted, body) in enumerate(rows, 1):
        text = zlib.decompress(body).decode("utf-8")
        rec = {"accession": accession, "cik": cik, "symbol": symbol,
               "filed_date": filed, "accepted_at": accepted}
        rec.update(raw_features(text, lex))
        records.append(rec)
        vectors.append(hashed_vector(TOKEN.findall(split_narrative(text)[0].lower())))
        if i % 2000 == 0:
            print(f"   {i:,}/{len(rows):,}", flush=True)
    df = pd.DataFrame(records)
    df["filed_date"] = pd.to_datetime(df.filed_date)
    vectors = np.vstack(vectors)

    print("Measuring how much each release was rewritten …")
    df["sim_prev"], df["sim_year"] = language_change(df, vectors)

    # How late this release is by the company's own standards. Fiscal quarter
    # ends are not in the filing, but the gap since the company's previous
    # earnings release is, and a company stretching its usual 91 days is the
    # same signal.
    df = df.sort_values(["cik", "filed_date"])
    df["days_since_prev"] = df.groupby("cik").filed_date.diff().dt.days
    df = df.reset_index(drop=True)

    print("Normalising against each company's own history …")
    df = df.sort_values(["cik", "filed_date"]).reset_index(drop=True)
    df = pd.concat([df, own_history_z(df, NORMALISE)], axis=1)

    print("Ranking against the trailing cross-section …")
    for col in ["tone_z", "guide_dir", "number_density_z"]:
        if col in df:
            df[f"{col}_pct"] = trailing_percentile(df.filed_date, df[col])

    out = sqlite3.connect(DB_FILE, timeout=60)
    df.to_sql("features", out, if_exists="replace", index=False)
    out.close()

    have = df.tone_z.notna().mean()
    print(f"\n{len(df):,} filings featurised; {have * 100:.0f}% have enough own history "
          f"for a normalised score")
    print(df[["tone_z", "guide_dir", "sim_prev", "sim_year", "days_since_prev_z",
              "nongaap_density_z"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
