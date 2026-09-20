# ═══════════════════════════════════════════════════════════════════════
#  What is actually new in an earnings release (layer 0).
#
#  Companies repeat most of each release every quarter: contacts, the
#  company description, safe harbor, table labels, and the sentence that
#  says "revenue was $X, up Y%". Only what changed can carry news the
#  market has not already seen in the numbers.
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
#  Both the research bulk run (research/v3_layer0.py) and the daily run
#  import from here, so the live pipeline classifies exactly as the one
#  the model was fitted on. No returns or labels are touched: this is
#  text bookkeeping, safe to run on any release including the exam set.
# ═══════════════════════════════════════════════════════════════════════

import re
import zlib
from collections import deque

import numpy as np
from scipy import sparse

PRIOR_K  = 4        # compare with the previous year of releases
EDITED   = 0.45     # word-pair Jaccard for "edited": the low point of the sentence
                    # similarity histogram (research/design_sentiment_v3.md)
HASH_DIM = 1 << 20
CLASSES  = ("boilerplate", "template", "edited", "new")

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


def fingerprint(release_blocks):
    """What a release contributes to its company's history."""
    us = units(release_blocks)
    keys = [template_key(t) for _, _, t in us]
    return {"exact": {exact_key(t) for _, _, t in us}, "keys": set(keys),
            "pairs": [word_pairs(k) for k in keys]}


def classify(release_blocks, history, edited=EDITED):
    """Label every unit of one release against `history`, a sequence of
    fingerprints of that company's earlier releases (oldest first, at most
    PRIOR_K of them). Returns [block, kind, class, similarity, seen_before
    in this release, text] — the layer 0 record."""
    us = units(release_blocks)
    exact = [exact_key(t) for _, _, t in us]
    keys = [template_key(t) for _, _, t in us]
    pairs = [word_pairs(k) for k in keys]

    seen_exact = set().union(*(h["exact"] for h in history)) if history else set()
    seen_keys = set().union(*(h["keys"] for h in history)) if history else set()
    sim = max_similarity(pairs, [p for h in history for p in h["pairs"]])

    rows, in_doc = [], set()
    for (blk, kind, text), e, k, s in zip(us, exact, keys, sim):
        if e in seen_exact:
            cls, s = "boilerplate", 1.0
        elif k in seen_keys:
            cls, s = "template", 1.0
        elif s >= edited:
            cls = "edited"
        else:
            cls = "new"
        rows.append([blk, kind, cls, round(float(s), 3), k in in_doc, text])
        in_doc.add(k)
    return rows


def classify_company(releases, edited=EDITED, prior_k=PRIOR_K):
    """releases: [(accession, filed_date, blocks)] for one company, any
    order. One record per release, oldest first. Only strictly earlier
    releases count as history."""
    history = deque(maxlen=prior_k)
    out = []
    for n_prior, (accession, filed, rb) in enumerate(sorted(releases, key=lambda r: r[1])):
        rows = classify(rb, list(history), edited)
        counts = {c: sum(r[2] == c for r in rows) for c in CLASSES}
        out.append({"accession": accession, "filed_date": filed, "n_prior": n_prior,
                    "n_units": len(rows), **{f"n_{c}": counts[c] for c in CLASSES},
                    "units": rows})
        history.append(fingerprint(rb))
    return out


def changed_sentences(units_rows, cap=None):
    """The sentences layer 1 reads: new and edited, in document order,
    capped at `cap` because the news sits at the top of a release."""
    out = [u[5] for u in units_rows if u[1] == "s" and u[2] in ("new", "edited")]
    return out[:cap] if cap else out
