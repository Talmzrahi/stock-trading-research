# ═══════════════════════════════════════════════════════════════════════
#  v3 layer 1 comparison, step 1: five free readers score the same text.
#
#  Design: research/design_sentiment_v3.md ("Reader comparison"). The
#  sample is 8,000 mature S&P 500 releases (development data only), fixed
#  by seed. It began at 2,000; measured power showed 2,000 detects even a
#  real text effect only 8% of the time (85% at 8,000), so it was grown to
#  8,000 as a superset, and nothing already scored was wasted.
#
#  What gets read is each release's NEW and EDITED sentences from layer 0,
#  in document order. Boilerplate and template sentences are not read.
#
#  Readers, fastest first so results arrive early. The default run is the
#  owner's choice of three (2026-09-19); the other two stay available:
#    minilm             all-MiniLM-L6-v2: a 384-number "what is this about"
#                       fingerprint per sentence (mean-pooled, unit length)
#    distilroberta_fin  lighter financial-mood model: 3 logits
#    finbert            ProsusAI/finbert: 3 logits
#    finbert_tone       v1's second model: 3 logits
#    twitter_roberta    v1's third model: 3 logits
#
#  Mood logits are stored RAW, in the fixed order [positive, negative,
#  neutral], with no temperature, no bias and no averaging. v1's
#  adjustments were cosmetic, and averaging destroys the disagreement
#  between models that later layers want to test.
#
#  Everything is stored as it is produced (the owner's instruction: keep
#  the data from this test):
#    v3.db reader_sample   the exact sample, also exported to
#                          data_archive/v3_reader_sample.csv.gz
#    v3.db reader_out      one row per (reader, release): float16 array
#                          of shape (sentences, dims). Resumable: a sleep
#                          or crash loses at most one chunk.
#
#    python research/v3_readers.py            the default three, resumable
#    python research/v3_readers.py minilm     one reader
# ═══════════════════════════════════════════════════════════════════════

import json
import sqlite3
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")

ROOT     = Path(__file__).resolve().parent.parent
V3_DB    = ROOT / "data" / "v3.db"
ARCHIVE  = ROOT / "data_archive" / "v3_reader_sample.csv.gz"
SAMPLE_N = 8000
SEED     = 20260919
BATCH    = 64
MAX_TOK  = 128          # changed sentences: median 34 tokens, 90th percentile 63
CHUNK    = 25           # releases per commit
THREADS  = 6

READERS = {
    "minilm":            ("sentence-transformers/all-MiniLM-L6-v2", "embed"),
    "distilroberta_fin": ("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", "mood"),
    "finbert":           ("ProsusAI/finbert", "mood"),
    "finbert_tone":      ("ldeb/solved-finbert-tone", "mood"),
    "twitter_roberta":   ("cardiffnlp/twitter-roberta-base-sentiment-latest", "mood"),
}
DEFAULT = ["minilm", "distilroberta_fin", "finbert"]


def changed_sentences(units_blob):
    """The sentences layer 1 reads: new and edited, in document order."""
    return [u[5] for u in json.loads(zlib.decompress(units_blob))
            if u[1] == "s" and u[2] in ("new", "edited")]


def draw_sample(conn):
    """SAMPLE_N mature releases with a readable release and a finite
    reaction. Stored, so every reader sees the same releases. If a smaller
    sample already exists it is kept whole and topped up with a second
    seeded draw from the remaining candidates, so it only ever grows."""
    have = conn.execute("SELECT name FROM sqlite_master WHERE name='reader_sample'").fetchone()
    old = (pd.read_sql_query("SELECT * FROM reader_sample", conn) if have
           else pd.DataFrame(columns=["accession"]))
    if len(old) >= SAMPLE_N:
        return old.sort_values("accession").reset_index(drop=True)
    cand = pd.read_sql_query(
        """SELECT l.accession, l.event_key, l.symbol, l.entry_date, l0.n_prior, l0.units
           FROM labels l JOIN layer0 l0 USING (accession)
           WHERE l0.n_prior >= 4 AND l.readable = 1""", conn)
    cand["n_sent"] = [len(changed_sentences(b)) for b in cand.units]
    cand = cand[cand.n_sent > 0].drop(columns="units").drop_duplicates("accession")
    cand = cand[~cand.accession.isin(old.accession)]
    extra = cand.sample(n=SAMPLE_N - len(old), random_state=SEED + len(old))
    sample = pd.concat([old, extra], ignore_index=True).sort_values("accession")
    sample.to_sql("reader_sample", conn, index=False, if_exists="replace")
    conn.commit()
    ARCHIVE.parent.mkdir(exist_ok=True)
    sample.to_csv(ARCHIVE, index=False, compression="gzip")
    return sample.reset_index(drop=True)


def label_order(model):
    """Indices of [positive, negative, neutral] in this model's output."""
    lab = {i: v.lower() for i, v in model.config.id2label.items()}
    pick = lambda word: next(i for i, v in lab.items() if word in v)
    return [pick("pos"), pick("neg"), pick("neu")]


def reader(name):
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    repo, kind = READERS[name]
    tok = AutoTokenizer.from_pretrained(repo)
    if kind == "embed":
        model = AutoModel.from_pretrained(repo).eval()

        def run(texts):
            enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAX_TOK, padding=True)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(emb, dim=1).numpy()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        order = label_order(model)

        def run(texts):
            enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAX_TOK, padding=True)
            with torch.no_grad():
                return model(**enc).logits[:, order].numpy()
    return run


def score(conn, name, sample):
    done = {a for (a,) in conn.execute("SELECT accession FROM reader_out WHERE reader = ?", (name,))}
    todo = [a for a in sample.accession if a not in done]
    if not todo:
        print(f"{name}: already complete", flush=True)
        return
    run = reader(name)
    t0, n_sent = time.time(), 0
    print(f"{name}: {len(todo):,} releases to score", flush=True)
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        texts, owner = [], []
        for a in chunk:
            (blob,) = conn.execute("SELECT units FROM layer0 WHERE accession = ?", (a,)).fetchone()
            s = changed_sentences(blob)
            texts += s
            owner += [a] * len(s)
        order = np.argsort([len(t) for t in texts])        # similar lengths pad less
        out = [None] * len(texts)
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            res = run([texts[j] for j in idx])
            for j, r in zip(idx, res):
                out[j] = r
        owner = np.array(owner)
        for a in chunk:
            arr = np.stack([out[j] for j in np.flatnonzero(owner == a)]).astype(np.float16)
            conn.execute("INSERT OR REPLACE INTO reader_out VALUES (?,?,?,?,?)",
                         (name, a, arr.shape[0], arr.shape[1], zlib.compress(arr.tobytes(), 6)))
        conn.commit()
        n_sent += len(texts)
        rate = n_sent / (time.time() - t0)
        left = sum(sample.set_index("accession").loc[todo[start + CHUNK:], "n_sent"]) / rate
        print(f"   {name}: {min(start + CHUNK, len(todo)):,}/{len(todo):,} releases, "
              f"{rate:.0f} sentences/s, ~{left / 60:.0f} min left", flush=True)


def main():
    torch.set_num_threads(THREADS)
    names = sys.argv[1:] or DEFAULT
    conn = sqlite3.connect(V3_DB, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS reader_out (
                        reader TEXT, accession TEXT, n INTEGER, dim INTEGER, data BLOB,
                        PRIMARY KEY (reader, accession))""")
    sample = draw_sample(conn)
    print(f"Sample: {len(sample):,} releases, {int(sample.n_sent.sum()):,} changed sentences "
          f"({sample.entry_date.min()} → {sample.entry_date.max()})", flush=True)
    for name in names:
        score(conn, name, sample)
    conn.close()


if __name__ == "__main__":
    main()
