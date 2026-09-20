# ═══════════════════════════════════════════════════════════════════════
#  v3 layer 1 comparison, step 2: which reader (or combination) helps?
#
#  Design: research/design_sentiment_v3.md ("Reader comparison").
#  DEVELOPMENT DATA ONLY: the 2,000-release S&P 500 sample scored by
#  research/v3_readers.py. Exploratory: readers are compared here, and the
#  winner still has to pass the pre-registered exam on S&P 400/600.
#
#  Yardstick (idea 1): out-of-sample improvement in predicting the
#  announcement reaction beyond the earnings surprise. Folds are blocks of
#  years (grouped K-fold), so a reader is never scored on years it was
#  fitted on. Significance is by year: 16 paired differences in squared error.
#
#  Features per release, from its new + edited sentences:
#    mood readers   mean(p_pos - p_neg), share negative, share positive
#    minilm         a sentence-level map from the 384-number fingerprint to
#                   the reaction the surprise did not explain, learned on
#                   the training folds' ~85,000 sentences and averaged per
#                   release (the market labels each sentence)
#    disagreement   (owner's idea) each mood model's distance from the
#                   others, and the spread across models per sentence
#    bias-corrected (owner's idea) each mood model's score minus what it
#                   usually says about this kind of sentence (fitted from
#                   the fingerprint on training folds)
#
#  Idea 2, exploratory and underpowered at 2,000 events: does the gap
#  predict drift, and is it bigger where the readers disagree?
#
#  Stored (owner's instruction): per-release features -> v3.db
#  reader_features + data_archive/v3_reader_features.csv.gz; the printed
#  report -> data_archive/v3_readers_report.txt
#
#    python research/v3_readers_eval.py
# ═══════════════════════════════════════════════════════════════════════

import io
import json
import re
import sqlite3
import sys
import zlib
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from trader.backtest import load_market  # noqa: E402
from trader.config import load_config  # noqa: E402
from v3_readers import CAP, READERS, changed_sentences  # noqa: E402

V3_DB    = ROOT / "data" / "v3.db"
WORDS    = ROOT / "data_archive" / "loughran_mcdonald.json.gz"
FEATURES = ROOT / "data_archive" / "v3_reader_features.csv.gz"
REPORT   = ROOT / "data_archive" / "v3_readers_report.txt"
HOLD     = 60
N_FOLDS  = 5
MOODS    = [m for m, (_, kind) in READERS.items() if kind == "mood"]
TOKEN    = re.compile(r"[a-z']+")


# ── loading ────────────────────────────────────────────────────────────

def load_outputs(conn, sample):
    """{reader: {accession: array}} for readers that finished the sample."""
    out = {}
    for name in READERS:
        rows = conn.execute("SELECT accession, n, dim, data FROM reader_out WHERE reader = ?",
                            (name,)).fetchall()
        if len(rows) < len(sample):
            continue
        out[name] = {a: np.frombuffer(zlib.decompress(d), dtype=np.float16)
                     .reshape(n, dim).astype(np.float32)[:CAP] for a, n, dim, d in rows}
    return out


def base_table(conn, cfg):
    s = pd.read_sql_query("SELECT accession FROM reader_sample", conn)
    lab = pd.read_sql_query("SELECT * FROM labels", conn)
    d = s.merge(lab.drop_duplicates("accession"), on="accession")
    market = load_market(cfg)
    px = market.closes.ffill()
    col = {k: i for i, k in enumerate(px.columns)}
    a, n = px.to_numpy(), len(market.cal)
    g = d.gap_entry_idx.to_numpy()
    ok = g + HOLD < n
    c, b = d.symbol.map(col).to_numpy(), col[cfg.benchmark]
    d["drift"] = np.nan
    d.loc[ok, "drift"] = ((a[g[ok] + HOLD, c[ok]] / a[g[ok], c[ok]] - 1)
                          - (a[g[ok] + HOLD, b] / a[g[ok], b] - 1))
    lo, hi = d.sue.quantile([0.01, 0.99])
    d["sue_w"] = d.sue.clip(lo, hi)
    d["year"] = pd.to_datetime(d.entry_date).dt.year
    d["quarter"] = pd.PeriodIndex(pd.to_datetime(d.gap_entry_date), freq="Q").astype(str)
    years = sorted(d.year.unique())
    blocks = np.array_split(years, N_FOLDS)
    d["fold"] = d.year.map({y: i for i, blk in enumerate(blocks) for y in blk})
    return d.reset_index(drop=True)


def lm_tone(conn, d):
    lex = {k: set(v) for k, v in json.load(__import__("gzip").open(WORDS, "rt", encoding="utf-8")).items()}
    tone = []
    for a in d.accession:
        (blob,) = conn.execute("SELECT units FROM layer0 WHERE accession = ?", (a,)).fetchone()
        toks = [t for s in changed_sentences(blob) for t in TOKEN.findall(s.lower())]
        p, n = sum(t in lex["positive"] for t in toks), sum(t in lex["negative"] for t in toks)
        tone.append((p - n) / (p + n + 1))
    return np.array(tone)


# ── fitting ────────────────────────────────────────────────────────────

def ridge(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    A[-1, -1] -= alpha
    beta = np.linalg.solve(A, Z.T @ y)
    return lambda Xn: np.column_stack([(Xn - mu) / sd, np.ones(len(Xn))]) @ beta


def out_of_fold(d, X, y, alpha=10.0):
    pred = np.full(len(d), np.nan)
    for f in range(N_FOLDS):
        tr, te = (d.fold != f).to_numpy(), (d.fold == f).to_numpy()
        pred[te] = ridge(X[tr], y[tr], alpha)(X[te])
    return pred


ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]


class FoldSums:
    """Sufficient statistics per outer fold for a sentence-level ridge, so a
    fit on any union of folds is a 384x384 solve, not a pass over ~100k
    sentences. Embeddings are unit-length, so they are centred but not scaled."""

    def __init__(self, E, y, fold_of_row):
        self.k = sorted(set(fold_of_row))
        self.s = {f: self._stats(E[fold_of_row == f], y[fold_of_row == f]) for f in self.k}

    @staticmethod
    def _stats(X, y):
        return X.T @ X, X.T @ y, X.sum(0), y.sum(), len(y)

    def fit(self, folds, alpha):
        G, h, sx, sy, n = (sum(self.s[f][i] for f in folds) for i in range(5))
        mx, my = sx / n, sy / n
        beta = np.linalg.solve(G - n * np.outer(mx, mx) + alpha * np.eye(len(mx)), h - n * mx * my)
        return lambda X: (X - mx) @ beta + my


def sentence_map(E, y, owner, release_fold, target_release):
    """Out-of-fold, per-release average of a sentence-level ridge map.

    The penalty is chosen by inner cross-validation over the training folds,
    scored on release-level error: a release's sentences share one label,
    so a fixed penalty overfits badly (seen on a 325-release dry run)."""
    fold_of_row = release_fold[owner]
    sums = FoldSums(E, y, fold_of_row)
    out = np.full(len(release_fold), np.nan)
    chosen = []
    for f in sums.k:
        train = [g for g in sums.k if g != f]
        err = {}
        for a in ALPHAS:
            e = []
            for g in train:
                rows = fold_of_row == g
                p = sums.fit([h for h in train if h != g], a)(E[rows])
                rel = pd.Series(p).groupby(owner[rows]).mean()
                e.append(((target_release[rel.index] - rel.to_numpy()) ** 2).mean())
            err[a] = np.mean(e)
        best = min(err, key=err.get)
        chosen.append(best)
        rows = fold_of_row == f
        p = sums.fit(train, best)(E[rows])
        rel = pd.Series(p).groupby(owner[rows]).mean()
        out[rel.index] = rel.to_numpy()
    return out, chosen


def sentence_features(d, outs):
    """Fold-aware features that need sentence-level fitting: the minilm map
    to the unexplained reaction, and each mood model's bias correction."""
    feats = {}
    if "minilm" not in outs:
        return feats
    emb = [outs["minilm"][a] for a in d.accession]
    owner = np.repeat(np.arange(len(d)), [len(e) for e in emb])
    E = np.vstack(emb)
    fold = d.fold.to_numpy()
    base = out_of_fold(d, d[["sue_w", "conviction"]].to_numpy(float), d.reaction.to_numpy())
    resid = d.reaction.to_numpy() - base
    # the market labels each sentence with its release's unexplained reaction
    feats["minilm_map"], chosen = sentence_map(E, resid[owner], owner, fold, resid)
    print(f"   minilm map: penalty chosen per fold {chosen}")
    for m in MOODS:
        if m not in outs:
            continue
        net = np.concatenate([mood_net(outs[m][a]) for a in d.accession])
        net_rel = pd.Series(net).groupby(owner).mean().to_numpy()
        # what this reader usually says about this kind of sentence
        habit, _ = sentence_map(E, net, owner, fold, net_rel)
        feats[f"{m}_biascorr"] = net_rel - habit
    return feats


def mood_net(logits):
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return p[:, 0] - p[:, 1]          # stored order: positive, negative, neutral


def release_features(d, outs):
    f = pd.DataFrame(index=d.index)
    nets = {}
    for m in MOODS:
        if m not in outs:
            continue
        per = [outs[m][a] for a in d.accession]
        nets[m] = [mood_net(x) for x in per]
        f[f"{m}_mean"] = [x.mean() for x in nets[m]]
        f[f"{m}_neg"] = [(x.argmax(1) == 1).mean() for x in per]
        f[f"{m}_pos"] = [(x.argmax(1) == 0).mean() for x in per]
    have = list(nets)
    if len(have) >= 2:
        stacked = [np.vstack([nets[m][i] for m in have]) for i in range(len(d))]
        f["spread"] = [s.std(0).mean() for s in stacked]
        for j, m in enumerate(have):
            f[f"{m}_dev"] = [(s[j] - np.delete(s, j, 0).mean(0)).mean() for s in stacked]
    return f


# ── reporting ──────────────────────────────────────────────────────────

def compare(d, f):
    base = ["sue_w", "conviction"]
    moods = [m for m in MOODS if f"{m}_mean" in f]
    specs = {"surprise only": [], "+ word-list tone (prototype's reader)": ["lm_tone"]}
    for m in moods:
        specs[f"+ {m}"] = [f"{m}_mean", f"{m}_neg", f"{m}_pos"]
    if "minilm_map" in f:
        specs["+ minilm (market-labelled fingerprint)"] = ["minilm_map"]
    all_moods = [c for m in moods for c in (f"{m}_mean", f"{m}_neg", f"{m}_pos")]
    if len(moods) >= 2:
        specs["+ all mood readers"] = all_moods
        specs["+ all mood readers + disagreement"] = all_moods + ["spread"] + [f"{m}_dev" for m in moods]
    if "minilm_map" in f and moods:
        specs["+ all mood readers + minilm"] = all_moods + ["minilm_map"]
        bias = [f"{m}_biascorr" for m in moods if f"{m}_biascorr" in f]
        dis = (["spread"] + [f"{m}_dev" for m in moods]) if len(moods) >= 2 else []
        specs["+ everything (moods, minilm, disagreement, bias-corrected)"] = (
            all_moods + ["minilm_map"] + dis + bias)

    X_all = pd.concat([d, f], axis=1)
    y = d.reaction.to_numpy()
    preds, rows = {}, []
    for name, extra in specs.items():
        X = X_all[base + extra].to_numpy(float)
        preds[name] = out_of_fold(d, X, y)
    err0 = (y - preds["surprise only"]) ** 2
    print(f"\n── Idea 1: out-of-fold fit to the announcement reaction ({len(d):,} releases, "
          f"{N_FOLDS} year-block folds) ──")
    print(f"   {'model':<62}{'R²':>7}{'rank':>7}{'gain vs surprise, by year':>30}")
    for name, p in preds.items():
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rc = stats.spearmanr(p, y)[0]
        if name == "surprise only":
            g = ""
        else:
            per_year = pd.Series(err0 - (y - p) ** 2).groupby(d.year).mean()
            g = f"{per_year.mean() * 1e4:+.2f}bp² (p={stats.ttest_1samp(per_year, 0).pvalue:.3f})"
        rows.append((name, r2, rc))
        print(f"   {name:<62}{r2:>7.4f}{rc:>7.3f}{g:>30}")
    return preds


def gap_tests(d, f, preds):
    ok = d.drift.notna().to_numpy()
    best = max((k for k in preds if k != "surprise only"),
               key=lambda k: -np.nanmean((d.reaction - preds[k]) ** 2))
    print(f"\n── Idea 2 (exploratory, ~±1pp noise at this size): gap → 60-session drift ──")
    for name in ["surprise only", best]:
        gap = pd.Series(preds[name] - d.reaction.to_numpy())[ok]
        drift, qtr = d.drift[ok], d.quarter[ok]
        q = pd.qcut(gap.rank(method="first"), 5, labels=False)
        means = drift.groupby(q.values).mean() * 100
        sel = ((q == 0) | (q == 4)).to_numpy()
        fit = sm.OLS(drift.to_numpy()[sel], sm.add_constant((q == 4).to_numpy()[sel].astype(float))).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(qtr.to_numpy()[sel])[0]}, use_t=True)
        print(f"   {name}: drift by gap quintile " + " ".join(f"{m:+.2f}" for m in means)
              + f" pp; top − bottom {fit.params[1] * 100:+.2f}pp (p={fit.pvalues[1]:.3f})")
    if "spread" in f:
        gap = preds[best] - d.reaction.to_numpy()
        amb = (f.spread > f.spread.median()).to_numpy()
        for label, mask in [("readers AGREE (low spread)", ~amb), ("readers DISAGREE (high spread)", amb)]:
            m = ok & mask
            rc = stats.spearmanr(gap[m], d.drift[m])[0]
            print(f"   {label:<32} gap vs drift rank corr {rc:+.3f}  (n={m.sum():,})")
    return best


def main():
    cfg = load_config()
    conn = sqlite3.connect(f"file:{V3_DB}?mode=ro", uri=True, timeout=60)
    sample = pd.read_sql_query("SELECT accession FROM reader_sample", conn)
    outs = load_outputs(conn, sample)
    d = base_table(conn, cfg)
    d["lm_tone"] = lm_tone(conn, d)
    conn.close()

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"Readers complete on the sample: {', '.join(outs) or 'none'}")
        print(f"Sample: {len(d):,} S&P 500 releases, {d.entry_date.min()} → {d.entry_date.max()}")
        f = release_features(d, outs)
        for k, v in sentence_features(d, outs).items():
            f[k] = v
        preds = compare(d, f)
        gap_tests(d, f, preds)
    text = buf.getvalue()
    print(text)

    REPORT.write_text(text, encoding="utf-8")
    table = pd.concat([d[["accession", "event_key", "entry_date", "reaction", "drift", "sue_w",
                          "conviction", "lm_tone", "fold"]], f], axis=1)
    table.to_csv(FEATURES, index=False, compression="gzip")
    out = sqlite3.connect(V3_DB, timeout=60)
    table.to_sql("reader_features", out, if_exists="replace", index=False)
    out.close()
    print(f"Saved: {REPORT.name}, {FEATURES.name}, v3.db reader_features")


if __name__ == "__main__":
    main()
