# ═══════════════════════════════════════════════════════════════════════
#  v3 end-to-end prototype, DEVELOPMENT DATA ONLY (S&P 500).
#
#  Design: research/design_sentiment_v3.md. A cheap first pass through
#  every layer, to see whether the ideas have any life before the
#  expensive reader (layer 1) is built. Nothing here is evidence: it is
#  exploratory, on data that will be looked at many times. The S&P
#  400/600 exam is the only test that counts.
#
#  Layer 1 (cheap stand-in): Loughran-McDonald tone and guidance words,
#     measured separately on each layer-0 class of sentence
#  Layer 3: ridge predicting the announcement reaction from the surprise
#     alone, and from surprise + text, walk-forward by year (each year is
#     predicted by a model fitted only on earlier years)
#  Layer 4: gap = predicted − actual reaction; does it predict the drift
#     from the gap entry (next close) over 60 sessions?
#
#  Built-in placebo: tone on BOILERPLATE sentences (repeated verbatim from
#  earlier releases) should explain nothing. If it does, the method is
#  picking up something other than news.
#
#    python research/v3_prototype.py
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
import statsmodels.api as sm
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from trader.backtest import load_market  # noqa: E402
from trader.config import load_config  # noqa: E402
from edgar_features import CUTS, GUIDANCE, RAISES  # noqa: E402

V3_DB   = ROOT / "data" / "v3.db"
WORDS   = ROOT / "data_archive" / "loughran_mcdonald.json.gz"
HOLD    = 60
FIRST_Y = 2013          # predictions start once three years of history exist
MATURE  = 4             # releases need a year of own history for layer 0 classes
TOKEN   = re.compile(r"[a-z']+")
GROUPS  = {"all": {"boilerplate", "template", "edited", "new"},
           "changed": {"edited", "new"},
           "boilerplate": {"boilerplate"}}


def text_features(units, lex):
    """Tone and guidance direction per group of sentences."""
    pos, neg = lex["positive"], lex["negative"]
    out = {}
    for g, classes in GROUPS.items():
        sents = [u[5] for u in units if u[1] == "s" and u[2] in classes]
        toks = [t for s in sents for t in TOKEN.findall(s.lower())]
        p, n = sum(t in pos for t in toks), sum(t in neg for t in toks)
        out[f"tone_{g}"] = (p - n) / (p + n + 1)
        guide = " ".join(s for s in sents if GUIDANCE.search(s))
        up, down = len(RAISES.findall(guide)), len(CUTS.findall(guide))
        out[f"guide_{g}"] = (up - down) / (up + down + 1)
    n_s = sum(u[1] == "s" for u in units)
    out["share_new"] = sum(u[1] == "s" and u[2] == "new" for u in units) / max(n_s, 1)
    out["share_edited"] = sum(u[1] == "s" and u[2] == "edited" for u in units) / max(n_s, 1)
    return out


def load(cfg):
    conn = sqlite3.connect(f"file:{V3_DB}?mode=ro", uri=True)
    labels = pd.read_sql_query("SELECT * FROM labels", conn)
    lex = {k: set(v) for k, v in json.load(gzip.open(WORDS, "rt", encoding="utf-8")).items()}
    rows = []
    for accession, n_prior, blob in conn.execute(
            "SELECT accession, n_prior, units FROM layer0 WHERE n_prior >= ?", (MATURE,)):
        units = json.loads(zlib.decompress(blob))
        rows.append({"accession": accession, **text_features(units, lex)})
    conn.close()
    d = labels.merge(pd.DataFrame(rows), on="accession")

    market = load_market(cfg)
    px = market.closes.ffill()
    col = {s: i for i, s in enumerate(px.columns)}
    a = px.to_numpy()
    d = d[d.gap_entry_idx + HOLD < len(market.cal)].copy()
    g, c, b = d.gap_entry_idx.to_numpy(), d.symbol.map(col).to_numpy(), col[cfg.benchmark]
    d["drift"] = (a[g + HOLD, c] / a[g, c] - 1) - (a[g + HOLD, b] / a[g, b] - 1)
    d["year"] = pd.to_datetime(d.entry_date).dt.year
    d["quarter"] = pd.PeriodIndex(pd.to_datetime(d.gap_entry_date), freq="Q").astype(str)
    d = d[d.readable & np.isfinite(d.drift) & np.isfinite(d.reaction)]
    # Winsorise the surprise: price-scaled SUE has extreme tails that would
    # otherwise dominate a linear fit.
    lo, hi = d.sue.quantile([0.01, 0.99])
    d["sue_w"] = d.sue.clip(lo, hi)
    return d.reset_index(drop=True)


def walk_forward(d, cols, alpha=1.0):
    """Out-of-sample prediction of the reaction, each year from earlier years."""
    pred = pd.Series(np.nan, index=d.index)
    for y in sorted(d.year.unique()):
        if y < FIRST_Y:
            continue
        tr, te = d.year < y, d.year == y
        X, Xt = d.loc[tr, cols].to_numpy(float), d.loc[te, cols].to_numpy(float)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
        A = Z.T @ Z + alpha * np.eye(Z.shape[1])
        A[-1, -1] -= alpha
        beta = np.linalg.solve(A, Z.T @ d.loc[tr, "reaction"].to_numpy())
        pred[te] = np.column_stack([(Xt - mu) / sd, np.ones(len(Xt))]) @ beta
    return pred


def by_quarter(values, flag, quarters):
    """Difference in means, flagged minus not, SE clustered by quarter."""
    y = np.asarray(values, float)
    X = sm.add_constant(np.asarray(flag, float))
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(quarters)[0]},
                           use_t=True)
    return float(fit.params[1]), float(fit.pvalues[1])


def main():
    cfg = load_config()
    d = load(cfg)
    oos = d[d.year >= FIRST_Y].copy()
    print(f"{len(d):,} mature S&P 500 events with release, reaction and drift "
          f"({d.entry_date.min()} → {d.entry_date.max()}); out-of-sample from {FIRST_Y}: {len(oos):,}")

    # ── Idea 1: does text explain the market's reaction beyond the numbers? ──
    base = ["sue_w", "conviction"]
    specs = {"surprise only": base,
             "+ tone/guidance on ALL sentences": base + ["tone_all", "guide_all"],
             "+ tone/guidance on CHANGED sentences": base + ["tone_changed", "guide_changed",
                                                             "share_new", "share_edited"],
             "+ PLACEBO: tone/guidance on BOILERPLATE": base + ["tone_boilerplate",
                                                                "guide_boilerplate"]}
    print("\n── Idea 1: out-of-sample fit to the announcement reaction ──")
    print(f"   {'model':<42}{'R²':>8}{'rank corr':>11}{'gain vs surprise (p, by year)':>32}")
    preds = {}
    for name, cols in specs.items():
        p = walk_forward(d, cols)[oos.index]
        preds[name] = p
        r = oos.reaction
        r2 = 1 - ((r - p) ** 2).sum() / ((r - r.mean()) ** 2).sum()
        rc = stats.spearmanr(p, r)[0]
        if name == "surprise only":
            gain = ""
        else:
            err_b = (r - preds["surprise only"]) ** 2
            err_m = (r - p) ** 2
            per_year = (err_b - err_m).groupby(oos.year).mean()
            gain = f"{per_year.mean() * 1e4:+.2f}bp² (p={stats.ttest_1samp(per_year, 0).pvalue:.3f})"
        print(f"   {name:<42}{r2:>8.4f}{rc:>11.3f}{gain:>32}")

    # ── Idea 2: does the gap predict the drift? ──
    print("\n── Idea 2: gap = predicted − actual reaction → 60-session drift from the next close ──")
    for name in ["surprise only", "+ tone/guidance on CHANGED sentences"]:
        gap = preds[name] - oos.reaction
        q = pd.qcut(gap.rank(method="first"), 5, labels=False)
        means = oos.drift.groupby(q).mean() * 100
        top, bot = q == 4, q == 0
        sel = top | bot
        diff, p = by_quarter(oos.drift[sel], top[sel], oos.quarter[sel])
        rc = stats.spearmanr(gap, oos.drift)[0]
        print(f"   {name}")
        print(f"      drift by gap quintile (low→high): "
              + "  ".join(f"{m:+.2f}" for m in means) + " pp")
        print(f"      highest − lowest {diff * 100:+.2f}pp (p={p:.3f}, quarter-clustered); "
              f"rank corr {rc:+.3f}")
    text_part = preds["+ tone/guidance on CHANGED sentences"] - preds["surprise only"]
    q = pd.qcut(text_part.rank(method="first"), 5, labels=False)
    top, bot = q == 4, q == 0
    diff, p = by_quarter(oos.drift[top | bot], top[top | bot], oos.quarter[top | bot])
    print(f"   text's own contribution to the gap (changed-sentence model minus surprise-only):")
    print(f"      highest − lowest quintile drift {diff * 100:+.2f}pp (p={p:.3f}, quarter-clustered)")


if __name__ == "__main__":
    main()
