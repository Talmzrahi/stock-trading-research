# ═══════════════════════════════════════════════════════════════════════
#  v3: is there ANY tradable signal in the reader? DEVELOPMENT DATA ONLY.
#
#  Design: research/design_sentiment_v3.md. The readers explain the
#  announcement reaction better than the surprise alone, but explaining
#  what the market already did is not money. This is a small, fixed
#  battery of checks for something that predicts what happens AFTER.
#
#  Everything is computed from features already saved by
#  research/v3_readers_eval.py (v3.db reader_features), out-of-fold.
#
#  Three things are tried, each at 5, 20 and 60 sessions from the gap
#  entry (the next close after the reaction is complete):
#    gap        predicted reaction minus actual: the market under-reacted
#    score      the predicted reaction itself: continuation
#    top-5%     does the text sort the trades the live strategy takes
#
#  THE COUNT MATTERS. This is 3 x 3 + 3 = 12 tests on data that has been
#  looked at many times. Anything that survives is a candidate for the
#  one-shot S&P 400/600 exam, not a finding.
#
#    python research/v3_signal_probe.py
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
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
from v3_readers_eval import N_FOLDS, out_of_fold  # noqa: E402

V3_DB    = ROOT / "data" / "v3.db"
HORIZONS = [5, 20, 60]
BASE     = ["sue_w", "conviction"]


def load(cfg):
    conn = sqlite3.connect(f"file:{V3_DB}?mode=ro", uri=True, timeout=60)
    f = pd.read_sql_query("SELECT * FROM reader_features", conn)
    lab = pd.read_sql_query("SELECT accession, gap_entry_idx, symbol, gap_entry_date, conviction "
                            "FROM labels", conn).drop_duplicates("accession")
    conn.close()
    d = f.merge(lab, on="accession", suffixes=("", "_lab"))
    market = load_market(cfg)
    px = market.closes.ffill()
    col = {k: i for i, k in enumerate(px.columns)}
    a, n = px.to_numpy(), len(market.cal)
    g = d.gap_entry_idx.to_numpy()
    c, b = d.symbol.map(col).to_numpy(), col[cfg.benchmark]
    for h in HORIZONS:
        ok = g + h < n
        d[f"fwd{h}"] = np.nan
        d.loc[ok, f"fwd{h}"] = ((a[g[ok] + h, c[ok]] / a[g[ok], c[ok]] - 1)
                                - (a[g[ok] + h, b] / a[g[ok], b] - 1))
    d["quarter"] = pd.PeriodIndex(pd.to_datetime(d.gap_entry_date), freq="Q").astype(str)
    return d


def spread(values, rank_on, quarters, n=5):
    """Top minus bottom fifth, with standard errors clustered by quarter."""
    q = pd.qcut(pd.Series(rank_on).rank(method="first"), n, labels=False).to_numpy()
    sel = (q == 0) | (q == n - 1)
    y, top = np.asarray(values)[sel], (q[sel] == n - 1).astype(float)
    fit = sm.OLS(y, sm.add_constant(top)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(np.asarray(quarters)[sel])[0]},
        use_t=True)
    return float(fit.params[1]), float(fit.pvalues[1])


def main():
    cfg = load_config()
    d = load(cfg)
    text_cols = [c for c in d.columns if any(c.startswith(p) for p in
                 ("distilroberta_fin", "finbert", "minilm", "spread", "lm_tone"))]
    d["score"] = out_of_fold(d, d[BASE + text_cols].to_numpy(float), d.reaction.to_numpy())
    d["gap"] = d.score - d.reaction
    print(f"{len(d):,} development releases, {d.gap_entry_date.min()} → {d.gap_entry_date.max()}")
    print(f"text features used: {len(text_cols)}")

    print("\n── Does the reader predict what happens after? (top minus bottom fifth) ──")
    print(f"   {'signal':<34}" + "".join(f"{f'{h} sessions':>22}" for h in HORIZONS))
    for name, col in [("gap (market under-reacted)", "gap"), ("score (continuation)", "score")]:
        cells = []
        for h in HORIZONS:
            ok = d[f"fwd{h}"].notna()
            diff, p = spread(d.loc[ok, f"fwd{h}"], d.loc[ok, col], d.loc[ok, "quarter"])
            cells.append(f"{diff * 100:+.2f}pp (p={p:.2f})")
        print(f"   {name:<34}" + "".join(f"{c:>22}" for c in cells))

    top = d[d.conviction >= cfg.cutoff]
    print(f"\n── Among the top-{round((1 - cfg.cutoff) * 100)}% surprise events the strategy trades "
          f"({len(top):,} here) ──")
    for h in HORIZONS:
        ok = top[f"fwd{h}"].notna()
        if ok.sum() < 100:
            print(f"   {h} sessions: only {int(ok.sum())} events — too few")
            continue
        diff, p = spread(top.loc[ok, f"fwd{h}"], top.loc[ok, "score"], top.loc[ok, "quarter"])
        print(f"   {h:>2} sessions: better-read half minus worse {diff * 100:+.2f}pp (p={p:.2f}), "
              f"mean {top.loc[ok, f'fwd{h}'].mean() * 100:+.2f}pp")

    print("\n12 tests on data already looked at many times. Nothing here is evidence:")
    print("anything that survives is a candidate for the one-shot exam.")


if __name__ == "__main__":
    main()
